"""Sensor-surface (soiling / damage) occlusion injectors for RGB cameras.

Failure Mode 3: sensor-surface occlusion applied to the lens or its cover.
A contaminant or crack sits on one camera's optics: it is sensor-local, affects only that
camera, needs no scene-geometry consistency, and so composes cleanly with the multi-platform
(drone + ground) setting.

Contaminant kinds:
    "dirt"    soft accumulated blobs (mud / dust), procedural
    "scratch" thin linear grooves, procedural
    "crack"   branching glass fracture. By default this loads a real crack image bundled with
              the package (assets/crack.png) and overlays it with a RANDOM affine transform
              (scale, rotation, flip, position) so each use is different -- not a static overlay.
              If the bundled image is absent it falls back to a procedural fracture. Pass
              texture_path="" to force the procedural generator.

Compositing models (independent of the kind / geometry source):

  model="baseline"      Faithful reproduction of Occluded nuScenes (arXiv:2510.18552):
                          dirt:          I' = clip(I + alpha * M, 0, 255)   (additive)
                          scratch/crack: I' = (1 - a) * I + a * S           (alpha over)
  model="transmission"  Physically-grounded transmission-veiling model (Koschmieder form):
                          I' = t * B + (1 - t) * A,   t = 1 - rho * (M * h_sigma)
                        With A = 0 an opaque contaminant can only attenuate (I' <= I). t(x) is a
                        per-pixel channel multiplier, so t_obj (mean transmission over object
                        pixels) is a scene-comparable, information-meaningful severity unit.

Texture overlay (the paper's method) is used automatically for "crack"; you can also point any
kind at a real PNG via texture_path. A real alpha channel is used when present; otherwise the
mask is derived from luminance with auto-detected polarity. The overlay is randomly scaled,
rotated, flipped and positioned per call (texture_scale_range / texture_rotate / texture_flip,
and placement for the position), so a single PNG yields varied cracks across frames; with
temporal="persistent" the transform is fixed across a clip (one physical crack on the lens).

Severity is always the strength/extent knob:
  severity_mode="opacity"   per-pixel strength: baseline -> alpha, transmission -> rho.
  severity_mode="coverage"  target object-coverage c in [0,1]; procedural density / texture SIZE
                            is searched to hit c (rho from `opacity`, A from `veiling`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

Box = Tuple[float, float, float, float]
_DEFAULT_CRACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "crack.png")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class OcclusionConfig:
    kind: str = "dirt"                 # "dirt" | "scratch" | "crack"
    model: str = "transmission"        # "baseline" | "transmission"
    severity_mode: str = "opacity"     # "opacity" | "coverage"
    severity: float = 0.3              # strength (opacity mode) OR target coverage c

    opacity: float = 1.0               # rho used in COVERAGE mode (in opacity mode rho = severity)
    veiling: float = 0.0               # A in [0,1]: 0 = dark (opaque), >0 = bright haze/scatter
    defocus_sigma: float = 4.0         # near-aperture defocus PSF radius (px)
    placement: str = "random"          # "random" | "targeted" | "avoid"
    temporal: str = "iid"              # "iid" | "persistent"

    # procedural geometry
    n_blobs: int = 14
    blob_scale: float = 0.06
    scratch_width: float = 1.6
    n_impacts: int = 1
    bottom_bias: float = 0.0

    # texture overlay
    texture_path: Optional[str] = None     # None -> packaged default for crack; "" -> force procedural
    texture_invert: Optional[bool] = None  # None = auto polarity; True if mark is dark on light
    texture_flip: bool = True              # random horizontal flip
    texture_rotate: bool = True            # random rotation (0..360 deg)
    texture_scale_range: Tuple[float, float] = (0.4, 0.9)  # crack max-extent as fraction of min(H,W)

    coverage_tol: float = 0.03
    coverage_max_iter: int = 18
    seed: int = 42

    def __post_init__(self) -> None:
        if self.kind not in ("dirt", "scratch", "crack"):
            raise ValueError(f"kind must be 'dirt', 'scratch' or 'crack', got {self.kind!r}")
        if self.model not in ("baseline", "transmission"):
            raise ValueError(f"model must be 'baseline' or 'transmission', got {self.model!r}")
        if self.severity_mode not in ("opacity", "coverage"):
            raise ValueError(f"severity_mode must be 'opacity' or 'coverage', got {self.severity_mode!r}")
        if self.placement not in ("random", "targeted", "avoid"):
            raise ValueError(f"placement must be 'random', 'targeted' or 'avoid', got {self.placement!r}")
        if self.temporal not in ("iid", "persistent"):
            raise ValueError(f"temporal must be 'iid' or 'persistent', got {self.temporal!r}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        # crack defaults to the packaged real texture (random affine per use); "" forces procedural
        if self.kind == "crack" and self.texture_path is None and os.path.exists(_DEFAULT_CRACK):
            self.texture_path = _DEFAULT_CRACK


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _to_float01(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0, True
    img = image.astype(np.float32)
    if img.max() > 1.5:
        return img / 255.0, False
    return img, False


def _from_float01(img: np.ndarray, was_uint8: bool, like: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    if was_uint8 or like.dtype == np.uint8:
        return (img * 255.0 + 0.5).astype(np.uint8)
    if like.max() > 1.5:
        return (img * 255.0).astype(like.dtype)
    return img.astype(like.dtype)


def _gray(img01: np.ndarray) -> np.ndarray:
    if img01.ndim == 2:
        return img01
    return img01[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _blur_color(img01: np.ndarray, sigma: float) -> np.ndarray:
    if img01.ndim == 2:
        return gaussian_filter(img01, sigma)
    return np.stack([gaussian_filter(img01[..., c], sigma) for c in range(img01.shape[2])], axis=-1)


def _directional_blur(img01: np.ndarray, theta: float, length: int) -> np.ndarray:
    from scipy.ndimage import convolve
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), dtype=np.float32)
    cx = length // 2
    for i in range(length):
        off = i - cx
        x = int(round(cx + off * np.cos(theta)))
        y = int(round(cx + off * np.sin(theta)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    k /= max(k.sum(), 1.0)
    if img01.ndim == 2:
        return convolve(img01, k, mode="reflect")
    out = np.empty_like(img01)
    for c in range(img01.shape[2]):
        out[..., c] = convolve(img01[..., c], k, mode="reflect")
    return out


def _im(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
def _sample_centers(n, h, w, rng, placement, boxes, bottom_bias) -> np.ndarray:
    if placement == "targeted" and boxes:
        areas = np.array([max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes])
        probs = areas / areas.sum()
        pts = [(rng.uniform(boxes[i][0], boxes[i][2]), rng.uniform(boxes[i][1], boxes[i][3]))
               for i in rng.choice(len(boxes), size=n, p=probs)]
        return np.array(pts, dtype=np.float32)
    if placement == "avoid" and boxes:
        pts, tries = [], 0
        while len(pts) < n and tries < n * 50:
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            if not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes):
                pts.append((x, y))
            tries += 1
        while len(pts) < n:
            pts.append((rng.uniform(0, w), rng.uniform(0, h)))
        return np.array(pts, dtype=np.float32)
    x = rng.uniform(0, w, size=n)
    y = h * rng.beta(1.0, 1.0 + 3.0 * bottom_bias, size=n) if bottom_bias > 0 else rng.uniform(0, h, size=n)
    return np.stack([x, y], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Procedural density maps (in [0, 1])
# --------------------------------------------------------------------------- #
def _dirt_density(shape, cfg, rng, boxes, density_mul) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = max(1, int(round(cfg.n_blobs * density_mul)))
    centers = _sample_centers(n, h, w, rng, cfg.placement, boxes, cfg.bottom_bias)
    base_r = cfg.blob_scale * min(h, w)
    M = np.zeros((h, w), dtype=np.float32)
    for cx, cy in centers:
        r = base_r * rng.uniform(0.5, 1.8)
        M += rng.uniform(0.5, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r * r))
    return np.clip(M, 0.0, 1.0)


def _scratch_density(shape, cfg, rng, boxes, density_mul) -> Tuple[np.ndarray, float]:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = max(1, int(round(cfg.n_blobs * density_mul)))
    S = np.zeros((h, w), dtype=np.float32)
    thetas = []
    for _ in range(n):
        if cfg.placement == "targeted" and boxes:
            b = boxes[rng.integers(len(boxes))]
            cx, cy = rng.uniform(b[0], b[2]), rng.uniform(b[1], b[3])
            length = rng.uniform(0.1, 0.3) * min(h, w)
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            length = rng.uniform(0.2, 0.6) * min(h, w)
        theta = rng.uniform(0, np.pi)
        thetas.append(theta)
        nx, ny = -np.sin(theta), np.cos(theta)
        tx, ty = np.cos(theta), np.sin(theta)
        perp = np.abs((xx - cx) * nx + (yy - cy) * ny)
        along = (xx - cx) * tx + (yy - cy) * ty
        prof = np.exp(-(perp ** 2) / (2.0 * cfg.scratch_width ** 2)) * (np.abs(along) < length / 2.0)
        S = np.maximum(S, prof.astype(np.float32))
    return np.clip(S, 0.0, 1.0), float(np.mean(thetas)) if thetas else 0.0


def _crack_density(shape, cfg, rng, boxes, density_mul) -> np.ndarray:
    h, w = shape
    diag = float(np.hypot(h, w))
    step = max(3.0, 0.012 * diag)
    impacts = _sample_centers(max(1, int(round(cfg.n_impacts))), h, w, rng, cfg.placement, boxes, cfg.bottom_bias)
    segs = []

    def grow(x, y, ang, length, depth, inten):
        for _ in range(max(2, int(length / step))):
            ang += rng.normal(0.0, 0.18)
            nx, ny = x + step * np.cos(ang), y + step * np.sin(ang)
            segs.append((x, y, nx, ny, inten))
            x, y = nx, ny
            if not (0 <= x < w and 0 <= y < h):
                break
            if depth > 0 and rng.random() < 0.13:
                grow(x, y, ang + rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.0),
                     length * 0.55, depth - 1, inten * 0.85)

    n_radial = max(3, int(round(6 * density_mul + 3)))
    for ix, iy in impacts:
        base_len = (0.16 + 0.12 * density_mul) * diag
        for a in rng.uniform(0, 2 * np.pi) + np.linspace(0, 2 * np.pi, n_radial, endpoint=False):
            grow(ix, iy, a + rng.normal(0, 0.1), base_len * rng.uniform(0.6, 1.2), 2, 1.0)
        for _ in range(int(rng.integers(1, 3))):
            rr = rng.uniform(0.02, 0.08) * diag
            a0, span = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.5)
            pa = np.linspace(a0, a0 + span, 12)
            px, py = ix + rr * np.cos(pa), iy + rr * np.sin(pa)
            for i in range(len(pa) - 1):
                segs.append((px[i], py[i], px[i + 1], py[i + 1], 0.7))

    canvas = Image.new("F", (w, h), 0.0)
    draw = ImageDraw.Draw(canvas)
    for x0, y0, x1, y1, inten in segs:
        draw.line([(x0, y0), (x1, y1)], fill=float(inten), width=1)
    return np.clip(gaussian_filter(np.asarray(canvas, dtype=np.float32), 0.6), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Texture overlay (real PNG) with randomized affine transform
# --------------------------------------------------------------------------- #
_TEX_NATIVE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _extract_texture(cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Native-resolution contaminant mask M and colour map S from the texture image.

    The crack mask uses the alpha channel ONLY when alpha is a genuine sparse cutout
    (transparent background). A fully-opaque alpha -- a crack image with a solid matte
    background, which is common for downloaded overlays -- must NOT be used as the mask,
    or the whole texture rectangle becomes opaque and you get a solid block instead of
    thin cracks. In that case (and for plain RGB) the mask is derived from luminance with
    auto-detected polarity, which isolates the crack lines.
    """
    path = cfg.texture_path
    if path in _TEX_NATIVE:
        return _TEX_NATIVE[path]
    # convert("RGBA") normalises any source mode (palette 'P', grayscale 'L', 'RGB', 'RGBA')
    # to 4 channels with real colours + alpha. A raw palette load returns indices, not colours.
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    use_alpha = (arr[..., 3] < 0.9).mean() > 0.02   # >2% meaningfully transparent = real cutout
    if use_alpha:
        M, S = arr[..., 3], lum
    else:
        invert = cfg.texture_invert
        if invert is None:
            invert = (lum > 0.8).mean() > 0.5            # bright background -> dark crack
        if invert:                                       # dark mark on light background
            bg = float(np.percentile(lum, 90)); T = bg - 0.12
            M = np.clip((T - lum) / (T - float(lum.min()) + 1e-6), 0.0, 1.0)
            S = np.full_like(lum, 0.92)
        else:                                            # bright mark on dark background
            bg = float(np.percentile(lum, 10)); T = bg + 0.12
            M = np.clip((lum - T) / (float(lum.max()) - T + 1e-6), 0.0, 1.0)
            S = lum
    res = (M.astype(np.float32), S.astype(np.float32))
    _TEX_NATIVE[path] = res
    return res


def _sample_texture_transform(shape, cfg, rng, boxes) -> Dict:
    h, w = shape
    return {
        "angle": float(rng.uniform(0, 360)) if cfg.texture_rotate else 0.0,
        "base_scale": float(rng.uniform(*cfg.texture_scale_range)),
        "flip": bool(cfg.texture_flip and rng.random() < 0.5),
        "center": tuple(map(float, _sample_centers(1, h, w, rng, cfg.placement, boxes, cfg.bottom_bias)[0])),
    }


def _apply_texture(M_nat, S_nat, shape, cfg, tf, size_mul) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate / scale / flip / paste the native crack onto a full frame at the sampled transform."""
    h, w = shape
    Mi, Si = _im(M_nat), _im(S_nat)
    if tf["angle"]:
        Mi = Mi.rotate(tf["angle"], resample=Image.BILINEAR, expand=True, fillcolor=0)
        Si = Si.rotate(tf["angle"], resample=Image.BILINEAR, expand=True, fillcolor=0)
    target = max(4.0, tf["base_scale"] * size_mul * min(h, w))
    cw, ch = Mi.size
    s = target / max(cw, ch)
    nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
    Mi, Si = Mi.resize((nw, nh), Image.BILINEAR), Si.resize((nw, nh), Image.BILINEAR)
    Mr, Sr = _arr(Mi), _arr(Si)
    if tf["flip"]:
        Mr, Sr = Mr[:, ::-1].copy(), Sr[:, ::-1].copy()
    cx, cy = tf["center"]
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    Mf, Sf = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x0 + nw), min(h, y0 + nh)
    if dx1 > dx0 and dy1 > dy0:
        sx0, sy0 = dx0 - x0, dy0 - y0
        Mf[dy0:dy1, dx0:dx1] = Mr[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        Sf[dy0:dy1, dx0:dx1] = Sr[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return np.clip(Mf, 0.0, 1.0), np.clip(Sf, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# PSF / coverage utilities
# --------------------------------------------------------------------------- #
def _psf(density, cfg) -> np.ndarray:
    if cfg.model == "transmission" and cfg.defocus_sigma > 0:
        return np.clip(gaussian_filter(density, cfg.defocus_sigma), 0.0, 1.0)
    return density


def _binary_mask(density, thr=0.05) -> np.ndarray:
    return density > thr


def _box_pixels(shape, boxes):
    if not boxes:
        return None
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        xa, ya, xb, yb = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = True
    return m


def _coverage(mask, boxes) -> float:
    bp = _box_pixels(mask.shape, boxes)
    if bp is None:
        return float(mask.mean())
    return float((mask & bp).sum() / bp.sum()) if bp.sum() else 0.0


def _transmission_obj_mean(t, boxes) -> float:
    bp = _box_pixels(t.shape, boxes)
    if bp is None or bp.sum() == 0:
        return float(t.mean())
    return float(t[bp].mean())


# --------------------------------------------------------------------------- #
# Injector
# --------------------------------------------------------------------------- #
@dataclass
class SensorOcclusionInjector:
    cfg: OcclusionConfig
    _rng: np.random.Generator = field(init=False, repr=False)
    _cached: Optional[Dict] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)
        self._cached = None

    # -- procedural density (+ coverage targeting) -------------------------- #
    def _make_proc(self, shape, boxes, mul):
        r = np.random.default_rng(self._rng.integers(0, 2**32 - 1))
        if self.cfg.kind == "dirt":
            return _dirt_density(shape, self.cfg, r, boxes, mul), 0.0, None
        if self.cfg.kind == "scratch":
            dens, theta = _scratch_density(shape, self.cfg, r, boxes, mul)
            return dens, theta, None
        return _crack_density(shape, self.cfg, r, boxes, mul), 0.0, None

    def _build_procedural(self, shape, boxes):
        cfg = self.cfg
        if cfg.severity_mode == "opacity":
            return self._make_proc(shape, boxes, 1.0)
        target, lo, hi = cfg.severity, 0.0, 1.0
        dens, theta, s = self._make_proc(shape, boxes, hi)
        tries = 0
        while _coverage(_binary_mask(_psf(dens, cfg)), boxes) < target and hi < 8.0 and tries < cfg.coverage_max_iter:
            hi *= 1.8
            dens, theta, s = self._make_proc(shape, boxes, hi)
            tries += 1
        best = (dens, theta, s)
        for _ in range(cfg.coverage_max_iter):
            mid = 0.5 * (lo + hi)
            dens, theta, s = self._make_proc(shape, boxes, mid)
            best = (dens, theta, s)
            c = _coverage(_binary_mask(_psf(dens, cfg)), boxes)
            if abs(c - target) <= cfg.coverage_tol:
                break
            lo, hi = (mid, hi) if c < target else (lo, mid)
        return best

    # -- texture overlay (fixed random transform per build, size varies for coverage) -- #
    def _build_texture(self, shape, boxes):
        cfg = self.cfg
        r = np.random.default_rng(self._rng.integers(0, 2**32 - 1))
        M_nat, S_nat = _extract_texture(cfg)
        tf = _sample_texture_transform(shape, cfg, r, boxes)   # one transform per build
        if cfg.severity_mode == "opacity":
            M, S = _apply_texture(M_nat, S_nat, shape, cfg, tf, 1.0)
            return M, 0.0, S
        target, lo, hi = cfg.severity, 0.0, 1.0

        def cov(mul):
            M, S = _apply_texture(M_nat, S_nat, shape, cfg, tf, mul)
            return _coverage(_binary_mask(_psf(M, cfg)), boxes), M, S

        c, M, S = cov(hi)
        tries = 0
        while c < target and hi < 8.0 and tries < cfg.coverage_max_iter:
            hi *= 1.8
            c, M, S = cov(hi)
            tries += 1
        best = (M, S)
        for _ in range(cfg.coverage_max_iter):
            mid = 0.5 * (lo + hi)
            c, M, S = cov(mid)
            best = (M, S)
            if abs(c - target) <= cfg.coverage_tol:
                break
            lo, hi = (mid, hi) if c < target else (lo, mid)
        return best[0], 0.0, best[1]

    def _build_density(self, shape, boxes):
        if self.cfg.texture_path:
            return self._build_texture(shape, boxes)
        return self._build_procedural(shape, boxes)

    # -- compositing -------------------------------------------------------- #
    def _composite(self, img01, density, theta, s_color=None):
        cfg = self.cfg
        is_color = img01.ndim == 3
        if cfg.model == "baseline":
            if cfg.kind == "dirt":
                M = density * _gray(img01)
                add = cfg.severity * M
                out = np.clip(img01 + (add[..., None] if is_color else add), 0.0, 1.0)
            else:
                a = cfg.severity * density
                aa = a[..., None] if is_color else a
                S = (s_color[..., None] if is_color else s_color) if s_color is not None else np.float32(0.92)
                out = (1.0 - aa) * img01 + aa * S
            t = 1.0 - np.clip(cfg.severity * density, 0.0, 1.0)
            return out, t

        rho = cfg.severity if cfg.severity_mode == "opacity" else cfg.opacity
        Md = _psf(density, cfg)
        t = np.clip(1.0 - rho * Md, 0.0, 1.0)
        A = np.float32(cfg.veiling)
        if cfg.kind == "scratch":
            B = _directional_blur(img01, theta, int(0.04 * min(img01.shape[:2])))
        elif cfg.kind == "crack":
            B = _blur_color(img01, 1.2)
        else:
            B = img01
        tt = t[..., None] if is_color else t
        return np.clip(tt * B + (1.0 - tt) * A, 0.0, 1.0), t

    # -- public API --------------------------------------------------------- #
    def build_mask(self, image_shape, gt_boxes_2d=None) -> Dict:
        shape = image_shape[:2]
        if self.cfg.temporal == "persistent" and self._cached is not None and self._cached["shape"] == shape:
            return self._cached
        density, theta, s_color = self._build_density(shape, gt_boxes_2d)
        rec = {"shape": shape, "density": density, "theta": theta, "s_color": s_color}
        if self.cfg.temporal == "persistent":
            self._cached = rec
        return rec

    def __call__(self, image, gt_boxes_2d=None) -> Tuple[np.ndarray, Dict]:
        img01, was_u8 = _to_float01(image)
        rec = self.build_mask(image.shape, gt_boxes_2d)
        out01, t = self._composite(img01, rec["density"], rec["theta"], rec.get("s_color"))
        out = _from_float01(out01, was_u8, image)
        mask = _binary_mask(_psf(rec["density"], self.cfg))
        info = {
            "kind": self.cfg.kind, "model": self.cfg.model,
            "source": "texture" if self.cfg.texture_path else "procedural",
            "placement": self.cfg.placement, "temporal": self.cfg.temporal,
            "severity_mode": self.cfg.severity_mode, "severity": self.cfg.severity,
            "veiling_A": self.cfg.veiling,
            "coverage_achieved": _coverage(mask, gt_boxes_2d),
            "mask_area_frac": float(mask.mean()),
            "transmission_obj_mean": _transmission_obj_mean(t, gt_boxes_2d),
            "transmission_frame_mean": float(t.mean()),
            "seed": self.cfg.seed,
        }
        return out, info
