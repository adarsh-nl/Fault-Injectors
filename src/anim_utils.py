"""
anim_utils.py
-------------
Shared helpers for rendering fault-injection animations.

This module centralises the pieces that were previously copy-pasted across
``render_fault_animations.py`` and the visualisation notebooks (Parts 5/7/9):
image loading + downsampling, intrinsic scaling, the camera-stream registry,
the bird's-eye-view axis styling, the video-writer factory (with an
ffmpeg -> pillow fallback), and two generic multi-panel animation builders.

Every renderer in the project should call these instead of redefining them, so
there is exactly one implementation of each to maintain and test.
"""

from __future__ import annotations

import os
import glob
from typing import Callable, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DARK_BG = "#0d0d0d"


# --------------------------------------------------------------------------- #
# Camera registry / file discovery
# --------------------------------------------------------------------------- #
def camera_specs(veh_root: str, drone_root: Optional[str] = None) -> dict:
    """Map stream name -> (root, sensor) for every camera.

    e.g. ``{'vehicle_front': (veh_root, 'front'), ..., 'drone_bottom': (drone_root, 'bottom')}``.
    """
    spec = {f"vehicle_{c}": (veh_root, c) for c in ("front", "back", "left", "right")}
    if drone_root:
        spec.update({f"drone_{c}": (drone_root, c)
                     for c in ("front", "back", "left", "right", "bottom")})
    return spec


def camera_image_files(root: str, sensor: str) -> List[str]:
    """Sorted list of PNG frames for one camera stream."""
    return sorted(glob.glob(os.path.join(root, "camera", sensor, "*.png")))


# --------------------------------------------------------------------------- #
# Image loading / intrinsics
# --------------------------------------------------------------------------- #
def load_image_ds(path: str, ds: int = 1) -> np.ndarray:
    """Load an RGB image, optionally downsampled by an integer factor ``ds``."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if ds > 1:
        w, h = img.size
        img = img.resize((w // ds, h // ds), Image.BILINEAR)
    return np.array(img)


def scale_intrinsics(K: np.ndarray, ds: int) -> np.ndarray:
    """Scale a 3x3 intrinsic matrix to match a ``ds``-times downsampled image."""
    if ds == 1:
        return K.copy()
    Ks = K.copy()
    Ks[0, :] /= ds
    Ks[1, :] /= ds
    return Ks


# --------------------------------------------------------------------------- #
# Writer / saving
# --------------------------------------------------------------------------- #
def make_writer(fps: int, bitrate: int = 3000, title: str = "Griffin fault injection"):
    """Return ``(ext, writer)``: an MP4 FFMpegWriter if ffmpeg is available,
    otherwise a GIF PillowWriter so the notebooks render without ffmpeg."""
    from matplotlib.animation import FFMpegWriter, PillowWriter
    try:
        return "mp4", FFMpegWriter(fps=fps, bitrate=bitrate, metadata={"title": title})
    except Exception:
        return "gif", PillowWriter(fps=fps)


def save_animation(anim: FuncAnimation, fig, out_stem: str, fps: int,
                   outdir: str, dpi: int = 100, **writer_kw) -> str:
    """Save an animation with the best available writer; return the written path."""
    os.makedirs(outdir, exist_ok=True)
    ext, writer = make_writer(fps, **writer_kw)
    path = os.path.join(outdir, f"{out_stem}.{ext}")
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# BEV axis styling
# --------------------------------------------------------------------------- #
def style_bev_ax(ax, rng: float = 60) -> None:
    """Apply the standard dark BEV styling and draw the ego marker at the origin."""
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_aspect("equal")
    ax.tick_params(colors="#555", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.plot(0, 0, "w^", markersize=8, zorder=10)


# --------------------------------------------------------------------------- #
# Generic multi-panel animations
# --------------------------------------------------------------------------- #
def _resolve_titles(axes, titles):
    """Set static panel titles now; return per-axis title artists for dynamic ones."""
    for j, ax in enumerate(axes):
        if titles and titles[j] is not None and not callable(titles[j]):
            ax.set_title(titles[j], color="white", fontsize=10)
    return [ax.title for ax in axes]


def animate_image_panels(panels: Sequence[Sequence[np.ndarray]], n_frames: int,
                         out_stem: str, outdir: str, fps: int = 10,
                         titles: Optional[List] = None,
                         suptitle_fn: Optional[Callable[[int], str]] = None,
                         figsize_per=(5.0, 4.5)) -> str:
    """Animate K side-by-side image sequences and save to ``outdir``.

    panels      : K sequences, each length ``n_frames``, of HxWx3 arrays.
    titles      : optional list of K entries; each a str (static) or k->str (dynamic).
    suptitle_fn : optional k -> suptitle string.
    """
    K = len(panels)
    fig, axes = plt.subplots(1, K, figsize=(figsize_per[0] * K, figsize_per[1]),
                             facecolor=DARK_BG, squeeze=False)
    axes = list(axes[0])
    ims = [ax.imshow(panels[j][0], aspect="auto") for j, ax in enumerate(axes)]
    for ax in axes:
        ax.axis("off")
    title_objs = _resolve_titles(axes, titles)
    sup = fig.suptitle("", color="white", fontsize=12, fontfamily="monospace")

    def update(k):
        for j in range(K):
            ims[j].set_data(panels[j][k])
            if titles and callable(titles[j]):
                title_objs[j].set_text(titles[j](k))
        if suptitle_fn:
            sup.set_text(suptitle_fn(k))
        return ims + title_objs + [sup]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    return save_animation(anim, fig, out_stem, fps, outdir)


def animate_bev_panels(pts_panels: Sequence[Sequence[np.ndarray]], n_frames: int,
                       out_stem: str, outdir: str, fps: int = 10, rng: float = 60,
                       titles: Optional[List] = None,
                       suptitle_fn: Optional[Callable[[int], str]] = None,
                       cmap: str = "plasma", vmin: float = -2, vmax: float = 6) -> str:
    """Animate K side-by-side bird's-eye-view scatter sequences (ego frame)."""
    K = len(pts_panels)
    fig, axes = plt.subplots(1, K, figsize=(7 * K, 7), facecolor=DARK_BG, squeeze=False)
    axes = list(axes[0])
    scs = [ax.scatter([], [], c=[], cmap=cmap, s=1.2, vmin=vmin, vmax=vmax) for ax in axes]
    for ax in axes:
        style_bev_ax(ax, rng)
    title_objs = _resolve_titles(axes, titles)
    sup = fig.suptitle("", color="white", fontsize=12, fontfamily="monospace")

    def _set(sc, p):
        if len(p) > 0:
            sc.set_offsets(p[:, :2])
            sc.set_array(p[:, 2])
        else:
            sc.set_offsets(np.empty((0, 2)))
            sc.set_array(np.array([]))

    def update(k):
        for j in range(K):
            _set(scs[j], pts_panels[j][k])
            if titles and callable(titles[j]):
                title_objs[j].set_text(titles[j](k))
        if suptitle_fn:
            sup.set_text(suptitle_fn(k))
        return scs + title_objs + [sup]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    return save_animation(anim, fig, out_stem, fps, outdir)
