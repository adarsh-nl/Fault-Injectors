# Sensor-Surface Occlusion Injection for Multi-Platform Cooperative Perception

Mathematical formulation and design rationale for the occlusion fault injector
(`src/fault_injectors/sensor_occlusion.py`).

This document explains, with intuition alongside the mathematics, what the injector does and
why its design departs from prior occlusion-robustness work — in particular the additive /
alpha-blend compositing of *Occluded nuScenes* (Kumar et al., arXiv:2510.18552), the lens-soiling
line (SoilingNet, WoodScape), and the point-cloud corruption benchmarks (Robo3D, Sun et al.).

---

## 1. Scope and motivation

The research question concerns how multi-platform multi-modal fusion (drone + ground vehicle)
preserves **robustness and information quality** under realistic failure modes, of which occlusion
is the central case. Occlusion is the canonical motivation for cooperative perception: the entire
reason a second platform helps is that it can see what is occluded from the first. An occlusion
injector for this setting must therefore satisfy two constraints that single-platform injectors do
not:

1. **Cross-platform consistency.** Whatever degrades one platform must be physically reconcilable
   with what the other platform sees of the *same scene*.
2. **Measurable information loss.** The severity axis must express *how much task-relevant
   information was destroyed*, not merely how opaque a patch is, so it couples to the mutual-information
   instrument used to answer the information-quality question.

Both constraints drive the design choices below.

---

## 2. Taxonomy: occlusion is not sparsification

A frequent and consequential error is to label LiDAR beam reduction or random point dropout as
"occlusion". They are distinct failure modes, and the robustness literature treats them separately
(Robo3D groups *beam missing* under sensor degradation; Sun et al. define `Occlusion` and
`Cutout`/`Local_Density_Dec` as separate corruption types).

Let $R \subseteq \Omega$ be the set of sensor measurements removed by a corruption over the sensing
domain $\Omega$ (pixels, or LiDAR rays).

- **Occlusion** is *spatially structured* removal: $R$ is a contiguous region determined by a
  foreground occluder or a sensor-surface contaminant. All evidence for whatever lies in $R$ is
  removed together. Formally, the removal indicator is highly spatially correlated:
  $\mathrm{Cov}\big(\mathbb{1}[x\in R],\,\mathbb{1}[x'\in R]\big)$ is large for neighbouring $x,x'$.

- **Sparsification** (beam / point reduction) is *spatially unstructured* removal: each measurement
  is thinned roughly independently, $\mathbb{1}[x\in R]\sim\mathrm{Bernoulli}(p)$ i.i.d., so every
  object keeps *some* evidence everywhere.

The distinction is load-bearing: a model robust to uniform thinning can fail catastrophically under
structured occlusion, because occlusion removes **all** evidence for a specific object while
sparsification only reduces density. Conflating them produces a benchmark that does not measure what
it claims to. This injector is strictly an **occlusion** injector; sparsification belongs to a
separate degradation mode.

---

## 3. Why sensor-surface (soiling) occlusion

Occlusion can be introduced at three physical depths, with very different consequences for a
multi-platform setting:

| Depth | Example | Cross-platform property |
|---|---|---|
| Scene-geometry | a phantom car / pole inserted into the scene | **Shared world** — must appear consistently in *every* platform's view |
| Airborne | a dust cloud / fog volume between sensor and scene | **Shared world** — both platforms fly through the same air |
| Sensor-surface | dirt, a scratch, a cracked lens cover | **Sensor-local** — affects only the one camera it sits on |

Inserting a *controllable* scene-geometry occluder requires placing a phantom object and then making
it appear consistently across the drone and ground views — i.e. environment generation — or it
violates the shared-world constraint of Section 1. Airborne particulates have the same problem (and
are properly classified as adverse weather, a different mode). A **sensor-surface contaminant**
sidesteps both: it is local to one sensor's optics, so it needs no world-consistency and no scene
modification. This is grounded in the lens-soiling literature, where soiling is explicitly an
occlusion mode (SoilingNet, WoodScape, Occluded nuScenes).

There is a second, deeper reason. Sensor-surface occlusion is **asymmetric by construction**: a leaf
on the ground camera's lens cannot appear on the drone's camera, because it is a different physical
lens. This *is* the cooperative-recovery experiment — one platform locally blinded, the other clear —
handed to the design for free. Scene-geometry occluders couple to geometry that is not controlled
and can blind both platforms at once, muddying exactly the signal the research question probes.

---

## 4. Clean input

A clean, synchronised sample from platform $i$ at frame $k$ is
$$
X_{i,k} = (I_{i,k},\, P_{i,k}), \qquad
I_{i,k}\in\mathbb{R}^{H\times W\times 3},\quad
P_{i,k}\in\mathbb{R}^{N\times C}.
$$
The occlusion injector acts on the camera stream $I_{i,k}$. Because the contaminant is sensor-local,
the LiDAR $P_{i,k}$ on the same platform — and every sensor on the partner platform — is untouched,
which is precisely the information available for fusion to recover the occluded region.

---

## 5. Compositing models

The injector separates **what** the contaminant is (its coverage map $M\in[0,1]^{H\times W}$, from
dirt blobs, scratch lines, or a real crack image) from **how** it alters the image (the compositing
model). Two models are provided.

### 5.1 Baseline (Occluded nuScenes)

The published baseline composites additively for dirt and with an alpha-over for scratches/cracks:
$$
\text{dirt:}\quad I' = \mathrm{clip}\!\Big(I + \alpha\textstyle\sum_k M_k,\ 0, 255\Big),
\qquad
\text{scratch/crack:}\quad I' = (1-\alpha)\,I + \alpha\,S,
$$
with $\alpha$ a global opacity and $S$ a (near-white) contaminant colour. This is reproduced exactly
so it can serve as an ablation. Its limitations motivate the model that follows:

- **Energy violation.** Additive blending can *increase* radiance ($I' > I$): a contaminant that
  merely sits on the lens cannot add scene-independent light, yet the additive model does.
- **No defocus.** A contaminant millimetres from the sensor is far outside the focal plane and is
  therefore heavily blurred; the baseline produces crisp-edged patches that no real dirty lens shows.
- **Severity has no physical meaning.** $\alpha$ is not comparable across scenes — $\alpha=0.2$ over
  bright sky and over a dense crowd are the "same severity" yet destroy very different information.

### 5.2 Transmission–veiling model (ours)

We adopt the radiative-transfer (Koschmieder) forward model used in atmospheric dehazing, repurposed
as the contaminant-compositing operator:
$$
\boxed{\,I'(x) = t(x)\,B(x) + \big(1 - t(x)\big)\,A(x)\,}, \qquad
t(x) = 1 - \rho\,\big(M * h_\sigma\big)(x).
$$

| Symbol | Meaning |
|---|---|
| $M\in[0,1]$ | contaminant coverage map (dirt / scratch / crack geometry) |
| $h_\sigma$ | **defocus point-spread function** (near-aperture blur) |
| $\rho\in[0,1]$ | opacity: $1$ = opaque (mud, leaf), $<1$ = translucent (dust film) |
| $A\in[0,1]$ | **veiling luminance**: scattered light added by a translucent contaminant ($0$ = dark/opaque, bright = haze/scatter) |
| $t\in[0,1]$ | **transmission**: the fraction of scene radiance reaching the sensor |
| $B$ | background: $I$ for dirt; a directional blur of $I$ for a scratch; an isotropic scatter blur of $I$ for a crack |

Intuition: of the light arriving at pixel $x$, a fraction $t(x)$ is the scene transmitted through the
contaminant and a fraction $1-t(x)$ is light the contaminant *scatters* toward the sensor (the
veiling term $A$). An opaque mud speck transmits nothing and scatters nothing dark; a dust film
transmits partially and scatters bright haze; a crack scatters light along the fracture (so it
appears bright).

This single equation **subsumes the baseline's four separately hand-tuned effects** as special cases:

| Effect | $\rho$ | $A$ | $B$ | Result |
|---|---|---|---|---|
| Opaque dirt / mud | $\to 1$ | $0$ | $I$ | soft darkening $t\,I$ |
| Translucent dust / soiling | mid | bright | $I$ | haze / contrast loss (the dehazing forward model) |
| Scratch | mid | bright | $\text{dir-blur}_\theta(I)$ | bright refractive streak |
| Glass crack | mid | bright | $\text{scatter}(I)$ | bright refractive fracture |

---

## 6. Why the transmission model is the correct equation

Three properties make this more than a cosmetic change.

**(a) Energy conservation.** With $A = 0$ (opaque),
$$
I'(x) = t(x)\,I(x) \le I(x)\quad\forall x,
$$
so an opaque contaminant can only *attenuate*, never brighten — as physics requires. The additive
baseline can produce $I' > I$, which is an energy-conservation violation, not a stylistic difference.

**(b) Near-aperture defocus.** The contaminant sits at distance $d \ll f$ from the sensor, far
outside the depth of field, so its image is the contaminant convolved with a wide defocus kernel
$h_\sigma$. Modelling $M * h_\sigma$ turns crisp masks into the soft, semi-transparent blobs a real
dirty lens produces; omitting it (as the baseline does) is the single most visible inaccuracy.

**(c) Information interpretation — the link to the research question.** $t(x)$ is a **per-pixel
channel multiplier** for the scene-to-pixel channel. Where $t(x)\to 0$, the output collapses to the
constant $A(x)$ and carries essentially *zero* mutual information about the scene at $x$; where
$t(x)\to 1$, the channel is intact. Consequently the expected information loss of an occlusion is a
functional of its transmission map, computable *before* any model is run. The physically-correct
equation and the information-calibrated severity axis (Section 7) are therefore the **same design
move**: parameterise occlusion by transmission $t$, and severity becomes an information quantity.

---

## 7. Severity: opacity vs information-calibrated coverage

The injector exposes two severity modes.

- **Opacity mode.** Severity is the per-pixel strength: $\alpha$ for the baseline, $\rho$ for the
  transmission model. Simple, but — like the baseline — not comparable across scenes.

- **Coverage mode.** Severity is a *target object coverage* $c\in[0,1]$: the fraction of projected
  ground-truth box pixels that are occluded. The contaminant density (procedural) or overlay size
  (texture) is searched until the realised coverage matches $c$. Coverage is computed from the
  projected 3D boxes (ego $\to$ sensor $\to$ pixels), model-free and reproducible.

The reported, scene-comparable severity is the **mean object transmission**
$$
\bar t_{\text{obj}} = \frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}} t(x),
\qquad \mathcal{B} = \bigcup_{\text{objects}} \text{box}(x),
$$
the fraction of object-bearing light that survives the occlusion. Unlike $\alpha$, $\bar t_{\text{obj}}$
is comparable across scenes and platforms, and it is exactly the quantity whose downstream effect the
mutual-information module measures:
$$
\Delta I = I(Z^{\text{clean}};Y) - I(Z^{\text{occ}};Y).
$$
Coverage *targets* information loss in label space; $\Delta I$ *measures* the realised loss in feature
space. Severity should be expressed in the unit that controls information, which is $\bar t_{\text{obj}}$,
not opacity.

(Practical note: coverage mode is appropriate for dirt, whose filled blobs can reach a target
fraction of object pixels. Cracks are thin lines and cannot fill a large fraction of a box, so for
cracks the natural severity axis is opacity.)

---

## 8. Contaminant geometry

### 8.1 Dirt (procedural)
A sum of randomly placed, randomly scaled Gaussian blobs,
$$
M(x) = \min\Big(1,\ \textstyle\sum_j a_j \exp\!\big(-\|x - c_j\|^2 / 2r_j^2\big)\Big),
$$
with centres $c_j$ from the placement model (uniform, optionally bottom-biased to mimic road spray;
or object-targeted) and the count scaled for coverage targeting.

### 8.2 Scratch (procedural)
Thin oriented line profiles: for a scratch through $c$ with tangent $\theta$, the perpendicular
distance gives a Gaussian cross-section of half-width $w$, truncated to the scratch length. In the
transmission model the scratch background $B$ is an **anisotropic** blur along $\theta$, reproducing
the way a real groove smears and scatters light along its length.

### 8.3 Glass crack (real texture by default; procedural fallback)
A crack is hard to synthesise convincingly, so by default the injector overlays a **real crack image**
bundled with the package and supplies the fracture as the geometry $M$. The texture loader:

1. converts any input mode to RGBA (`convert("RGBA")`) so palette, grayscale, RGB and RGBA images
   are handled uniformly — a raw palette-PNG load would return colour *indices*, not colours;
2. uses the **alpha channel** as $M$ only when it is a genuine sparse cutout (a transparent
   background); a fully-opaque alpha (a solid-matte crack image) is *not* used as the mask, or the
   whole rectangle would be opaque;
3. otherwise derives $M$ from luminance with auto-detected polarity (dark crack on light background,
   or bright crack on dark background), isolating the thin fracture lines.

A procedural branching fracture is available as a licence-clean fallback: radial cracks emanate from
one or more impact points, each recursively spawning sub-branches, plus a few concentric arcs near
the impact, rasterised to a thin bright map.

The key point: the *real fracture supplies the geometry*, but it is composited through the
**transmission model** (refractive scatter), not painted on with an alpha-blend as in the baseline.

---

## 9. Dynamic placement

The published method overlays a single stock crack PNG, effectively a static full-frame stamp. That
is both unrealistic (one fixed pattern) and uninformative (no distribution over placements). Instead,
each application draws a random **affine transform** of the native crack into a sub-region of the
frame:
$$
M_{\text{frame}} = \mathrm{paste}_{(c_x,c_y)}\!\Big(\mathrm{flip}\circ\mathrm{scale}_{s}\circ
\mathrm{rotate}_{\phi}\,(M_{\text{native}})\Big),
$$
with rotation $\phi\sim\mathcal{U}[0,2\pi)$, scale $s$ drawn so the crack's extent is a random
fraction of $\min(H,W)$, an optional horizontal flip, and centre $(c_x,c_y)$ from the placement model
(uniform / object-targeted / object-avoiding). A single crack image therefore yields a *distribution*
of cracks across frames and runs, all reproducible under the seed.

---

## 10. Temporal model

A lens contaminant does not change frame-to-frame: one crack sits on the glass while the world flows
past. Two temporal modes capture the relevant regimes:

- **`persistent`** — the transform is sampled once and the mask is fixed for the whole clip. This is
  the physically correct single-drive model, and it is required for honest temporal evaluation: a
  fixed degradation persists, so the right behaviour is for the system to become *more* aware of the
  loss over time, not to be re-surprised each frame.
- **`iid`** — the transform is re-randomised per frame, for sampling many configurations or as data
  augmentation.

The published baseline applies its masks per frame, i.e. implicitly i.i.d.; for scene-level animation
and for any temporal model this is incorrect, and `persistent` is the default for the animations.

---

## 11. Cross-platform asymmetry and recovery

The injector is applied per platform with an explicit correlation knob $\varrho$ between platforms'
occlusions, ranging from independent ($\varrho=0$, complementary occlusion — best case for fusion) to
identical ($\varrho=1$, redundant occlusion — worst case). The contribution metric is the
**information recovery ratio**, measured with the mutual-information module:
$$
R \;=\; \frac{I\big(Z^{\text{fused}};Y\big)\big|_{A\,\text{occluded}} \;-\; I\big(Z^{\text{occ}}_A;Y\big)}
             {I\big(Z^{\text{clean}}_A;Y\big) \;-\; I\big(Z^{\text{occ}}_A;Y\big)}.
$$
$R=1$ means fusion fully recovers the information occlusion destroyed on platform $A$; $R=0$ means
fusion does not help; $R<0$ means *modality conflict* — fusion is worse than the occluded sensor
alone. This turns the qualitative claim "fusion cushions but does not immunize" into a measurable
quantity in nats, cross-platform — the result the injector exists to produce.

---

## 12. Comparison to prior work

| | Occluded nuScenes | SoilingNet / WoodScape | Robo3D / common corruptions | This injector |
|---|---|---|---|---|
| Platforms | single (ego) | single | single | **multi-platform (drone + ground)** |
| Occlusion type | camera soiling + naive LiDAR removal | camera soiling | weather / sensor / sparsity | **sensor-surface, platform-independent** |
| Compositing | additive / alpha-over | alpha overlay | per-corruption | **transmission–veiling (energy-correct, defocused)** |
| Crack source | static stock PNG | — | — | **real PNG, randomized affine; procedural fallback** |
| Severity unit | opacity $\alpha$ | qualitative | discrete levels | **$\bar t_{\text{obj}}$ (object light surviving)** |
| Validation | SSIM (image-space) | classification | mCE / accuracy | **mutual information of the fused representation** |
| Evaluation target | accuracy preservation | accuracy | accuracy | **information recovery $R$ under asymmetry** |

In words: prior work degrades a *single* platform, composites with additive/alpha overlays, grades by
*accuracy* under corruption, and validates "realism" by image distance (SSIM). This injector degrades
*per platform* with a physically grounded, energy-correct, information-meaningful operator, and the
object of study is whether a *second platform* recovers the information one platform loses. The
single-platform benchmarks become the ego-only ablation of the multi-platform case.

---

## 13. Limitations and honest caveats

- **Sim-real gap (the unresolved crux).** No synthetic occlusion has been shown to reproduce the
  *failure distribution* of real soiled sensors; SSIM and visual inspection measure image distance,
  not failure fidelity. Claims are therefore scoped to **relative, cross-platform** comparisons under
  a fixed protocol, where a constant fidelity bias cancels, and rest on a *geometric* recovery
  mechanism (a second viewpoint) that transfers across the sim-real gap better than appearance does.
  Whether robustness *rankings* under synthetic occlusion match rankings under real occluded data is
  the decisive open question.
- **Coverage mode for cracks.** Thin fractures cannot fill a large fraction of object pixels, so
  coverage targeting undershoots for cracks; opacity is the correct severity axis there.
- **ISP constraint.** The injector operates on final 8-bit, ISP-processed images (as does the
  baseline), so raw-sensor and auto-exposure feedback effects are out of scope.
- **Asset licensing.** A real crack texture bundled with the package ships with the code; its licence
  must permit redistribution, or the procedural fracture (licence-clean) should be used.

---

## 14. References

- B. R. Kumar et al., *Occluded nuScenes: A Multi-Sensor Occlusion Robustness Dataset*, arXiv:2510.18552, 2025.
- M. Uricar et al., *SoilingNet: Soiling Detection on Automotive Surround-View Cameras*, IEEE ITSC, 2019.
- S. Yogamani et al., *WoodScape: A Multi-Task, Multi-Camera Fisheye Dataset for Autonomous Driving*, ICCV, 2019.
- L. Kong et al., *Robo3D: Towards Robust and Reliable 3D Perception against Corruptions*, ICCV, 2023.
- J. Sun et al., *Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions*, arXiv:2201.12296.
- Y. Dong et al., *Benchmarking Robustness of 3D Object Detection to Common Corruptions*, CVPR, 2023.
- H. Koschmieder, *Theorie der horizontalen Sichtweite*, 1924 (atmospheric scattering / dehazing forward model).
- Where2comm; V2X-ViT; UniV2X (cooperative multi-agent perception).
