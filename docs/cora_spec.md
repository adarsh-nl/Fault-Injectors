# CoRA implementation spec (arXiv 2512.13191, AAAI 2026)

One-pass reimplementation contract. CoRA has no released code or weights, so
this document IS the ground truth the code is written against: every equation
as the paper states it, every hyperparameter the paper gives, and — the part
that makes a one-shot possible — every detail the paper does NOT give,
pre-decided here with rationale, so no ambiguity survives to surface as a
training bug. Sources: the paper as transcribed in `docs/corabench_design.md`
§1 (the paper is not fetchable from this cluster; that transcription, with
its equation numbers, is the repo's authority) plus the measured post-mortems
of the previous implementation (§1.5–1.6 there), which are treated as
*known-failure data*, not as design to copy.

The package implementing this spec is `corabench/` (rebuilt from scratch; the
name is kept because `cpbench/tests/test_layering.py` — which must not be
touched — hard-codes it in its importability contract). Each module cites its
section here.

---

## 1. Architecture (paper §Method; equations as numbered in the paper)

Two parallel branches per ego agent i, N agents, noisy shared poses
`T̂_j = T_j ⊕ ΔT_j`.

### 1.1 Encoder E  [spec §1.1]

PointPillars, voxel 0.4 m, per-agent BEV feature `F_j ∈ R^{C×H×W}`, all
agents expressed in the ego frame using the shared (possibly noisy) poses.

* **PAPER UNSPECIFIED — where the warp happens.** Chose: project each
  agent's *points* into the ego frame before voxelisation (OpenCOOD
  `proj_first` convention), so every agent shares the ego BEV grid and no
  feature-space warp is needed. Because: it is the convention of the OpenCOOD
  base the paper builds on, and it makes pose error act on exactly the tensor
  the paper's robustness protocol perturbs (the shared pose).
* Encoder weights are shared across agents (paper: single encoder E).
* Implementation: `cpbench.models.PointPillarEncoder` (consolidated; the
  layering test forbids redefining it).

### 1.2 CIT — Competitive Information Transmission  [spec §1.2, 2-round]

Round 1: `M⁽¹⁾_j→i = H_conf(F_j) ∈ R^{1×H×W}` (confidence logits).
Ego demand `D_i = 1 − σ(H_conf(F_i))`; relevance `S_j = D_i ⊙ σ(M⁽¹⁾_j→i)`;
winner-take-all `I_win = argmax_j Stack({S_j})`; exclusive binary requests
`Q_j ∈ {0,1}^{1×H×W}`. Round 2: `M⁽²⁾_j→i = F_j ⊙ Q_j`; disjoint masks sum:
`F_coll = Σ_j M⁽²⁾_j→i`.

* **PAPER AMBIGUOUS — σ placement in S_j (Eq. 4).** The paper writes
  `S_j = D_i ⊙ M⁽¹⁾` with `M⁽¹⁾` a confidence-head output (a logit, per
  Eq. 2/3's own σ on the ego side). Applied `σ(M⁽¹⁾)` so winner-take-all
  compares probabilities, not a probability × logit — faithful to intent.
  *(User-approved 2026-08-05.)*
* **Non-differentiability of argmax:** the winner mask is computed under
  `no_grad` and applied as a constant. Gradients reach collaborators through
  the selected features (round 2), the confidence heads through `S_coll` and
  the demand path. This is inherent to WTA, not a defect.
* Ablation strategies behind config: `winner_take_all` (default) | `topk` |
  `maxout` (paper Table 4).

### 1.3 LC — Lightweight Collaboration  [spec §1.3; CSSM = paper Eq. 8]

1. Confidence weighting: `F̂_coll = F_coll ⊙ S_coll`, `F̂_i = F_i ⊙ σ(H_conf(F_i))`.
   * **A2 (unspecified S_coll):** winner's confidence per cell,
     `S_coll = Σ_j σ(M⁽¹⁾_j) ⊙ Q_j` — the masks are exclusive so this is a
     well-defined single confidence per cell.
2. Attention harmonisation of the collaborator stream.
   * **A1 (paper cites "(Xu et al. 2022b)" = OPV2V AttFusion).** Chose:
     per-pixel scaled-dot attention over the 2-token stack `{F̂_i, F̂_coll}`
     with the output taken at the collaborator token. Because: AttFusion
     attends over the *agent axis* per pixel, and after CIT's disjoint sum
     the only agent axis that still exists is {ego, collaborators-merged};
     full pixel-dim self-attention at H×W ≈ 35k tokens is O(L²) and appears
     nowhere in the paper's budget.
3. Conv branches → `Z_coll, Z_i`; `Z_fused = Z_coll + Z_i` (verbatim).
4. **CSSM (Eq. 8):** `X_ssm = CSSM(x = Z_fused, Δ = Linear(Z_fused),
   C = Z_i)` — "based on Mamba". Everything numerical is unspecified; §3 of
   this spec fixes it.
5. **Gating unit (Eq. 9-10, verbatim):** `g = σ(DWConv(Conv(X_ssm)))` with
   **`g ∈ R^{1×H×W}` — single-channel spatial gate** (paper-explicit shape;
   Conv maps C→1, DWConv acts on the 1-channel map, gate broadcast over
   channels); `F_out = Conv(MLP(X_ssm) ⊙ g)`. *(2026-08-05 fidelity review
   against the arXiv HTML: an earlier draft gated per-channel; corrected.)*
6. **Teacher distillation (training only, Eq. 11):**
   `L_align = ‖F_out − F_teacher‖²`, teacher fed the dense (un-masked)
   collaborator features.
   * **A5 (unspecified teacher):** independent EMA target network
     (deepcopy, `requires_grad_(False)`, momentum 0.999, buffers copied,
     teacher output detached). Because: the measured alternative — weight
     sharing with output detach — let the target outrun the student
     (`L_align → 5e16`, commit 5813c24 post-mortem).
   * **RECON-3 resolution (reduction):** Eq. 11 is a sum with no coefficient;
     a literal sum at unit weight is ~1.8e7× every other term. Chose `mean`
     reduction with `λ_align = 1.0`, stated plainly: this weight is OURS, not
     the paper's, and is resolution-dependent by construction.

### 1.4 Detection heads  [spec §1.4]

Anchor-based PointPillars head (`cpbench.models.DetectionHead`), A = 2
anchors (0°, 90°), Ncls = 1.

* **A8/RECON-5 (yaw encoding):** `reg_dim = 8` — (sin, cos) yaw channels,
  decoded with `atan2`. Never `asin` (180°-ambiguous, gradient singular at
  ±1 — a measured failure of the previous implementation). Encode:
  `cpbench.data.TargetAssigner(reg_dim=8)`; decode:
  `cpbench.data.BoxDecoder(reg_dim=8)` (atan2 path).
* Focal prior: cls bias = `−log((1−π)/π)`, π = 0.01 → −4.595 (cpbench head
  ships −4.59; asserted at construction). Healthy init cls loss ≈ 1.1, not
  ~50 (job 547612's measured failure without the prior propagating).

### 1.5 PAC — Pose-Aware Correction  [spec §1.5; gating = paper Eq. 13]

Collaborators transmit local head outputs `O_j = {C_j (logits), R_j}`.

1. Selection + semantic association. The selection stage is **paper prose,
   not an equation** — verified against the arXiv HTML: *"we first employ
   convolutional operations to select high-confidence results from
   collaborator outputs"*, a distinct step before Eq. 12. Then
   `A_j = σ(f_attn(Concat(PE(O_i), PE(O_j))))` (Eq. 12); Eq. 13:
   `C′_j = C_j ⊙ A_j`.
   * **RECON-1 resolution (the known inversion): gates on CLASSIFICATION
     LOGITS are ADDITIVE in log-space.** `C′ = C + logsigmoid(a_sel) +
     logsigmoid(a_attn)` where `a_*` are raw gate logits. Because: in logit
     space zero is p = 0.5, so multiplying a logit toward zero makes the
     model LESS certain of background — the measured inversion (§1.5 of the
     old doc; gate gradient had the wrong sign). `logsigmoid` never forms
     log 0. Regression maps are deltas in linear space, where multiplicative
     shrink toward the anchor default is meaningful: `R′ = R ⊙ σ(a)`.
   * **Consequence (double-count guard):** additive gating passes the −4.59
     prior through intact, so NO additional focal bias anywhere downstream
     in PAC. Fuse convs are identity-mean initialised (below) precisely so
     the prior survives to the output; asserted by probe at validation.
2. Geometric correction: `Δp_j = f_offset(Concat(C_i, R_i, C_j, R_j))`,
   `C″ = DeformConv(C_j, Δp_j)`, `R″ = DeformConv(R_j, Δp_j)`
   (torchvision `deform_conv2d`, kernel 3; its backward is nondeterministic —
   accepted, warn-only determinism).
   * **PAPER UNSPECIFIED — deform init.** Chose identity: centre tap 1,
     elsewhere 0, offsets init 0 (zero-init final offset conv). Because: at
     init the correction must be a no-op so the branch starts from the
     collaborators' calibrated logits, not noise.
3. Fuse (C′,R′) with (C″,R″):
   * **A3:** 1×1 conv over channel concat, weights initialised to the mean of
     the two inputs (0.5/0.5), zero bias — identity-mean, prior-preserving.
4. `PE` — **A7:** sinusoidal embedding of the 8-vector (x,y,z,l,h,w,α,δ) per
   cell, from the anchor-decoded box parameters; decode inside PE uses
   sin/cos channels directly (no angle reconstruction needed — the embedding
   consumes sin α and cos α as two of its inputs, sidestepping atan2 in the
   differentiable path entirely).
   * **PAPER UNSPECIFIED — PE dim:** 64 per parameter group, projected to
     pe_dim=64. Standard transformer sinusoid, frequencies 2^k.

### 1.6 Adaptive final fusion  [spec §1.6]

`Concat(C_lc, C_pac) → conv → (U_lc, U_pac)`; recalibrate; pool decoded
boxes from both branches; 3-D NMS.

* **A4 (unspecified recalibration form): logit-space additive,**
  `z′ = z − U` (score = σ(z − U); U > 0 down-weights). Because: (a) it is
  exactly one logit, so every loss stays `*_with_logits` and fp16-safe with
  no clamps — the probability-space product `σ(z)·σ(−U)` is not the sigmoid
  of any logit and forced the old implementation into a float32 island with
  a measured fp16 clamp no-op; (b) monotone in U with the same semantics.
* **RECON-2 resolution (u_reg):** `L_u = mean(U_lc²) + mean(U_pac²)` with
  `u_reg = 1e-2`. Because: measured — at 1e-4 the term contributed ≤ 0.05%
  while |U| excursed 45×; 1e-2 makes the term ~1% of the objective at that
  excursion scale, enough to bind, small enough not to pin U at 0.

## 2. Loss  [A6; five terms]

```
L = w_local·L_det(local_i) + w_lc·L_det(z′_lc, R_lc) + w_pac·L_det(z′_pac, R_pac)
    + λ_align·L_align + u_reg·L_u
w_local = w_lc = w_pac = 1.0, λ_align = 1.0, u_reg = 1e-2, reg_weight = 2.0
```

`L_det` = `cpbench.training.DetectionLoss(reg_dim=8)`: focal (α 0.25, γ 2)
on logits + smooth-L1 on positive anchors. LC/PAC classification is
evaluated on the RECALIBRATED logits `z′` (that is what trains U beyond the
regulariser, and what inference decodes). `L_local` is the ego's local head
against ego targets (collaborator GT in their own frames is not available at
the ego, and the encoder+head are shared, so ego supervision trains them).

## 3. CSSM numerics  [RECON-4 resolution — the divide-free scan]

* State matrix `A = −exp(a_log)`, `a_log` init `log(1..N)`, N = d_state = 16
  (Mamba S4D-real init; negativity is structural — no gradient step can flip
  it).
* `Δ = softplus(dt_proj(z_fused))`, **bias init = softplus⁻¹(dt)** with
  `dt ~ exp(U(log 1e-3, log 1e-1))` — Mamba's dt range. Asserted at
  construction: `softplus(bias) ∈ [1e-3, 1e-1]` elementwise. No runtime
  clamp on Δ: the SSD scan below is stable for any Δ > 0, and the measured
  Δ explosions of the old implementation were driven by the `b/E` form it
  replaced (268% wrong vs float64; SSD 3.1e-7).
* **Scan (divide-free, log-space):** per chunk,
  `logE = cumsum(Δ·A)` (non-increasing);
  `h_t = exp(logE_t)·h_0 + Σ_{s≤t} exp(logE_t − logE_s)·b_s`,
  `b = (Δ·x) ⊗ B`. For `s ≤ t` the pairwise exponent is ≤ 0, its exp is
  bounded by 1 — asserted. **No division, no clamp**; underflow forgets
  completely (correct) instead of a clamped "forget nothing" (the measured
  defect). Output `y_t = C_t·h_t + D·x_t` with C from `Z_i` (Eq. 8's "output
  matrices C = Z_i"). Chunk 64; per-chunk gradient checkpointing
  (config-switchable) for the (Lc×Lc) pairwise memory.
* **A9 (scan order):** VMamba-style cross-scan, 4 directions (row-major ±,
  column-major ±), outputs averaged; `avg_pool2d(2)` before the scan and
  bilinear upsample after (memory; documented as ours).

## 4. Training protocol  [paper experimental setup]

OPV2V + DAIR-V2X-C, OpenCOOD base: comm range 70 m, Adam lr 1e-3, weight
decay 1e-4, multistep ×0.1 at [15, 25], 30 epochs, batch 2, AMP. Grid:
voxel (0.4, 0.4), range (−140.8,−40,−3, 140.8,40,1) → 704×200 pillar canvas,
feature 352×100, C = 256. Robustness protocol (evaluation-time, via this
repo's injectors): pose σ ∈ {0, 0.2/0.2, 0.4/0.4, 0.6/0.6} (m/deg); latency
0–400 ms at 0.6/0.6; collaborator count 1–5. Training is CLEAN by default;
pose-noise-robust training (σ up to 0.6/0.6) is the `trainer=robust` group.

## 5. Self-checks (construction- and validation-time; `corabench/selfcheck.py`)

1. **No hard clamps on differentiable paths.** Static source scan of the
   package: `.clamp(`/`torch.clamp` are forbidden in model/fusion source
   except on lines explicitly marked `# no-grad-ok` (which must sit under
   `no_grad` or on detached statistics). Bounds are smooth (tanh, softplus,
   logsigmoid) or log-space-before-exp.
2. **Focal prior bias** asserted on every cls-producing conv at
   construction: `bias ≈ −log((1−π)/π), π=0.01`; PAC's prior-preservation
   asserted by probe (mean init cls logit ∈ [−6, −3]).
3. **dt init** asserted in `[1e-3, 1e-1]`; scan asserted divide-free
   (pairwise exponents ≤ 0 at runtime, cheap).
4. **Yaw**: reg_dim = 8 everywhere (assigner, head, decoder asserted equal);
   decode is atan2; **no `asin` anywhere in the package** (static scan).
5. **fp16**: no probability clamps (static scan for `clamp(1e-`); all cls
   losses `*_with_logits` on single logits.
6. **Optimizer guard**: `clip_grad_norm_`; if the returned norm is
   non-finite, skip the step and `zero_grad` — never step on inf/NaN.
7. **Shape asserts** at every cross-agent seam: CIT entry/exit, LC fusion,
   PAC concat, final fusion (`selfcheck.assert_shape`).

## 6. Validation gate (before any training claim)

`corabench/validate.py`: synthetic multi-agent batch
(`cpbench.data.SyntheticCooperativeDataset`) through full train-mode forward
(teacher on) + loss + backward under `torch.autograd.set_detect_anomaly(True)`.
Must hold: finite outputs of documented shapes; finite gradient for every
parameter; init loss components in sane ranges (cls ≈ O(1), not ~50; Δ in
[1e-3, 1e-1]; reg not orders off); the static self-checks pass. These are
the eight-debugging-rounds failure classes, checked at t=0.
