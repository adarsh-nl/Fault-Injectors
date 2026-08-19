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

---

## 7. Training-stability investigation — closed hypotheses

Two mitigations were proposed for the intermittent NaN failure and both are
**closed by evidence, not merely unsupported**. Do not revisit either without
new data that contradicts what is recorded here.

Failure being explained: job 559279 gate-aborted at step 199 (50% non-finite
loss) while job 559057, at the *same* `dt_bound=0.2`, same seed, same data and
same node type, ran 5000 steps with 9 skips. The 5-seed diagnostic (job
559461, seeds 2026–2030, 500 steps each, identical config differing only in
`--seed`) measured a base rate of **1 failure in 5**, so the two earlier runs
are consistent with a coin flip rather than with a config difference.

### 7.1 CLOSED — `dt_bound` / Δ saturation is not the driver

Within-window (controlling for the fact that both Δ and the NaN rate drift up
with step count), `delta_p99` does **not** separate non-finite steps from
finite ones in job 559279; the separation is ~−0.003, i.e. the *wrong sign*.
41 steps had `p99 ≥ 0.1995` with a finite loss and 3 non-finite steps had
`p99 < 0.17`. Saturation is neither necessary nor sufficient. Δ is retained as
a co-symptom readout only. The dt_max A/B (job 559278) was a correct
experiment aimed at the wrong mechanism.

### 7.2 CLOSED — GradScaler `init_scale` reduction / scale warmup

Hypothesis: basin entry is selected by early GradScaler backoffs, so lowering
`init_scale` or warming the scale would avoid it. **The evidence contradicts
this rather than failing to support it**, on three independent counts (job
559461):

* **Mean opening skips are identical.** Bad 3.00 vs survived 3.00 over the
  first 10 steps; 4.00 vs 4.25 over the first 50.
* **The failing seed backs off LESS than average**, not more (4 skips in 50
  vs a survivor mean of 4.25).
* **Seed 2026's opening pattern `000110111111` is byte-identical to job
  559279's, and 2026 survived** 500 steps with zero non-finite losses where
  559279 died. The same opening produces both outcomes.

Additionally the seed with the *most* opening backoffs (2026, the only one to
drop to `scale=2048`) was a survivor. `--init_scale` remains wired in
`train_opencood.py` and costs nothing to pass, but it is not a candidate
mitigation for this failure.

### 7.3 CLOSED — `cssm_in` magnitude is not usable as an early-warning breaker

The first read of job 559461 flagged a ×3.65 ramp in `cssm_in_max` before
seed 2028's first non-finite step, which looked like a tripwire. That was an
artifact of comparing 2028's 82-step window against the survivors' full
500-step windows. Over a **matched** 0–82 window:

| metric | 2028 (failed) | worst survivor | verdict |
|---|---|---|---|
| `cssm_in_p99` log-slope | 0.02245 /step | **0.02594** (2026) | not separable |
| `cssm_in_max` log-slope | 0.02426 /step | **0.03538** (2026) | not separable |
| `cssm_in_p99` peak | 2.6875 | 3.0820 (lowest survivor) | not separable |
| `cssm_in_max` peak | 15.8047 | 18.3750 (lowest survivor) | not separable |

The failing seed is **not an outlier on either rate or level — it is the
lowest of all five on both peaks**, so an absolute threshold would fire on
every survivor *before* it fired on 2028, and a rate threshold would fire on
2026 first. `cssm_in` is a symptom that arrives too late and too weakly to act
on. The activation probe (`--act_probe`) is kept for diagnosis; it is not a
basis for a runtime breaker.

**What remains open:** why the LC forward produces a non-finite value at all.
The NaN reaches `loss_lc_cls`, `loss_pac_cls` and `loss_align` at the same
step while `loss_local_cls` is never non-finite, and no weight is non-finite
at abort (0 of 167 tensors in `abort_step199.pt`), so the fault is in the
forward activations of the LC path and the parameters survive it.

### 7.4 CLOSED — the launched configuration contradicted §4, and §4 was right

Jobs **559595 / 559596** (seeds 2026 / 2029) were launched on 2026-08-09 and
cancelled 45 min later, before either had taken an optimiser step, because
they were training to the wrong geometry.

`cora_full.sbatch` copied `~/opencood-eval/cobevt/config.yaml` as its hypes
and rewrote only `root_dir`/`validate_dir`. That inherited **CoBEVT's
architecture choices wholesale**:

| parameter | §4 / OpenCOOD OPV2V | what was launched | class |
|---|---|---|---|
| detection range | `[-140.8, -40, -3, 140.8, 40, 1]` | `[-140.8, -38.4, -3, 140.8, 38.4, 1]` | **training** |
| pillar canvas | **200 × 704** | 192 × 704 | **training** |
| feature_stride | **2** → head 100×352 | 4 → head 48×176 | **training** |
| score_threshold | 0.20 | 0.25 | eval |

The range and stride are **not fixable by re-evaluation**: they set the pillar
grid the scatter writes into and the lattice the regression head learns
offsets against, so they are baked into the weights. Only `score_threshold`
was eval-time.

These are not V2XSet artifacts — CoBEVT uses ±38.4 / stride 4 on OPV2V *and*
V2XSet, and Where2comm's OpenCOOD config carries the same. **A baseline
model's config is never a valid base for CoRA.** The correct base was in the
OpenCOOD repo the whole time:
`opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml` — OPV2V-native,
`IntermediateFusionDataset` + `SpVoxelPreprocessor`, range ±40, stride 2,
score_threshold 0.20, voxel 0.4. The repo ships ~20 OPV2V-native configs and
none had been used.

Paper basis, quoted: *"We adopted the basic settings from OpenCOOD (Xu et al.
2022b). ... with each voxel's side length set to 0.4 meters."* The paper
enumerates almost nothing else, so OpenCOOD's OPV2V defaults **are** the
specification for everything unstated — which is exactly what §4 recorded.

**§4 was correct and the launch contradicted it.** The spec was written from
the paper during the reimplementation and states "range (−140.8,−40,−3,
140.8,40,1) → 704×200 pillar canvas, feature 352×100"; nothing reconciled it
against what the sbatch actually loaded.

**Two fixes, so this cannot recur:**

1. `cora_full.sbatch` now points explicitly at the OPV2V-native OpenCOOD
   config, with the reason recorded inline.
2. `train_opencood.assert_geometry()` hard-asserts the **derived** canvas and
   head grid at startup — before the model is built, before any step — via
   `--assert_canvas 200x704 --assert_head 100x352`. It checks the geometry
   that was actually built, not the YAML that was meant to produce it, because
   both failures so far (557631's stride mismatch and this one) had plausible
   YAML and wrong derived geometry. Verified to fire on the CoBEVT config
   (`canvas is (192, 704), expected (200, 704); head/anchor grid is (48, 176),
   expected (100, 352)`) and pass on the OPV2V-native one; covered by a
   doctest.

### 7.5 CORRECTION — `loss_align` is a passenger, not the driver

Job **561546** (seed 2029) aborted at step 7,667. Its loss decomposition, per
200-step window, contradicts the framing that had guided the investigation:

| window | loss | **align** | loc_cls | lc_cls | pac_cls | align % |
|---|---|---|---|---|---|---|
| 5000 | 4.157 | 0.0050 | 0.296 | 0.226 | 0.224 | 0.1% |
| 7200 | 4.403 | 0.0076 | 0.379 | 0.279 | 0.273 | 0.2% |
| 7400 | 11.386 | 0.1146 | 1.590 | 3.007 | 1.865 | 1.0% |
| 7600 | 28.780 | 0.2117 | 2.922 | **14.525** | **6.895** | 0.7% |

`loss_align` grows 40×, but from 0.005: it never exceeds **1.0%** of the total
(whole-run max 3.98%, mean 0.19%). What explodes is `lc_cls` (52×) and
`pac_cls` (25×), with `loc_cls` lagging far behind.

Re-checking **559640**, where align was previously described as driving the
divergence: its share peaks at **8.2%** at the blow-up (0.0578 → 36.85 in
absolute terms). Larger than 561546, still a minority. The earlier claim came
from reading align's absolute growth without normalising by the total, and was
wrong in emphasis.

**Standing conclusion: four failures, one location.** Two NaN failures (559279
step 199, 561545 step 382) and two divergences (559640 ~16k, 561546 7,667) all
show the same signature -- `lc_cls` and `pac_cls` move together while
`loc_cls` lags. Align sits downstream of LC (it compares LC's `f_out` to the
teacher's), so it rises when LC's output magnitude rises: a symptom of the
same thing. **Every hypothesis tested so far -- `dt_bound` (§7.1),
`init_scale` (§7.2), the `cssm_in` breaker (§7.3), and now align -- has been
ADJACENT to the LC/CSSM block rather than in it.**

### 7.6 Gate observation — failure-agnostic beat mechanism-specific, again

Both rules added after 559640 fired on 561546:

* two-sided ratio **WARN at step 7,600** -- ratio 0.2462 < 0.25, "the STUDENT
  is 4.1x the teacher";
* loss-trend **ABORT at step 7,667** -- trailing-200 mean 18.173 vs running
  minimum 3.621 (4.97x, threshold 5x).

It aborted at 7,667 instead of running to 18,760+ as 559640 did: **~41 h of
GPU saved**. Non-finite rate at the abort was 0.38%, nowhere near the 20%
skip-rate threshold -- the original breaker alone would have missed it.

Critically, **`t_student_absmax` stayed FLAT** (1.30 in the 6000-7999 block vs
1.31 for 559640 in the same block), where on 559640 it was the dramatic signal
(1.17 -> 17.69). A breaker keyed on student magnitude would have caught 559640
and **missed** 561546.

This is the **second** occasion on which a mechanism-specific breaker would
have failed on the next variant while the failure-agnostic one caught it (the
first: §7.3, where the `cssm_in` tripwire was not separable and the
loss-trend rule was). Prefer failure-agnostic breakers.

### 7.7 PRE-REGISTERED experiment: four arms, 9,000 steps

Registered BEFORE the runs land so the reading cannot be fitted afterwards.

| arm | change | 
|---|---|
| **CONTROL** | unchanged configuration |
| **A** | `lambda_align = 0.0` |
| **B** | `lambda_align = 0.1` |
| **C** | `ema_momentum = 0.99` (tau 100 instead of 999), lambda unchanged |

**PREDICTION (from §7.5):** arm **A diverges at roughly the same point as the
control**, and `lc_cls`/`pac_cls` explode in both. Align is a passenger, so
removing it should not move the failure.

What each alternative outcome would mean:

* **A survives, B diverges** -> the term is scale-sensitive and lambda=1.0 is
  simply too high. The sum->mean + lambda=1.0 judgement call (§1.6) was wrong.
* **A and B survive, C also survives** -> ambiguous; align involvement cannot
  be separated from the EMA lag at this length.
* **C survives where the control dies, A does not** -> the teacher lag IS
  load-bearing and the §3 reading (that the detached teacher is a RESTORING
  force, so the lag is a consequence not a cause) is wrong.
* **All four diverge together** -> neither the loss term nor the teacher; the
  LC/CSSM block itself, and the next experiment belongs there.

The discriminator is **which loss term moves**, not survival: every arm logs
the full per-component decomposition per 200-step window.

**BOUND, stated up front:** 9,000 steps clears 561546's abort (7,667) but NOT
559640's turn (~16k). A negative result bounds the FAST failure mode only, and
**a surviving arm at 9k is not evidence of a fix.**

Note `parts["loss_align"]` logs the RAW term, not the lambda-weighted
contribution, so arm A still reports align's magnitude even though it
contributes nothing to the total -- deliberately, so the term stays observable.

### 7.8 STANDING LESSON — a traceback shows where execution stopped, not what caused it

Twice in this investigation a mechanism was asserted from a traceback plus an
injector contrast, and twice it was wrong.

1. **`loss_align` as CoRA's driver** (§7.5). Align's absolute growth was
   striking (0.005 -> 3.859 per block on 559640), and it was the largest
   single component in a last-100 snapshot. Normalised against the total it is
   a *minority* contributor -- max 1.0% in 561546, 8.2% in 559640 -- while
   `lc_cls` and `pac_cls` carry the divergence. The error was reading an
   absolute magnitude without dividing by the quantity it was claimed to
   dominate.

2. **CoSDH `agent_drop`'s root cause** (see `tools/sweep/aggregate.py`
   `NOT_APPLICABLE`). Two injectors failed; one had a confirmed `record_len`
   mismatch; the second was assumed to share it because the contrast
   (only the two agent-set-changing injectors fail) was compelling. Direct
   measurement showed `agent_drop` has NO mismatch -- `record_len=1` and
   exactly 1 agent produced features -- and it dies somewhere else entirely,
   in ground-truth assembly, before AP is computed. The two failures are
   unrelated.

**Rule: instrument the quantity before writing a claim about a published
model.** A traceback localises the *stop*, not the *cause*; an injector
contrast localises the *trigger*, not the *mechanism*. Both are evidence for
where to measure, never a substitute for measuring. This applies with
particular force to claims that will appear in a paper about someone else's
released code, where being wrong is a statement about their work, not ours.

Corollary, from the same two cases: when a single explanation covers several
observations neatly, that is the moment to measure each one separately. Both
errors took the form of one tidy story absorbing a second data point that did
not belong to it.

### 7.9 STANDING LESSON — a 90-second smoke run before a 50-minute one

Companion to §7.8. That entry says *instrument the quantity*; this one says
*prove the instrument runs first*.

The GT-union analysis took **four attempts**, and three failed on things a
minimal run exposes in under two minutes:

1. called `__getitem__` on 5 datasets x 120 frames x 2 models -- 1,200 full
   voxelization passes -- and was killed by its own 50-minute timeout **with
   stdout buffered to a file, so every line was lost**. A 0-byte log after
   50 minutes of compute.
2. rewritten to skip voxelization, but 8 frames still exceeded 100 s: the cost
   was **dataset index construction**, not the frames. The assumption about
   where time went was never checked.
3. `AttributeError: 'IntermediateFusionDataset' object has no attribute
   'generate_object_center'` -- it lives on `post_processor`. A **3-frame run
   would have surfaced this in 90 seconds**; instead it surfaced after a
   full-length run.

Only attempt 4, preceded by that 3-frame smoke test, produced anything.

**Rules, both cheap:**

* **Smoke-run at minimum scale before full scale.** N=3, one model, one tier.
  It costs ~90 s and catches every API error, every scope error, and every
  wrong-shape assumption.
* **Never buffer output you might lose.** Long runs get `python -u` and
  incremental `flush=True`, so a kill leaves partial results rather than an
  empty file. A timeout is a likely outcome, not an exceptional one.

The same failure appeared in the CoRA arms: four jobs produced 0-byte CSVs for
three days on a `NameError` at model construction, which `corabench/validate.py`
-- a one-minute synthetic gate that already existed -- would have caught before
submission. Three separate incidents, one root cause: **spending long before
verifying cheap.**

### 7.10 PROTOCOL DECISION for future work — the GT denominator

NOT to be done now; recorded so the choice is explicit rather than inherited.

`agent_drop` is measured against a MOVING denominator (see
`manifest['gt_union_confound']`). Two ways to fix it properly:

**Option 1 -- log `n_gt` per cell and re-run `agent_drop`.** Keeps AP's
meaning intact: the denominator is what the surviving agents can actually be
asked to detect. Makes the shift *measurable* rather than *removed*, so the
correction is reported rather than applied silently. Costs a re-run of 3 tiers
x 4 models. `n_gt` is now in the bundle schema (`tools/fi_inference.py`, both
top-level and per-stratum) so future cells carry it; EXISTING cells are NOT
backfilled and keep the caveat.

**Option 2 -- fix GT to the clean union so the denominator is constant across
tiers.** Makes the tiers directly comparable and removes the confound
outright. But it CHANGES WHAT AP MEANS for this fault: objects that NO
surviving agent observes stay in the denominator, so the model is scored on
targets it has no sensor evidence for. That is arguably the right question for
a robustness benchmark -- "how much perception is lost" rather than "how well
does it do on what remains" -- but it is a different question, and mixing it
with the other injectors' AP would be inconsistent.

The tradeoff is not cost, it is semantics. Option 1 measures degradation
relative to the achievable; option 2 measures it relative to the ideal.

### 7.11 PRE-REGISTERED — ablation ladder, rungs 2 and 3

Registered BEFORE the runs land. Rung 1 is deliberately HELD (see below).

Five failures now, and LC is where it shows every time: 559279 (NaN, step
199), 561545 (NaN, 382), 559640 (divergence, ~16k), 561546 (divergence,
7,667), and arm B (NaN, 238, with `lc_cls` already 1.902 in its first window
against ~0.80 in every other arm). `loc_cls` stayed finite in all five.

| rung | configuration | flags |
|---|---|---|
| 2 | local + LC, no PAC, no teacher, no align | `--no_pac --no_teacher` |
| 3 | local + LC + PAC, no teacher | `--no_teacher` |
| 4 | full model | already have it: control arm, 2,078 clean steps |

2,000 steps, 2 seeds each (2029, 2026), 4 jobs.

**STRUCTURAL NOTE that reorders what the ladder isolates.** PAC consumes the
LOCAL branch's outputs, not LC's (`cora.py`: `ego_o = {"cls":
local["cls"][sl][0:1], ...}`). So PAC is NOT downstream of LC, and rung 2 vs
rung 3 separates PAC's own contribution rather than any LC interaction. The
LC-specific coupling is teacher/align, which is rung 3 vs rung 4.
`teacher_enabled=False` also zeroes align (`align_terms` is appended only when
`self.training and self._teacher_enabled`), so `--lambda_align` is redundant
on both rungs.

**PREDICTION.** Rung 2 is the discriminator, and the two outcomes point at
different problems:

* **Rung 2 DIVERGES** -> LC is isolated. The fault is inside LC/CSSM itself,
  not in its coupling to PAC or the teacher. Next work goes inside the CSSM,
  and **rung 1 becomes unnecessary** -- its control value is spent.
* **Rung 2 SURVIVES both seeds** -> LC alone is fine and the fault is in the
  COUPLING. Rung 3 then localises it: if rung 3 also survives, the coupling is
  teacher/align (rung 3 vs rung 4); if rung 3 diverges, it is PAC's
  interaction with the local branch. **Rung 1 then becomes worth running** as
  the control that confirms the local branch is clean on its own.

**TIMEOUT IS AN ACCEPTED OUTCOME, not a reason to resubmit** (decision
2026-08-18). The 10 h limit was sized at 8.7 s/step; the closest measured
comparable (565550) ran ~17 s/step including overhead, so a rung may wall
around step ~1,800. That costs almost nothing: divergence is unaffected by
truncation, and survival at 2,000 steps was ALREADY registered below as
non-conclusive. The 30-minute copy-back and the exit-trap rsync preserve
every CSV row either way. REPORTING RULE: a rung that walls clean is
reported as "survived ~N steps" with N STATED, never as "survived" -- the
unqualified word would silently upgrade a truncated run to the
pre-registered 2,000-step bound it did not reach.

**BOUND, stated up front.** Basin entry is ~1-in-5, so **two clean seeds is
suggestive, not proof** -- the chance a broken rung looks clean twice is not
small. And 2,000 steps clears only the FAST mode: arm B died at 238 and 561545
at 382, but 561546 turned at 7,667 and 559640 at ~16k. **Divergence is
conclusive; survival is not.** A surviving rung bounds the fast mode alone.

**Rung 1 is HELD** because it requires making `CoRALoss`'s LC term conditional
-- editing the component under investigation -- and its expected answer is
partly known already (`loc_cls` finite in all five failures). It is a control,
and a control is only needed once rung 2 says which branch of the story we are
in.

### 7.12 MEASURED — the scan math is closed; the trainable subsystem is not

Full report: `docs/cssm_verification.md` (jobs 567820/567827, harness
`tools/cssm_verify.py`). The handwritten parallel SSD scan was verified
against a naive per-step sequential recurrence and a third, independently
structured chunked-SSD implementation: forward AND backward agree at
machine epsilon in fp64 (~1e-16 rel) and fp32 rounding otherwise, for
every input and every parameter, including the torch-1.12 reentrant
checkpoint path (bitwise-identical grads, ckpt on vs off). fp64 gradcheck
passes. The mask is exactly the causal recurrence. Do not reopen scan-math
hypotheses without evidence that contradicts those logs.

**THE DISTINCTION THIS RESULT MUST NOT BE READ PAST.** What is
demonstrated is that the *recurrence is stable in isolation*: at frozen
inputs, ||h_t|| is stationary over all 8800 tokens, at delta pinned to the
bound and input scales up to 100x. What is NOT demonstrated is that the
*complete trainable CSSM subsystem* — the recurrence together with
dt/b/c projections, the shared a_log/dt_bias, the optimizer, and the
measured degree-3.0–3.4 activation-scale gain law feeding those gradients
— is stable under training. The live hypothesis (7.14) lives exactly in
that gap. Any future claim of the form "CSSM was verified correct, so the
instability must be elsewhere" is an over-read of 7.12 and is wrong: the
verification covers the map, not the closed loop of map + optimizer.

### 7.13 CONFIRMED fidelity bug — `_directions` "column-major" is not column-major

Separate from the instability; no demonstrated link to it. MEASURED (jobs
567820/567827): `CSSM._directions` computes `col = (idx % w) * h + idx //
w`, which is the *inverse* (scatter form) of the true column-major gather
permutation `(idx % h) * w + idx // h`. The two coincide **only when
h = w**, which is why square test grids never caught it and why any future
permutation/direction test MUST use a non-square grid. On the production
pooled 50x176 BEV, scan directions 3 and 4 traverse the map in jumps of
mean 72.1 cells, 0% raster-adjacent (true column-major: mean 1.97, 98%
adjacent). Two of four scan directions therefore learn over a spatially
scrambled ordering. Contradicts A9. It is numerically harmless — the
permutation is a bijection, applied consistently to all four scan inputs,
inverse restore exact, forward/backward verified — the damage is semantic
(locality), not numerical.

**PRE-REGISTERED fix arm, NOT applied now.** The one-line fix changes the
learned function and would confound the stability investigation, so it is
its own future arm: same protocol as 7.7/7.11, prediction NEUTRAL on
stability (no mechanism links the scramble to divergence), expected effect
on ACCURACY only (restoring vertical-neighborhood mixing). Do not bundle
it with any stability intervention; a bundled run answers neither question.

### 7.14 PRE-REGISTERED — backward-side precedence experiment

Registered BEFORE the instrumented runs land. The 7.12 gap plus the
measured cubic gain law yield one candidate mechanism: optimizer-mediated
positive feedback through the unnormalized degree-3 path (activation drift
-> cubically amplified gradients into the selective parameters -> larger
Adam steps -> more drift), with the post-scan LayerNorm keeping the
FORWARD loss looking healthy until late. LayerNorm makes downstream
feature magnitude insensitive to upstream scale; it does NOT make the
optimization insensitive to it. The experiment must therefore distinguish
"LayerNorm suppresses the activation and the problem is elsewhere" from
"LayerNorm suppresses the activation while gradients into the selective
parameters keep growing."

**Instrumentation** (`--grad_probe`, every optimizer step, one shared
timeline in the training CSV): per-group RAW gradient norms (post-unscale,
PRE-clip) and realized step ratios ||dtheta||/||theta|| for a_log,
dt_bias, dt_proj, b_proj, c_proj, out_norm, gate_out; activation norms and
boundary GRADIENT norms at z_fused -> y_pre_ln -> y_post_ln (hooks;
AMP-scale corrected); alongside loss_lc_cls, loss_pac_cls, total loss,
t_student_absmax, and the gate's own loss-trend statistic
(`RuntimeGate.trend_ratio`, logged per step).

**PREDICTED CHAIN**, in order:
`g_a_log rises -> g_dt_proj rises -> r_* (step ratios) rise -> LC
activation norms (z_fused) drift -> loss_lc_cls rises -> loss-trend
breaker fires.`

**Onset detector, fixed now:** for each series, baseline = median over
steps 500–2000 (or steps 50–150 for a run that aborts before step 2500;
for the REPLAY run, whose timeline begins at 6000, baseline = median over
steps 6000–6800 — chosen now, before data: two full gate windows, ending
>= 850 steps before the recorded abort at 7,667);
onset = first step whose trailing-100-step median is >= 5x baseline and
stays >= 5x for >= 100 consecutive steps (>= 50 for short runs).

**Decision rule, fixed now:**
* CONFIRMED: in >= 2 failing runs, onset(g_a_log) AND onset(g_dt_proj)
  precede onset(loss_lc_cls) by >= 300 steps, with the full chain ordered
  and each adjacent link separated by >= 50 steps; the breaker fires last.
* REFUTED: onset(loss_lc_cls) precedes the onset of EVERY probed gradient
  series by >= 300 steps in >= 2 failing runs.
* Links closer than 50 steps are SIMULTANEOUS (that link inconclusive);
  outcomes between the two rules are UNRESOLVED -> more seeds, no
  mechanism claim. A single ordered run is coincidence-compatible and
  claims nothing.

**Runs and cost** (measured 13.5 s/step, ctit091-class):
1. REPLAY of 561546 (seed 2029) from `ckpt_step6000.pt` with the probe:
   1,667 steps to the recorded abort, ~9 h incl. staging — ~4x cheaper
   than fresh-to-failure. Full resume contract (RNG, sampler position,
   scaler) restores the trajectory up to per-node nondeterminism; pinned
   a40 to match the original SKU. Risk accepted: drift may shift or lose
   the failure; if so the run is still a valid fresh-tail sample.
2. FRESH probed run, seed 2026 (561545's; it aborted at step 382, so its
   only checkpoints are post-abort and replay is impossible for it).
   Budgeted to 10,000 steps.

**THE TWO ARMS RUN ON DIFFERENT HARDWARE — decision 2026-08-18, recorded
because it affects how they may be compared.** The REPLAY (567949) stays
pinned `--constraint=a40`: it resumes 561546's trajectory from
`ckpt_step6000.pt`, 561546 ran on ctit091 (a40), and the determinism floor
on this cluster is PER-NODE, so the SKU match is load-bearing there. The
FRESH arm (567950) was moved to `l40|l40s` on `ps,main-gpu`: it is a new
trajectory on a new seed, so SKU matching buys essentially nothing, while
the a40 pin cost ~2.5 days queued behind a saturated pool (ctit086/090-094
held by multiple 16- and 41-day jobs on 2026-08-18).

CONSEQUENCE, stated so it is not rediscovered later: **any timing
comparison between the two arms carries a hardware term.** The 13.5 s/step
rate used to size these runs, and the per-node determinism context behind
the onset-detector windows, were both measured on a40. Steps-to-onset and
step ORDERING are the registered observables and are hardware-independent;
wall-clock, s/step, and any cross-arm claim resting on them are NOT
comparable between 567949 and 567950 without accounting for the SKU
difference. Onset detection is defined in STEPS throughout (7.14's
detector and decision rules), which is what makes this acceptable rather
than merely tolerable.
Per-step gradients are NOT reconstructable from checkpoints alone: Adam
state gives only exp_avg (beta1=0.9, ~10-step horizon) and exp_avg_sq
(beta2=0.999, ~1000-step horizon) at snapshot instants. That free coarse
check was run (job 567906): between steps 6000 and 7667 of 561546, the
grad-magnitude proxy grew 1.8x for a_log and dt_bias while every
reference group (encoder, heads, LC branches) stayed ~1.1x —
direction-consistent with the chain, too smoothed to resolve ordering.

**NO pre-norm arm yet** (decision 2026-08-18): pre-normalizing z_fused is
the obvious fix IF the mechanism holds, but it changes the architecture
away from the paper. Mechanism first; the fix arm is registered only
after CONFIRMED.

**THREE-BOUNDARY PATTERN, registered 2026-08-18 before the probe runs
land** (jobs 567949/567950 queued, no data seen). The z_fused / y_pre_ln /
y_post_ln taps admit three qualitatively different outcomes; each supports
a DIFFERENT mechanism and a DIFFERENT fix. Recorded now so the result
cannot be read to fit a preferred fix later:

* **Activation-first**: onset(z_fused) precedes onset(y_pre_ln) precedes
  the gradient-group onsets. Upstream representation growth drives the
  selective parameters; the fault is in what FEEDS z_fused (branches,
  AttFusion2, confidence weighting), not in CSSM. Would justify a
  pre-CSSM normalization arm. y_post_ln reading: if y_post_ln stays flat
  while z_fused/y_pre_ln rise, LayerNorm is confirmed to be masking the
  forward symptom downstream while the upstream drift proceeds — the
  "looks healthy until late" signature with an upstream cause.
* **Gain-first**: z_fused flat (never reaches onset), y_pre_ln rises.
  CSSM's own parameterization amplifies at roughly constant input scale —
  the measured degree-3.0–3.4 gain law acting through dt/B/C growth
  rather than input growth. Would justify reworking the Delta/B/C
  parameterization. y_post_ln reading: flat y_post_ln here means LayerNorm
  absorbs the internal amplification entirely on the forward side, so the
  ONLY escalating observable is backward — the strongest possible
  vindication of backward-side instrumentation (and the exact reason
  7.3's forward breaker failed).
* **Gradient-first**: both activation series flat, g_a_log and g_dt_proj
  onset first, activation and loss growth follow. The optimizer-mediated
  feedback hypothesis proper. Would justify per-group gradient clipping
  or an optimizer change. y_post_ln reading: y_post_ln flat until after
  the gradient onsets, then rising only with/after z_fused, confirms the
  forward path was never the leading indicator at ANY boundary.

In every case y_post_ln is the direct test of "LayerNorm masks the forward
symptom while the backward signal escalates": that claim is SUPPORTED only
if y_post_ln's onset is absent or later than the gradient-group onsets,
and REFUTED for the masking framing if y_post_ln moves with y_pre_ln.

**SECOND REFUTATION CONDITION — parameter movement, registered now.**
Adam normalizes by the second moment: a rising raw gradient with a
proportionally rising sqrt(exp_avg_sq) produces FLAT realized movement,
in which case the feedback loop is broken AT THE OPTIMIZER and the
mechanism fails even with the gradient ordering confirmed. The closing
signature that confirms the loop is therefore:

`g_a_log rises -> ||d a_log||/||a_log|| (r_a_log) rises -> z_fused rises
-> loss_lc_cls rises`

* FLAT, defined now (mirroring the onset detector): a step-ratio series
  r_g is flat iff its trailing-100-step median stays < 2x its baseline
  (same baseline windows as the onset detector) at EVERY step through
  abort. Onset for r_g uses the same 5x/100-step rule as every other
  series.
* REFUTED (optimizer-normalized): g-group onsets occur with the
  registered ordering while r_a_log AND r_dt_proj remain flat through
  abort in >= 2 failing runs. In that case the instability is NOT
  optimizer-mediated feedback through parameter growth, and per-group
  clipping/optimizer changes would treat a symptom that is not the cause.
* CONFIRMED (loop closed): onset(g) precedes onset(r) for a_log or
  dt_proj, which precedes onset(z_fused), which precedes
  onset(loss_lc_cls), each by >= 50 steps, in >= 2 failing runs.

**INTERPRETIVE STANCE, fixed before data:** the evidence as of this
registration (cubic gain law, 7.12's isolation result, the 1.8x Adam
second-moment growth on a_log/dt_bias vs ~1.1x references in 561546's
final 1,667 steps) supports an optimizer-mediated-instability HYPOTHESIS
involving the unnormalized selective-parameter pathway. It does NOT
establish causality. Only the precedence experiment can move it from
hypothesis to supported mechanism, and any report written from these runs
must use exactly that language until the decision rules above fire.

**Quarantine reminder:** the 7.13 column-major fidelity bug stays out of
every stability arm. No stability intervention may bundle it.

### 7.15 MEASURED — ablation ladder rungs 2 and 3; mode C; probe partial

#### Rung 2 VERDICT: diverges on BOTH seeds, by two DIFFERENT mechanisms

* **seed 2026 (567756)** — skip-rate breaker, **step 199**: 42% of the
  trailing 200 steps had a non-finite loss (limit 20%), 87 skipped.
* **seed 2029 (567755)** — **scaler collapse to a dead-gradient state by
  step ~214**. It reached 2,000 steps and exited 0, which would have been
  reported as a clean survival. It was not: 41 non-finite-gradient events
  each halved the GradScaler (65536 -> 2.98e-08 = 2^-41), the fp16
  gradients underflowed, and `grad_norm` was EXACTLY 0.0 for 88% of rows
  from step 214 on, while `stepped=1` throughout. 1,600 optimiser steps on
  identically zero gradients. The loss still fell (29.7 -> 8.6) because
  Adam's `weight_decay=1e-4` keeps shrinking weights toward a trivial
  all-background predictor. `lc_cls` averaged 24.0 vs `loc_cls` 1.49 in the
  first window and stayed 3-4x `loc_cls` throughout.

With PAC off, teacher off and align identically zero, **LC alone
reproduces the failure. The fault is inside LC/CSSM, not in its coupling
to PAC or the teacher.** Divergence is conclusive under the 7.11 bound.
**Rung 1's control value is spent** and it is not worth running.

#### Rung 3: clean on BOTH seeds — and what that does NOT license

567757 and 567758 both ran the full 2,000 steps: scale never below 1.0
(held 128 / 256), zero zero-gradient rows, 9 and 8 skips, all three cls
terms decaying together (0.82/0.83/0.79 -> 0.35/0.28/0.28).

So **the arm with LESS machinery is the one that dies.** Because PAC
consumes the LOCAL branch's outputs and not LC's, this CANNOT be PAC
shielding LC through the forward path. **Do not write that PAC stabilises
LC.** The measured claim, and the only one licensed here, is: **adding PAC
postpones failure beyond 2,000 steps.** Survival at 2,000 steps was
pre-registered as non-conclusive and clears only the fast mode.

**SHARED-TRUNK GRADIENT HYPOTHESIS — UNTESTED.** L_LC, L_PAC and L_local
all backpropagate into the same encoder/LC parameters, so removing L_PAC
changes the resultant gradient the shared trunk receives. That is the
remaining channel by which PAC could matter without being downstream of
LC. It is a HYPOTHESIS. Sec 7.15's gradient-cosine experiment tests it
directly and needs no training run.

#### FOURTH FAILURE PHENOTYPE — mode C, GRADIENT DEATH

The catalogue is now three modes, and they are distinct:

| mode | signature | detected by |
|---|---|---|
| A fast forward failure | loss goes non-finite within a few hundred steps | skip-rate breaker |
| B slow divergence | loss stays FINITE and climbs over thousands of steps | loss-trend breaker |
| **C gradient death** | **loss stays finite and FALLS; scale collapses; grad_norm exactly 0; no learning** | **scale-floor + zero-grad rules (new)** |

Mode C **demonstrates that the loss trend alone is insufficient**: the
trend was *favourable* the entire time. Note the standing pattern — this
is the **third** time a gate built for the previous mode missed the next
one (skip-rate missed B; the trend breaker missed C). Prefer
failure-AGNOSTIC observables. The instrument panel below exists for that
reason.

#### New gate rules, verified by replay against 13 CSVs on disk

* **SCALE FLOOR** — ABORT below 1e-6, WARN below 1.0. Justification: the
  scaler exists to LIFT gradients into fp16 range, so below 1.0 it is
  ATTENUATING them, which is never healthy. Measured: 6 clean runs hold
  128/128/256/512/512/512; every collapsed run passes far below.
  **ABORT is set at 1e-6 rather than 1.0 for one reason:** scale<1.0 fires
  on 561546 at step 6968, but 561546 aborted naturally at 7667 and job
  567949 REPLAYS that trajectory to test the 7.14 slow-mode rules. 1e-6
  fires at 7656 -- 11 steps before the natural abort -- preserving the
  approach window. Do not raise this floor while 567949 is pending.
* **ZERO-GRAD** — ABORT if >50% of the trailing 200 STEPPED updates have
  `grad_norm` exactly 0.0. Requires `stepped=1`, so a legitimately skipped
  step can never trigger it. Verified in isolation (scale floor disabled):
  fires ONLY on 567755, at step 347; never on the five other failures nor
  on any of the six clean runs. Peak trailing fraction elsewhere: 0.030.

Replay result through the REAL gate object (`RuntimeGate`), all 13 runs:
scale-floor fires at 7656 / 364 / 237 / 141 / 168 / 99 on the six
collapsing runs; 559640 is still caught by the pre-existing loss-trend at
18335 (unchanged); **all six clean runs run to the end of their CSV.**

**PERMANENT INSTRUMENT PANEL**, now on EVERY run and not only probe runs:
`scale`, `fraction_finite_grad`, `fraction_nonzero_grad`,
`median_grad_norm`, `loss`, `loss_trend`. That set makes mode C obvious in
one glance; its absence is the only reason 567755 was invisible for 1,600
steps.

#### 7.14 PARTIAL RESULT from probe 567950 — limits first

567950 (fresh, seed 2026, l40) aborted at **step 199** on the skip-rate
rule (66% non-finite, 136 skips). It captured the **FAST** mode, not the
slow one the 7.14 rules were written for.

**NO PRE-REGISTERED VERDICT APPLIES.** The run never reached step 500, so
the registered baseline window (steps 500-2000) cannot be computed and the
>=300-step lead requirement cannot be evaluated. `g_dt_proj` jumps 13x at
step 58, one to two steps before `y_pre_ln` explodes at 59-60; under the
registered <50-step rule that is **SIMULTANEOUS and inconclusive, NOT
confirmation**. **The 50-step threshold is NOT to be loosened now that the
result is known** -- that is precisely the move pre-registration exists to
prevent.

WHAT STANDS AS MEASURED (independent of the ordering question):

* `y_post_ln` is **FLAT at ~1480** from step 50 through the abort, while
  `y_pre_ln` goes 800 -> 1.03e6 -> `inf`. LayerNorm absorbed a
  thousand-fold internal explosion completely on the forward side. This
  **directly confirms the 7.14 LayerNorm-masking claim.**
* `z_fused` rises only ~8x (2,200 -> 17,000) across the same window.
  Input growth is therefore NOT what drives the blow-up: **the
  amplification is INSIDE CSSM, not inherited from its input.**
* Pattern match against the registered three-way: this is the
  **GAIN-FIRST** signature (z_fused ~stable, y_pre_ln rises), not
  activation-first and not gradient-first.

#### 7.15b PRE-REGISTERED — gradient-cosine experiment (no training)

Registered BEFORE `tools/grad_cosine.py` was run. Tests the shared-trunk
hypothesis directly: load a checkpoint, take ONE fixed batch, backward
each of L_LC / L_PAC / L_local separately, and restrict the gradients to
the parameters UPSTREAM of all three heads (head-specific params excluded,
since their gradients are disjoint by construction and would drag every
cosine toward 0 for a trivial reason).

REGISTERED INTERPRETATION:

* `cos(g_LC, g_PAC) < 0` -> PAC partially OPPOSES LC's gradient on the
  shared trunk; removing it leaves LC unopposed.
* `cos(g_LC, g_PAC) > 0` -> PAC REINFORCES the same direction, so the
  explanation is scale or conditioning, NOT opposition.
* a large `||g_LC|| / ||g_LC + g_local||` once PAC is removed supports LC
  becoming disproportionately dominant.

Two checkpoints: EARLY and NEAR-WHERE-RUNG-2-FAILED. Whatever the sign, it
is one batch at one or two points on one trajectory -- indicative, not a
mechanism proof.

#### 7.15c PRE-REGISTERED — fast-vs-slow comparison, before the replay lands

567950 gave the FAST phenotype under full probe instrumentation; 567949
(replay of 561546 from step 6000, a40) will give the SLOW one. Registered
NOW, before the replay lands, so the comparison can fail:

**SAME MECHANISM AT DIFFERENT RATES** would require ALL of:
1. the same **gain-first** signature: `z_fused` roughly stable (< ~10x over
   the approach window) while `y_pre_ln` rises by >= 2 orders;
2. `y_post_ln` flat -- LayerNorm masking on the forward side, as measured
   in the fast run;
3. the same gradient groups leading: `g_a_log` and `g_dt_proj` onsetting
   before `g_b_proj` / `g_c_proj` / `g_gate_out`;
4. ordering RESOLVABLE at slow speed where it was simultaneous at fast
   speed -- i.e. the >= 50-step separations that the fast run compressed
   into 1-2 steps now appear, satisfying the 7.14 chain.

**DIFFERENT MECHANISMS** is indicated by ANY of:
1. the slow run showing **activation-first** (`z_fused` onsets before
   `y_pre_ln`) -- the fast run's amplification was internal, so an external
   driver in the slow run is a genuinely different story;
2. `y_post_ln` NOT flat in the slow run (LayerNorm not masking, so the
   forward path was informative all along and 7.3's failure needs another
   explanation);
3. a different leading gradient group (`g_b_proj`, `g_c_proj` or
   `g_gate_out` onsetting before both `g_a_log` and `g_dt_proj`);
4. the slow run reaching its abort with NO gradient-group onset at all,
   which would refute the 7.14 chain outright (the registered REFUTED
   condition) regardless of what the fast run showed.

Explicitly NOT to be done before the replay lands: no pre-norm arm, no
column-major fix, no reopening of scan correctness. The replay is the
experiment designed to test the registered slow-mode rules, and adding
arms before it lands repeats the four-lambda-arm mistake (sec 7.7).
