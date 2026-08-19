# CSSM/SSD correctness and dynamics verification

Date 2026-08-18. Harness: `tools/cssm_verify.py` (read-only wrt production
code). Jobs: 567820 (smoke, ps,main-cpu, hpc-node12), 567827 (all phases,
same node). Environment: `opencood-official` (Python 3.7, torch
1.12.1+cu113) — the environment training actually runs in. Full logs:
`logs/cssm-smoke-567820.out`, `logs/cssm-verify-567827.out`.

Evidence labels: **MEASURED** (a number in the logs), **DERIVED** (follows
mathematically from code read line-by-line), **INFERRED** (consistent with
measurements but not demonstrated), **UNRESOLVED**.

## 1. Executive conclusion

**The handwritten parallel SSD scan is mathematically correct, its backward
pass is correct, and its masking is exactly the causal recurrence it claims
to implement.** Forward, backward, chunking, chunk-boundary state carry,
gradient checkpointing, and the `-inf → exp` mask were verified against a
naive per-step sequential recurrence and a third, independently structured
chunked SSD implementation: agreement is at machine epsilon in fp64
(~1e-16 relative) and at fp32 rounding level (~1e-7 relative) in fp32,
for forward outputs and for every gradient (cosine similarity 1.0000000).
fp64 `gradcheck` passes for both inputs and parameters. **MEASURED.**

Two real findings, neither a scan-math error:

1. **The "column-major" cross-scan directions are not column-major**
   (`CSSM._directions`, gather/scatter inverse confusion, exact on any
   `h ≠ w` grid). On the real pooled 50×176 BEV, scan directions 3 and 4
   traverse the map in jumps of mean 72 cells; 0% of consecutive scan
   tokens are spatially adjacent. Internally consistent (all inputs
   permuted together, inverse restore correct), so numerically harmless —
   but two of four scan directions operate on a spatially scrambled
   sequence, contradicting spec A9. **MEASURED.**
2. **The scan is an unnormalized polynomial block of degree ≈ 3 in the
   activation scale** (measured exponent k = 3.0–3.4). Nothing normalizes
   `z_fused` before it; the LayerNorm sits after. Gradient norms into the
   scan grow at least cubically with activation scale (‖∂L/∂a_log‖ =
   5.5e10 at input scale 100 vs 1e-2 at scale 1). The recurrence itself is
   dynamically stable (state norm stationary over 8800 tokens). The most
   likely instability mechanism is therefore optimizer-mediated feedback
   through this cubic gain, not any property of the scan math. §13.

## 2. What the current code mathematically implements

From `corabench/fusion/cssm.py`, read line-by-line (**DERIVED**, then
**MEASURED** equivalent to this recurrence at machine epsilon):

Per scan direction, per token t along the flattened pooled-BEV sequence:

```
dt_pre_t = softplus(dt_raw_t + dt_bias)            dt_raw = dt_proj(z_fused)
dt_t     = dt_bound * tanh(dt_pre_t / dt_bound)    dt_bound = 0.2
A        = -exp(a_log)                              (D, N), strictly < 0
A_bar_t  = exp(dt_t ⊗ A)                            (B, D, N), in (0, 1)
B_bar_x_t= (dt_t * x_t) ⊗ B_t                       B_t = b_proj(z_fused)_t
h_t      = A_bar_t * h_{t-1} + B_bar_x_t            h_0 = 0
y_t      = Σ_n C_{t,n} h_t[·, ·, n]                 C_t = c_proj(z_i)_t
y        = y + d_skip * x
```

Shapes: x, dt_raw (B, L, D=256); b_in, c_in (B, L, N=16); h (B, D, N).
The parallel form computes this per chunk of 64 via
`logE = cumsum(dt·A)`, pairwise exponent `logE_t − logE_s` masked to the
lower triangle, `h_t = exp(logE_t)·h0 + Σ_{s≤t} exp(logE_t − logE_s)·b_s`,
with the previous chunk's last state as `h0`. The identity between that
form and the recurrence above is exact algebra (telescoping products);
confirmed numerically in §7. All computation in fp32 regardless of AMP
(`fp32_island`); everything is input-dependent except `a_log`, `dt_bias`,
`d_skip` (parameters). No in-place ops on the differentiable path.

Wrapper (`CSSM.forward`): avg_pool2d(2) → flatten → shared
dt_proj/b_proj/c_proj → 4 direction permutations → scan each → inverse
permutation → mean of 4 → LayerNorm → bilinear upsample.

## 3. Sequence axis and its semantic meaning

- Tokens are **pooled BEV spatial cells**: 100×352 feature map →
  avg_pool2d(2) → 50×176 → L = 8800 tokens. **DERIVED** (code), confirmed
  by shapes at runtime.
- The axis is **not** time and **not** agents. The agent axis is reduced
  *before* CSSM by `AttFusion2` over the 2-token {ego, collaborators}
  stack (`lc.py`). Latency/asynchrony never reaches the scan as an axis.
- Ordering is the raster flattening order (and three variants). It is
  physically meaningful (spatial adjacency) for the two row-major
  directions only — see §11 for the col directions.
- The operation is **not** permutation-invariant and is not meant to be:
  it is a causal recurrence per direction, the standard VMamba cross-scan
  trick for 2-D data (spec A9). Averaging 4 directions removes net
  left/right asymmetry in aggregate but each direction is strictly causal
  (**MEASURED**: perturbation at t=32 changes nothing before t=32,
  everything at/after; reversal changes output by 25% relative).
- Is causality along a spatial raster *justified*? It is an architectural
  convention (VMamba/Vim), not a physical claim; the 4-direction average
  is precisely the mitigation for its arbitrariness. This is category B
  (chosen, defensible, paper-silent) — with the exception in §11.

## 4. Masking / causal structure

The mask is `tril(ones(Lc, Lc))` applied to the pairwise **exponent**
(`masked_fill(-inf)` before `exp`), producing exact zeros in the decay
matrix for s > t. Entry (t, s) of the decay matrix is
`exp(Σ_{u=s+1..t} dt_u·A)` — the product of stepwise decays between s and
t; the diagonal is exp(0)=1, so b_t enters undamped, matching
`h_t = A_bar·h_{t-1} + B_bar_x_t` exactly. Masking on the exponent is the
only safe place: upper-triangle exponents are positive, so masking after
the exp would compute `exp(+big) = inf` and then `0·inf = NaN` (this is
the documented job-558108 lesson; the code comment matches the math).
`exp(-inf) = 0` also kills the gradient path to masked entries — correct,
they are non-dependencies. **DERIVED**, and **MEASURED** to reproduce the
sequential recurrence exactly.

Answer to the two-part question: (a) the mask corresponds *exactly* to the
intended selective-SSD causal recurrence — yes, measured. (b) Semantic
appropriateness for the axis: appropriate in the VMamba-convention sense
for the two raster directions; undermined for the two "col" directions by
the permutation defect (§11). The mask itself is never the problem.

## 5. Equation-by-equation comparison with intended SSD/Mamba-2

| Item | Implementation | Intended (paper / Mamba) | Class |
|---|---|---|---|
| A param | −exp(a_log), init log(1..16) | S4D-real, matches Mamba | A |
| Discretization A | A_bar = exp(ΔA) (exact ZOH) | Mamba | A |
| Discretization B | B_bar = Δ·B (Euler simplification) | Mamba's own simplification | A |
| Δ path | softplus(Linear_{C→C}(z_fused) + bias), bias init softplus⁻¹(U-log[1e-3,1e-1]) | Mamba uses rank-reduced dt_rank with specific weight init; paper silent | B (bias matches Mamba; full-rank Linear with default init is ours) |
| Δ bound | 0.2·tanh(Δ/0.2) | Neither paper nor Mamba; ours, documented (§7.1 history) | B |
| C source | c_proj(z_i) | Paper Eq. 8 explicit: C = Z_i | A |
| D skip | d_skip·x | Mamba standard; paper silent | B |
| Mask/scan | pairwise SSD, tril on exponent | SSD (Mamba-2) math | A (measured) |
| Chunking + h0 carry + reentrant ckpt | chunk 64 | equivalence required | A (measured, incl. bitwise-equal grads ckpt on/off) |
| fp32 island | scan in fp32 under AMP | Mamba practice (fp32 scan) | A/B |
| Pre-normalization | **none** before the scan; LayerNorm only after the 4-direction mean | Mamba blocks are always pre-normed (RMSNorm); paper silent for CoRA | **C** — see §13 |
| Cross-scan directions | row, row_r, "col", "col_r" | spec A9: VMamba 4 raster orders | **D** for col/col_r on h≠w — see §11 |
| pool 2 + bilinear upsample | ours, documented | paper silent | B |

Nothing in class E. The paper (per spec §1.3/§3) specifies only Eq. 8's
argument wiring, which the code follows; everything numerical is spec §3's
choice and is labeled as such there.

## 6. Naive sequential reference

`sequential_scan()` in `tools/cssm_verify.py`: an explicit per-token
Python loop implementing exactly the recurrence in §2 with the same
parameters, initialization, discretization, bound, and ordering — no
chunking, no mask, no einsum tricks; each timestep individually
inspectable. A closed-form analytic case (x=b=c=1, dt_raw=0 → geometric
sum) was also checked against the parallel implementation directly,
independent of both implementations: agreement 1.2e-16 rel (fp64),
6.3e-8 (fp32). **MEASURED.**

## 7. Forward equivalence results (**MEASURED**, log lines 6–39)

Parallel vs sequential, max relative error (‖Δ‖∞ / ‖ref‖∞):

| Case | fp32 | fp64 |
|---|---|---|
| tiny B1 L8 D4 N2 | 1.3e-08 | 1.1e-16 |
| analytic closed form | 6.3e-08 | 1.2e-16 |
| near-zero Δ (dt_raw−12) | 0.0 | 1.9e-18 |
| Δ at bound (dt_raw+12) | 1.2e-07 | 2.0e-16 |
| large inputs ×100 | 6.5e-08 | 2.4e-16 |
| long L=1024 | 1.3e-07 | 4.9e-16 |
| L=100 (chunk 64 ∤ L) | 1.4e-07 | 1.8e-16 |
| realistic B2 L8800 D256 N16 | 8.1e-07 | — |

All finite, all shapes equal. The worst elementwise relative number
(6.9e-2, realistic case) sits on a near-zero output element with absolute
difference 8e-6 against ‖y‖∞ = 10.3 — cancellation on a tiny value, not a
divergence. Mixed precision: the production config runs the scan inside
the fp32 island, so under AMP the scan's internal dtype **is** fp32 — the
mode verified here. The island's cast/enter/exit plumbing under real CUDA
autocast was separately verified in the job-558108 post-mortem and is
unchanged; re-verifying it needs a GPU and was deliberately not queued
behind the 26-deep GPU backlog (UNRESOLVED-by-choice, low risk).

## 8. Backward equivalence results (**MEASURED**, log lines 40–127)

Loss = mean(y²), backward through both implementations independently.
Every gradient — inputs x, dt_raw, b_in, c_in and params a_log, dt_bias,
d_skip — matches with cosine similarity 1.0000000 and relative error
1e-8–5e-7 (fp32) / ≤5e-16 (fp64), all finite, across: moderate shapes,
checkpoint on and off, Δ at the bound, inputs ×100 (where ‖grad‖ reaches
5.5e10 and the two implementations still agree to 1.9e-7), and a realistic
D=256 slice. The feared forward-match/backward-mismatch signature did
**not** occur.

- fp64 `gradcheck` (analytic vs numerical Jacobian): **True** for inputs
  and for parameters (via a pure-functional rebuild).
  `gradcheck` with checkpointing on fails only with torch 1.12's known
  "checkpointing is not compatible with .grad()" limitation — an API
  restriction, not a gradient error, because:
- checkpoint on vs off through `.backward()`: **bitwise identical**
  gradients (max|Δ| = 0.0 exactly, fp32 and fp64). The reentrant
  checkpoint path used in training is exactly correct.
- Full-CSSM-level (pool → projections → 4 directions → LayerNorm →
  upsample), parallel vs sequential scan inside the same wrapper: all 14
  parameter/input gradients match at 1e-8 level, cos 1.0. The inverse
  permutation, direction averaging, and projection plumbing are correct
  in backward as well as forward.

## 9. Reference implementation

`mamba_ssm` is not importable in `opencood-official` (ModuleNotFoundError)
and cannot be installed there (Python 3.7, torch 1.12, no-install rule).
**Reference implementation unavailable in the existing environment; the
sequential recurrence remains the independent correctness reference.**
As a partial substitute, a third implementation was written from the
published Mamba-2 minimal-SSD structure (segsum + chunkwise state passing,
block 32 ≠ production chunk 64) — an independent algorithmic path, though
not an independent codebase. All three implementations agree: parallel vs
ssd_ref 6.2e-16 rel (fp64), 2.2e-07 (fp32). **MEASURED.**

## 10. First point of divergence

None found above tolerance in any phase. The largest deviations are fp32
rounding (order-of-operations differences between cumsum-difference and
step products), location-tracked in the logs and not systematic.

## 11. Permutation / causal-structure findings

**The one defect found anywhere in this investigation** (**MEASURED**,
log lines 144–153): `CSSM._directions` computes
`col = (idx % w) * h + idx // w`. That is the *inverse* (scatter form) of
the true column-major gather permutation `(idx % h) * w + idx // h`; the
two coincide only when h = w — which is why unit-scale square tests never
caught it (verified: on 8×8 it *is* column-major; on 2×3 and 50×176 it is
not). On the production pooled 50×176 grid:

| direction | mean spatial step | frac adjacent |
|---|---|---|
| row, row_r | 1.97 | 99.4% |
| col, col_r (current) | **72.13** | **0.0%** |
| true column-major | 1.97 | 98.0% |

Consequences, stated precisely: the permutation is still a bijection, all
four inputs are permuted together, and the inverse restore is exact
(round-trip identity measured True), so **forward and backward are
numerically valid** — this cannot produce NaN/inf or wrong gradients. What
it breaks is semantics: two of the four scan directions traverse the BEV
in ~72-cell jumps, so the decay kernel's locality is meaningless along
them, and the intended vertical-neighborhood mixing of a cross-scan
simply does not exist (the transpose-equivariance probe shows a small
residual asymmetry, 3.3e-3 rel, though pooling boundaries contribute to
that number too). Classification: **D vs spec A9** ("column-major ±");
the paper itself never specifies scan orders. Relation to the instability:
**no demonstrated link** — it makes half the directions useless, not
explosive. Do not fix silently: a fix changes the learned function and
must be its own arm (§14).

Agent-permutation testing is **not applicable**: the scan never sees an
agent axis (§3). Causality tests: strict causality confirmed; reversal
sensitivity 25% rel per direction, by design symmetric in aggregate.

## 12. Long-horizon dynamical analysis (**MEASURED**, log lines 175–195)

Sequential recurrence instrumented over L=8800 at the measured training
regime (Δ rides the 0.2 bound: p99 → 0.197, sat_frac 0.13 by step 2k in
job 565550):

- ‖h_t‖ is **stationary**: log-slope over the second half ≈ +2e-5 to
  +8e-5 per token (i.e., ~0); ‖h‖ end/mid ratios ~1. True at input scales
  1, 10, 100 and for Δ at bound. **The recurrence has no autonomous
  long-horizon instability** — A_bar < 1 strictly, and underflow forgets.
- R_t = ‖B̄x‖/‖Āh‖: p99 = 1.2–3.7 → injection-dominated but bounded.
  (The R mean in the log is meaningless — inflated by t≈0 where h≈0;
  read the p99.)
- **Gain law**: with all scan inputs scaled together by a (as they are in
  the real model, where x, Δ, B derive from z_fused and C from z_i),
  ‖y‖ ∝ a^k with k = 3.2, 3.4, 3.0 across successive decades. The scan
  is effectively a **degree-≥3 polynomial nonlinearity** in the LC trunk
  activation scale — the only such block in the network, and the only
  block with no normalization on its input.
- Gradients inherit this: at a=100, ‖∂L/∂a_log‖ = 5.5e10,
  ‖∂L/∂(inputs)‖ ≈ 8e8, vs 1e-2 / 3e-2 at a=1 — and a_log, dt_bias are
  shared parameters accumulating over all 8800 tokens × batch ×
  4 directions.

## 13. Most likely failure mechanism

**INFERRED** (consistent with all measurements; not yet demonstrated at
training time): optimizer-mediated positive feedback through the
unnormalized cubic path. Any upward drift in z_fused magnitude is
amplified ~cubically into the gradients of everything feeding the scan
(conv branches, AttFusion2, dt/b/c projections, and the shared a_log /
dt_bias). Larger gradients → larger Adam updates → larger activations →
still-larger gradients. The post-scan LayerNorm hides the growth from the
*forward* loss until late, which matches the recorded signature: forward
activation instrumentation was not predictive (§7.3), failures appear
LC-first with lc_cls/pac_cls moving together (both downstream of the LC
trunk), and the trajectory is slow-then-sudden (loss-trend breaker at
step 7,667 in 561546). What this mechanism is *not*: a scan-math bug
(ruled out, §§7–10), autonomous state growth (ruled out, §12), Δ
saturation per se (closed, §7.1), or align (closed, §7.5).

## 14. Experiments now justified

1. **Per-step gradient-norm instrumentation** on a_log, dt_bias, b_proj,
   c_proj, dt_proj (cheap scalars, backward-side). The §13 mechanism makes
   a falsifiable prediction: these grow steadily *before* the loss trend
   breaks, unlike the forward magnitudes that failed as predictors in
   §7.3. This is the direct test.
2. The already-queued ablation ladder (rungs 2/3, jobs 567755–567758)
   discriminates whether the LC/CSSM path alone reproduces divergence —
   unchanged, and now with a mechanism it can confirm or refute.
3. If (1) supports the mechanism: a **pre-normalization arm** (norm on
   z_fused before the scan, Mamba-style) as a pre-registered single-change
   arm. Not implemented now — correctness-first discipline, and §13 is
   INFERRED, not MEASURED.
4. A **col-scan fix arm** (true column-major) as a *fidelity* experiment,
   pre-registered, expectations neutral on stability. Never bundled with
   (3).

## 15. Experiments no longer necessary

- Any further scan-math bug hunt: forward, backward, mask, chunk carry,
  checkpointing, and parameterization are verified to machine epsilon.
- mamba-ssm installation efforts (recorded unavailable; substitute agreed).
- Re-testing Δ saturation, GradScaler warmup, cssm_in forward-magnitude
  breakers, or loss_align as drivers — closed in spec §7.1/7.2/7.3/7.5;
  nothing here reopens them, and §12 independently confirms the recurrence
  is stable at Δ = bound.
- Sequence-reversal/causality bug hunts: causal structure measured exact.
