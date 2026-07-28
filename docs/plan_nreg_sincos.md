# Plan — add the cos channel: Nreg 7 → 8

**Status: PLAN ONLY. No code written. Approve before implementing.**

Resolves RECON-5 (`docs/corabench_design.md` §1.5). Rewritten 2026-07-28
against the confirmed scope.

The system is **sin-only and self-consistent today**: `preprocessing.py`
encodes `sin(yaw_gt − yaw_anchor)` into channel 6, and both decodes invert it
with `anchor + asin(·)`. The anchor offset is already handled correctly. The
defect is solely that **one sine cannot represent a full angle**. So this is
**add the cos channel**, not a re-derivation of the encoding.

---

## 0. Three corrections to the stated scope

Flagged rather than silently propagated, since a plan with wrong paths is
worse than no plan.

| stated | actual |
|---|---|
| `preprocessing.py:247`, variable `tg` | **line 255**, variable `reg` — `reg[:, 6] = np.sin(g[:, 6] - an[:, 6])`. Line 247 is `reg[:, 1]`. The width literal `reg_t.reshape(h, w, a, 7)` is at **259** |
| `nms.py` is downstream, no change | **No `nms.py` exists.** NMS is `cpbench/utils/geometry.py:125 nms_bev`. The *conclusion* holds — it consumes decoded radians and needs no change |
| `visualize.py` is eval-only, update it | **No `visualize.py` exists.** The nearest file is `src/visualisation.py`, and it contains **no yaw decode at all** (no `asin`/`arcsin`/`atan2`/`[:, 6]`). Nothing to update; it cannot lie about yaw because it never reads it |
| "teacher reg head … bumps 7→8" | **There is no teacher reg head.** `TeacherBranch` wraps `LCModule` and emits a *feature*; `align_loss` is MSE on features. The only `reg_head` in the tree is `cpbench/models/heads.py:68` |

Net effect: the mechanical tail is **one** `reg_head` declaration, not two, and
there are no eval-only files to chase.

**Consequence: the teacher drops out of the coordinated set.** It owns exactly
one submodule (`self.lc = copy.deepcopy(lc)`, an `LCModule`), returns
`f_teacher` — a *feature* — and `align_loss` is MSE on features. It never
decodes a box, so it has no yaw convention to keep in sync. The coordinated
set is **four sites** (§1); the teacher is not one of them, and the width bump
reduces to the single parameterised `DetectionHead` declaration plus its two
instantiations in `corabench/models/cora.py:105-106`.

---

## 1. The four coordinated sites

One convention: **`sin` = channel 6, `cos` = channel 7, decode =
`anchor + atan2(ch6, ch7)`.**

| # | site | role | package |
|---|---|---|---|
| 1 | `cpbench/data/preprocessing.py:255` | encode — add `reg[:, 7] = cos(Δ)` | **shared** |
| 2 | `cpbench/training/losses.py` + `corabench/training/losses.py:121` | reg loss — regress both channels (see §2) | shared + corabench |
| 3 | `corabench/fusion/pac.py:183` | decode, **autograd** | corabench |
| 4 | `cpbench/data/postprocessing.py:82` (+ docstring :35) | decode, **numpy eval** | **shared** |

**Sites 3 and 4 are a matched pair.** Same formula, one in autograd, one in
numpy. Change one without the other and the model optimises a different yaw
than AP is scored on — a silent corruption no shape test catches.

**Verified interchangeable:** `|torch.atan2 − np.arctan2|` across sign
quadrants, both axes and the ±π boundary has max divergence **1.14e-07**, which
is fp32 round-off.

---

## 2. Reg-loss decision: **unit-norm PENALTY, not normalise-before-decode**

### 2.1 Normalising before decode is provably a no-op — do not do it

`atan2` is **homogeneous of degree 0**: `atan2(k·s, k·c) = atan2(s, c)` for all
`k > 0`. Verified across four decades of scale, all returning `0.348941827`
exactly. So `f(v) = atan2(v/‖v‖)` and `g(v) = atan2(v)` are the **same
function** — identical value *and* identical gradient. Measured in torch at
‖v‖ = 0.1: `|grad|` is **9.997 either way**.

Normalising would be code that reads as a safeguard and does **nothing**. This
project has now been bitten four times by exactly that shape (fp16 focal clamp,
`asin` clamp, `asin(tanh)`, `π·tanh`). Adding a fifth is not acceptable.

### 2.2 The real hazard: `atan2`'s gradient is `1/‖v‖`

```
d/ds atan2(s,c) =  c/(s²+c²)      d/dc atan2(s,c) = −s/(s²+c²)
|∇| = √(s²+c²)/(s²+c²) = 1/‖v‖
```

| ‖(sin,cos)‖ | 1.0 | 0.5 | 0.1 | 0.03 | 0.01 |
|---|---|---|---|---|---|
| \|∇\| (torch) | 0.9997 | 1.999 | 9.997 | 33.32 | 99.97 |

The singularity has **moved to the origin**, and it is genuinely milder than
what it replaces: at ‖v‖ ≈ 1 the gradient is **1**, against `asin`'s **707** at
its clamp bound. But an unbounded prediction drifting toward `(0, 0)` restores
the same class of blow-up.

### 2.3 Why plain smooth-L1 on both channels is not sufficient

At **positive** anchors the target lies on the unit circle, so per-channel
smooth-L1 does pull `‖v‖ → 1`. But the reg loss is **positive-anchors-only** —
measured **~44 of 70,400 anchors per frame** — while
`pac.py::_decode_params` decodes at **every cell** to build the PE. The
~70,356 unsupervised cells are free to drift to the origin, and those are
exactly the ones feeding the decode.

### 2.4 Decision — DENSE UNIT-NORM PENALTY (settled 2026-07-28)

**Decided: smooth-L1 on both channels (positive anchors) + a dense
`(‖v‖ − 1)²` penalty over all cells. NOT normalise-before-decode.**

Briefly reversed to normalisation and then reversed back, on the grounds that
a **soft constraint with a bounded gradient everywhere beats an exact
constraint with a singularity**, given this model's history: every failure
this session has been the same family — `asin` at ±1, `1/E` in the scan,
`tanh` saturation, `dt` unbounded. `v / ‖v‖` at `v ≈ 0` is that family again,
and the head starts near zero, so it is reachable at initialisation.

**Decode is unaffected either way — this is the key fact.** `atan2` is
homogeneous of degree 0: `atan2(0.342, 0.940)`, `atan2(0.0342, 0.0940)`,
`atan2(3.42, 9.40)` and `atan2(0.000342, 0.000940)` all return
**0.348941827** exactly. So the decoded yaw does not depend on ‖v‖ at all, and

> **the penalty's only job is keeping the head well-conditioned — it is not
> load-bearing for correctness.**

That is what makes it the safe choice: if `λ_norm` is mis-tuned, the yaw is
still decoded correctly; only the loss landscape changes.

Two further facts on record:

- At exactly `(0, 0)` both decodes return `0.0` — finite and deterministic,
  not NaN. The decode never breaks; it returns an arbitrary angle where there
  is no angle to be correct about.
- Normalisation would not have introduced a *new* singularity, but it would
  have computed the same one through worse intermediates: since
  `atan2(v/‖v‖) ≡ atan2(v)`, the total gradient is identical (`|grad| = 9.997`
  both ways at ‖v‖ = 0.1), but the normalised path routes it through
  `d(v/‖v‖)/dv` with its `1/‖v‖³` terms, against `atan2`'s single `1/‖v‖²`.

**Form:**

```
L_yaw_norm = λ_norm · mean( (‖(reg[:,6], reg[:,7])‖ − 1)² )      over ALL cells
```

Dense, because the smooth-L1 term supervises only **~44 of 70,400 anchors per
frame** while `_decode_params` decodes **every** cell — the sparse loss is
exactly what leaves the decoded cells unconstrained.

**`λ_norm = 1.0` to start, and log its per-step contribution.** At ‖v‖ = 0.9
the term is 0.01 against a typical `reg` of ~0.24 — about 4%, enough to bind
without dominating. It must be logged: `u_reg = 1e-4` was measured at
**1.5e-5 %** of the objective, never bound anything, and nobody noticed
because it was never reported (RECON-2).

### 2.5 The rejected alternative, kept for the record

**Smooth-L1 on both channels (positive anchors) + a DENSE unit-norm penalty
over all cells:**

```
L_yaw_norm = λ_norm · mean( (‖(reg[:,6], reg[:,7])‖ − 1)² )      over ALL cells
```

Dense, because the sparse loss is precisely what leaves the decoded cells
unconstrained.

**Starting `λ_norm = 1.0`, and log its per-step contribution.** At ‖v‖ = 0.9
the term is 0.01 against a typical `reg` of ~0.24 — about 4%, enough to bind
without dominating. The value must be reported per step: `u_reg = 1e-4` was
measured to contribute **1.5e-5 %** of the objective, i.e. it never bound
anything, and nobody noticed because it was never logged (RECON-2). Do not
repeat that.

---

## 3. `reg_dim` parameterisation — default 7, corabench sets 8

No `nreg` / `reg_dim` / `yaw_encoding` key exists in any YAML today; the widths
are eight literals:

| file | literal |
|---|---|
| `cpbench/models/heads.py:68` | `Conv2d(in_channels, num_anchors * 7, 1)` |
| `cpbench/data/preprocessing.py:259` | `reg_t.reshape(h, w, a, 7)` |
| `cpbench/data/postprocessing.py:64,65` | `reshape(a, 7, h, w)`, `reshape(-1, 7)` |
| `cpbench/training/losses.py:128-130` | `reshape(b, a, 7, h, w)`, `reshape(-1, 7)` |
| `corabench/training/losses.py:121` | **second** `smooth_l1_reg_loss` |
| `corabench/models/cora.py:117` | `PACModule(ncls_ch, num_anchors * 7, …)` |
| `corabench/fusion/pac.py:154` | `reg_map.reshape(b, a, 7, h, w)` |

`anchors.reshape(-1, 7)` (`postprocessing.py:66`, `preprocessing.py:229`) is
**BOX-7** — the tuple `(x,y,z,l,w,h,yaw)` — and **stays 7**.

**Why the default must be 7:** `cpbench/data/postprocessing.py` is shared by
**all five** packages, and `lgcpbench/perception/opencood/adapter.py` loads
**real released OpenCOOD checkpoints** (`hypes_yaml` + `checkpoint`,
`strict=False`) decoded through this exact `BoxDecoder`
(`lgcpbench/perception/decode.py:37`). Those weights have `num_anchors * 7`
regression heads. A global switch breaks every pretrained model lgcpbench
exists to benchmark — **at runtime, not in tests**, because the tests have no
checkpoints.

**Measured: zero yaw-value assertions exist in any of the four idle packages.**

| package | decode-touching test files | yaw-value assertions |
|---|---|---|
| lgcpbench | 3 | **0** |
| cobevtbench | 7 | **0** |
| w2cbench | 3 | **0** |
| v2xvitbench | 3 | **0** |

So with `reg_dim` defaulting to 7, **no golden in any idle package moves and
the layering test never goes red.**

### 3.1 The parameterisation lands as its own step, verified separately

**Step A** — thread `reg_dim: int = 7` through the shared sites. **No encoding
change. Run the full suite across all five packages plus
`cpbench/tests/test_layering.py` before and after.** Both must be green and
identical. If anything moves, the parameterisation is not inert and the design
is wrong before any encoding work starts.

**Step B** — only then the cos channel, the loss term, and both decodes.

---

## 4. Four-site diff sketch

```diff
# 1. cpbench/data/preprocessing.py:255  (encode, shared)
             reg[:, 6] = np.sin(g[:, 6] - an[:, 6])
+            if reg_dim >= 8:
+                reg[:, 7] = np.cos(g[:, 6] - an[:, 6])
@@ :259
-                "reg_target": torch.from_numpy(reg_t.reshape(h, w, a, 7))}
+                "reg_target": torch.from_numpy(reg_t.reshape(h, w, a, reg_dim))}

# 2. reg loss (cpbench/training/losses.py AND corabench/training/losses.py:121)
-    reg_pred = reg_map.reshape(batch, n_anchors, 7, height, width)
+    reg_pred = reg_map.reshape(batch, n_anchors, reg_dim, height, width)
     # smooth-L1 over positive anchors covers ch6 AND ch7 unchanged
+    # DENSE unit-norm penalty, all cells (see plan section 2.4):
+    norm = torch.linalg.vector_norm(reg_map_sincos, dim=ch)      # (B,A,H,W)
+    l_yaw_norm = lambda_norm * ((norm - 1.0) ** 2).mean()

# 3. corabench/fusion/pac.py:183  (decode, autograd)
-        alpha = anch[:, 6] + torch.asin(reg[:, 6].clamp(-1 + 1e-6, 1 - 1e-6))
+        alpha = anch[:, 6] + torch.atan2(reg[:, 6], reg[:, 7])

# 4. cpbench/data/postprocessing.py:82  (decode, numpy eval) -- MUST MATCH 3
-        boxes[:, 6] = an[:, 6] + np.arcsin(np.clip(rg[:, 6], -1.0, 1.0))
+        boxes[:, 6] = an[:, 6] + np.arctan2(rg[:, 6], rg[:, 7])
```

Also: `heads.py:68` width, `cora.py:117` / `pac.py:154` PAC width, and
**delete the now-dead `asin` ε-clamp comment block** in `pac.py` (96fbb2a) so
the next reader is not told about a guard that no longer exists.

`atan2(0,0)` needs no guard: torch returns `0.0` with gradients `(0.0, −0.0)`,
both finite; numpy returns `0.0`. Initialising channel 7's bias to **1.0** is
still right — zero residual means `cos Δ = 1` — but it is a **modelling prior,
not a NaN guard**, and must not be described as one.

---

## 5. Goldens — corabench only, each hand-checkable

> A golden regenerated from broken code passes forever on the bug.

### 5.1 The reference line

```
yaw_gt = 30°, yaw_anchor = 10°  ->  Δ = 20°
  (sin, cos) = (0.342, 0.940)
  anchor + atan2(0.342, 0.940) = 10 + 20.0000 = 30.0000°   CORRECT
  anchor + atan2(0.940, 0.342) = 10 + 70.0000 = 80.0000°   <- a sin/cos SWAP
```

The swap check is the point: it fails by **50°**, loudly, so a golden produced
by a channel-swapped decode cannot pass silently.

### 5.2 Per-golden basis

| golden | hand-checkable basis |
|---|---|
| `test_yaw_roundtrip` | `gt=30, anchor=10 → (0.342, 0.940) → 10 + 20 = 30`. Also the swap: `→ 80 ≠ 30` |
| `test_full_circle_recovered` | `gt=200, anchor=10 → Δ=190 → (−0.174, −0.985) → 10 + 190 = 200`. **The old decode returns `10 + asin(−0.174) = 0`** — 200° error. Proves the fix |
| `test_flipped_heading_distinguished` | `Δ=20` and `Δ=160` share `sin=0.342`, differ in `cos` sign (+0.940 / −0.940). Old: both → 30. New: 30 and 170 |
| `test_both_decodes_agree` | `pac.py` (torch) vs `postprocessing.py` (numpy) on identical input, tol 1e-6. Neither is oracle; *disagreement* is the signal. Current headroom 1.14e-07 |
| `test_agree_where_asin_was_valid` | For `\|Δ\| < 90°` the old encoding is correct, so both **must** agree. Pins the new scheme as a strict generalisation |
| reg width 7→8 shape assertions | **Arithmetic, not a golden**: 3 centre + 3 log-dim + sin + cos = 8. State the decomposition in the assertion message |
| `cpbench/tests/test_detection_loss.py` (6) | **Construct** a target where ch7 contributes a known smooth-L1 term; check the delta by hand |
| `corabench/tests/test_cssm.py` | **Must not move** — no yaw dependency. If it does, something is wrong |

### 5.3 Not at risk — verified

Every AP assertion in the suite is a hand-built TP/FP unit test that never
invokes the yaw decode (`corabench/tests/test_metrics.py:18,28`,
`test_logger.py:35,37`, `cpbench/tests/test_comms_metrics.py:182`,
`lgcpbench/tests/test_faults_end_to_end.py:407`). **No end-to-end AP golden
exists.** Re-confirm before landing.

---

## 6. Order

1. **Step A** — `reg_dim` parameterisation, default 7, no encoding change.
   Full suite × 5 packages + layering green **before and after**, identical.
2. Write §5 tests against `reg_dim=8`. **Confirm red.**
3. Encode (site 1).
4. **Both decodes together** (sites 3 and 4). Never one without the other.
5. Reg loss + norm penalty (site 2); log its contribution per step.
6. Head width, PAC width, channel-7 bias; delete the dead `asin` clamp block.
7. Re-run all five packages + layering. Only then a corabench run.

## 7. Not fixed by this

- `dt_max = 0.2` remains a control (RECON-4), removed and re-tested separately.
- AP becomes *meaningful*, not *good*. The first number is still first-epoch.
