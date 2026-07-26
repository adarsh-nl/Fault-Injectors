# Plan — yaw encoding: Nreg 7 → 8, `atan2(sin, cos)`

**Status: PLAN ONLY. No code written. Approve the critical path before implementing.**

Resolves RECON-5 (`docs/corabench_design.md` §1.5). Written 2026-07-27 after
jobs 549449 / 549494 established that the yaw decode, not the CSSM scan, is
the live blocker.

---

## 1. Why, in one paragraph

`reg[:, 6]` is a raw linear head output supervised against `sin(Δyaw)`, decoded
with `asin`. Three compounding defects: nothing constrains it to `[-1, 1]` so
the decode clamp fires as ordinary behaviour; `asin` covers only
`[-π/2, π/2]`, so half the yaw circle is unrepresentable; and
`sin(Δ) = sin(π-Δ)` makes heading **180° ambiguous**, which is why any AP
computed today is meaningless. `atan2(sin, cos)` dissolves all three at once.

Two numerical workarounds were measured and rejected before choosing this:

| candidate | why it failed | measurement |
|---|---|---|
| `asin(tanh(r))` | `tanh` saturates to **exactly 1.0** in fp32 at \|r\| ≥ 10, so `asin(1.0)` gives a **NaN** derivative — the singularity relocated, not removed. Distorts the decode inside `[-1,1]` by −8.3% at 0.5 and −44.8% at 1.0 | `head/reg_map` amax **82.75**; `reg[:, 6]` amax **17.9** |
| `π·tanh` bound at the head | starts in the dead-gradient zone from step 0 at amax 17.9, so yaw never learns — a *misleading* AP, not a throwaway one | same |

Both were the clamp trap again (fp16 focal clamp → `asin` clamp → `dt_max` →
these). **The encoding is wrong, not the numerics.** Stop patching it.

---

## 2. Scope: two different sevens, only one moves

- **BOX-7** — the box tuple `(x, y, z, l, w, h, yaw)`. Anchors, GT, IoU, NMS,
  rasterisation, `utils/geometry.py`, `data/samples.py`, `data/rasterize.py`.
  **STAYS 7.** `anchors.reshape(-1, 7)` and `anchors[..., 6]` are boxes.
- **REG-7 → 8** — the *encoding* the head predicts and the assigner supervises.
  **Only this moves**: `(dx, dy, dz, dl, dw, dh, sin Δ)` → `(…, sin Δ, cos Δ)`.

Getting this distinction wrong doubles the diff and breaks NMS.

---

## 3. The hard constraint — this CANNOT be a flag day

`lgcpbench/perception/opencood/adapter.py` loads **real released OpenCOOD
checkpoints** (`hypes_yaml` + `checkpoint`, `strict=False`) and decodes them
through the shared `cpbench.data.postprocessing.BoxDecoder`
(`lgcpbench/perception/decode.py:37`). Those weights have `num_anchors * 7`
regression heads. **Changing the shared decoder to expect 8 breaks every
pretrained model lgcpbench exists to benchmark.**

Therefore: add `yaw_encoding: "sin" | "sincos"` to `DetectionHead`,
`TargetAssigner` and `BoxDecoder`, **defaulting to `"sin"`**. Only corabench
sets `"sincos"`. Every other package keeps its current path bit-for-bit.

Shared-code consumers, for reference:

| class | consumers |
|---|---|
| `BoxDecoder` | corabench, **lgcpbench**, cobevtbench, w2cbench, v2xvitbench — *all five* |
| `TargetAssigner` | corabench, cobevtbench, w2cbench, v2xvitbench |
| `DetectionLoss` | cobevtbench, w2cbench, v2xvitbench |

---

## 4. CRITICAL PATH — what unblocks a corabench run

**8 edits across 2 packages.** This is the whole experiment-unblocking change.

| # | file | edit |
|---|---|---|
| 1 | `cpbench/models/heads.py:68` | `reg_head = Conv2d(in, num_anchors * (7 or 8), 1)`, switched on `yaw_encoding` |
| 2 | `cpbench/data/preprocessing.py:255` | `reg[:, 6] = sin(g6 - a6)` → also `reg[:, 7] = cos(g6 - a6)` |
| 3 | `cpbench/data/preprocessing.py:259` | `reg_t.reshape(h, w, a, 7 → nreg)` |
| 4 | `cpbench/data/postprocessing.py:64-65` | `reshape(a, 7 → nreg, h, w)`, `reshape(-1, 7 → nreg)` |
| 5 | `cpbench/data/postprocessing.py:82` | `arcsin(clip(rg[:,6]))` → `arctan2(rg[:,6], rg[:,7])` |
| 6 | `cpbench/training/losses.py:128-130` | reg reshapes → `nreg` |
| 7 | `corabench/training/losses.py:121` | **corabench's second `smooth_l1_reg_loss`** — same reshape. Easy to miss; there are two implementations |
| 8 | `corabench/models/cora.py:117` + `corabench/fusion/pac.py:154,183` | `PACModule(ncls_ch, num_anchors * 8, …)`; `reg.reshape(b, a, 8, h, w)`; `asin(clamp(...))` → `atan2(reg[:,6], reg[:,7])` |

**Both decoders change together (5 and 8).** `postprocessing.py` is the
inference/AP path, `pac.py` is the autograd path. Changing one and not the
other leaves them disagreeing — which is exactly the split that made
`asin(tanh)` unattractive.

Then: **anchor init for the new channel.** `cos Δ = 1` at zero residual, so
`reg_head.bias` for channel 7 should init to **1.0**, not 0 — otherwise every
anchor starts predicting `atan2(0, 0)`, which is `0` but with an undefined
gradient. Verify `atan2(0, 0)` behaviour explicitly; add an ε or bias-init.

---

## 5. LAYERING-KEEP-GREEN — the four idle packages

With the opt-in default these should need **no functional change**, only:

- shape docstrings `(B, A*7, H, W)` in `corabench/fusion/adaptive.py:38` and
  `observation/locations.py` × 4 (`corabench`, `cobevtbench`, `w2cbench`,
  `v2xvitbench`) — corabench's becomes `A*8`, the rest stay `A*7`.
- `lgcpbench/perception/{protocol,decode,native}.py`,
  `opencood/adapter.py` — docstrings only; must keep decoding 7.
- `cpbench/tests/test_layering.py` — re-run; no new cross-package imports are
  introduced, so it should stay green by construction.

**Audit, don't assume.** Every test listed in §6 must be *run* to confirm it
exercises the default path.

---

## 6. Goldens — the real risk in this change

> **A golden regenerated from broken code passes forever on the bug.**

No golden may be regenerated by running the new code and recording the output.
Each needs an oracle that is independent of the implementation.

### 6.1 Hand-computed oracle table

These values are trigonometry, computable with a calculator, and depend on no
code in this repository. **Put them in the test as literals.**

| Δyaw | `sin` | `cos` | `atan2` (new) | `asin` (old) | old error |
|---|---|---|---|---|---|
| 0.6000 | 0.5646 | 0.8253 | **0.6000** | 0.6000 | 0.0° |
| 2.5416 | 0.5646 | **−0.8253** | **2.5416** | 0.6000 | **111.2°** |
| 2.5000 | 0.5985 | −0.8011 | **2.5000** | 0.6416 | 106.5° |
| −2.0000 | −0.9093 | −0.4161 | **−2.0000** | −1.1416 | 49.2° |
| 3.0000 | 0.1411 | −0.9900 | **3.0000** | 0.1416 | 163.8° |
| −2.8416 | −0.2955 | −0.9553 | **−2.8416** | −0.3000 | 145.6° |

The row pair **0.6000 / 2.5416** is the 180°-ambiguity proof: identical `sin`,
opposite `cos` sign. Under the old encoding both decode to 0.6 and one is
111.2° wrong. This pair alone justifies the change and needs no code to check.

### 6.2 Required new tests, each with its independent justification

| test | oracle |
|---|---|
| `test_yaw_roundtrip_is_exact_over_the_full_circle` | **Identity**: encode Δ then decode must return Δ, for Δ across `[-π, π]`. Correct by definition of `atan2(sin θ, cos θ) = θ`; no reference implementation involved |
| `test_sin_only_encoding_cannot_distinguish_a_flipped_heading` | **Table §6.1 row pair.** Assert the *old* path collapses 0.6 and 2.5416 to the same value and the *new* path separates them. Hand-checked literals |
| `test_yaw_decode_matches_hand_computed_values` | **Table §6.1**, literals only |
| `test_atan2_gradient_is_finite_at_the_origin_and_at_saturation` | `atan2` has no singularity off the origin; test `(0,0)` explicitly and assert the chosen guard. Independent of any golden |
| `test_pac_and_boxdecoder_agree_on_yaw` | **Cross-implementation**: the autograd decode and the numpy decode must return the same angle for the same input. Neither is the oracle for the other — *disagreement* is the signal |
| `test_sincos_and_sin_agree_where_asin_is_valid` | For `\|Δ\| < π/2` the old encoding is correct, so the two paths **must** agree. Pins that `sincos` is a strict generalisation, not a different model |

### 6.3 Goldens that must be re-derived, and how

| golden | independent justification |
|---|---|
| reg width `7 → 8` in shape assertions | **Arithmetic, not a golden**: 3 centre + 3 log-dim + sin + cos = 8. State the decomposition in the assertion message |
| `cpbench/tests/test_detection_loss.py` (6 tests) | Loss values change only through the extra channel. Verify by **constructing** targets where the cos channel contributes a known smooth-L1 term and checking the delta by hand, not by recording the new number |
| `corabench/tests/{test_fusion, test_losses, test_dataset}` | Shape-driven; re-derive from the 8-channel decomposition |
| `corabench/tests/test_cssm.py` | Should be **untouched** — no yaw dependency. If it moves, something is wrong |

### 6.4 Goldens that are NOT at risk — verified

All AP assertions in the suite are hand-constructed unit tests that build TP/FP
lists directly and never invoke the yaw decode:
`corabench/tests/test_metrics.py:18,28`, `corabench/tests/test_logger.py:35,37`,
`cpbench/tests/test_comms_metrics.py:182`,
`lgcpbench/tests/test_faults_end_to_end.py:407`.
**No end-to-end AP golden exists**, which removes the worst class — a number
with no oracle at all. Confirm this still holds before landing.

---

## 7. Order of operations

1. Add `yaw_encoding` to the three shared classes, defaulting to `"sin"`.
   **Full suite must stay green with zero other edits** — that is the proof
   the opt-in is truly inert.
2. Write the §6.2 tests against `"sincos"`. They fail. **Confirm red.**
3. Implement encode (#2, #3), then both decoders together (#5, #8).
4. Head dim and the `cos` bias init (#1, plus §4's note).
5. Losses (#6, #7) — remember there are **two** `smooth_l1_reg_loss`.
6. Re-run all five packages plus `test_layering.py`.
7. Only then a corabench run.

---

## 8. What this does NOT fix

- The `asin` ε-clamp (96fbb2a) becomes **dead code** in `pac.py` once `atan2`
  lands — remove it in the same change or it will confuse the next reader.
- `dt_max = 0.2` remains a control, still to be removed and re-tested
  separately (RECON-4).
- The SSD scan is **still uncommitted** in the working tree, and the commit
  `d264fd7` referenced on 2026-07-26 is not in this repository. Land that
  before starting this, or the two changes tangle.
- AP only becomes *meaningful*; it does not become *good*. The first number
  after this lands is still a first-epoch number.
