# Baseline inventory — checkpoint-verified clean references

Each released checkpoint evaluated **at its own authors' setting on its own
native dataset**, via official code, with only `root_dir`/`validate_dir`
repointed at our staged copies. No retraining, no forced common setting.

Last updated 2026-08-04. **All three rows complete.**

Eval jobs: V2X-ViT 557133 (47 min), CoBEVT 557203 (~1 h), Where2comm 557204
(2 h 27 min). All exited 0, all staged-file and scenario counts verified
against source before inference.

---

## The table

| model | checkpoint | native dataset | authors' setting | published target | our AP@0.5 | our AP@0.7 | graded | match |
|---|---|---|---|---|---|---|---|---|
| **V2X-ViT** | `v2xset_checkpoints/v2x-vit/net_epoch60.pth` | V2XSet test | **Noisy** — `async: true`, `loc_err: true`, `compression: 32`, `backbone_fix: true`, σ_xyz/σ_ryp 0.2 | 0.836 / 0.614 | **0.84** | **0.62** | graded | **✓ match** (within `%.2f` rounding) |
| **CoBEVT** | `v2xset_checkpoints/cobevt_lidar.zip` → `net_epoch60.pth` | V2XSet test | **Perfect** — `async: false`, `loc_err: false`, `compression: 32`, `backbone_fix: true` | 0.849 / 0.660 | **0.85** | **0.66** | graded | **✓ match** (AP@0.7 exact) |
| **Where2comm** | `v2xset_checkpoints/point_pillar_where2comm_v2xset.zip` → `net_epoch50.pth` | **OPV2V** test | **Noisy** — `async: true`, `loc_err: true`, `compression: 0`, `backbone_fix: false` | **none** | **0.86** | **0.60** | **ungraded** | n/a — no oracle exists |

### Eval scale

| model | dataset split | scenarios | ego frames |
|---|---|---|---|
| V2X-ViT | `v2xset/test` | 19 | 2,834 |
| CoBEVT | `v2xset/test` | 19 | 2,834 |
| Where2comm | `opv2v/test` | 16 (verified) | ~2,170 |

`opv2v/test_culver_city` (4 scenarios, ~550 frames) is a **separate** split and
is excluded — the checkpoint's config names `/data/opv2x/test`.

---

## Why each model is graded against the row it is

**V2X-ViT — Noisy.** Its config ships `async: true`, `loc_err: true`. Under
OpenCOOD's documented 3-stage V2XSet recipe (perfect at compression 0 → add
compression 32 on perfect → enable async+loc_err) this is **stage 3**.
Confirmed empirically: it reproduced the Noisy row, not the Perfect one.

**CoBEVT — Perfect, NOT Noisy.** Its config ships `async: false`,
`loc_err: false` with `compression: 32` and `backbone_fix: true`. That is
**stage 2** — compression added, still on the perfect setting. Grading it
against the Noisy row (0.811 / 0.543) would credit it with a spurious ~+0.12
AP@0.7. The `xyz_std`/`ryp_std: 0.2` entries in its config are inert because
`loc_err` is false.

**Where2comm — ungraded, and on OPV2V.** Its config names
`root_dir: /data/opv2x/train`, and OpenCOOD's README links its download from
the **OPV2V** Box while CoBEVT/V2X-ViT come from the V2XSET Box. The
`_v2xset` in the zip filename is wrong. OpenCOOD's OPV2V benchmark table has
**no Where2comm row** (all 18 entries checked), and the V2XSet 0.534 belongs
to a different dataset — so there is no oracle. Its clean AP is the reference
for **its own** degradation curve, not a reproduction claim.

---

## Clean-reference configs for the fault study

The exact configuration each model's fault sweeps degrade **from**. This is
the model at its own setting with no injected fault.

| model | clean-reference config | working copy |
|---|---|---|
| V2X-ViT | released config, `root_dir`/`validate_dir` → `v2xset/test`. Protocol as shipped: async true, loc_err true, compression 32 | `~/opencood-eval/` pattern; run 557133 |
| CoBEVT | released config, paths → `v2xset/test`. async false, loc_err false, compression 32 | `~/opencood-eval/cobevt/` |
| Where2comm | released config, paths → `opv2v/test`. async true, loc_err true, compression 0 | `~/opencood-eval/where2comm/` |

Each directory keeps `config.yaml.orig` alongside the repointed `config.yaml`;
`diff` between them is exactly the two path lines.

### The confound this creates, stated explicitly

**The three checkpoints were not trained under the same conditions.** CoBEVT
is Perfect-trained; V2X-ViT and Where2comm are Noisy-trained. A cross-model
robustness comparison at a common fault therefore partly measures *which model
saw noise during training*, not *which architecture is inherently robust*. A
noise-trained model is expected to degrade less under pose error — that is
what it was fitted for.

Two clean options, and they answer different questions:

1. **Each at its own setting** (what this table does) — every number is
   anchored to a published value, so the pipeline is verified. Cross-model
   deltas are **not** comparable.
2. **All at `async=false, loc_err=false`** — a genuinely fault-free common
   reference, comparable across models, but off-protocol for the two
   noise-trained checkpoints, so none of the published rows apply and all
   three numbers become ungraded.

Doing (1) first and (2) second is the honest sequence: (1) proves the harness,
(2) provides the comparable baseline. Whichever is used for the fault study
must be stated wherever a degradation number is reported.

---

## Environment

| | |
|---|---|
| `v2xvit-official` | py3.7, torch **1.12.1+cu113**, spconv-cu113, v2x-vit `setup.py develop`. GPU-validated on L40S (`sm_89` runs the `sm_86` cubin via CUDA minor-version compatibility). Produced the V2X-ViT row. |
| `opencood-official` | **clone** of the above + OpenCOOD. Cloned rather than mutated so the validated env stays pristine. |

`requirements.txt` was **not** installed wholesale for OpenCOOD: it pins
`numba==0.49.0`, which requires `numpy<1.21` and would have downgraded the
validated stack. `numba` is never imported anywhere in `opencood/` — the pin
is vestigial. Only `timm`, `easydict`, `scikit-image`, `tqdm` were added, with
`--no-deps`.

## Guards on every eval job

1. **CUDA arch assertion first** — device name, `sm_` capability, compiled
   arch list, a real matmul, and the `box_overlaps` extension import. A
   wheel/GPU mismatch fails in seconds rather than mid-run.
2. **Staged-file count verified** against the source before inference.
3. **Scenario count asserted** (V2XSet 19, OPV2V 16). A truncated eval would
   otherwise produce a plausible AP over the wrong denominator.
4. `--constraint="a40|a100|l40|l40s"` — the SKUs cu113 can execute.
   Blackwell (`sm_120`) and Hopper (`sm_90`) are excluded; they are also
   another group's partitions.
