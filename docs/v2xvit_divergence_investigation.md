# V2X-ViT training divergence — investigation record

Running record of the investigation into why V2X-ViT training on V2XSet
diverges. Written for someone picking this up cold, or for me re-reading it
after a break. Chronological, with job IDs and file:line evidence throughout.

Last updated: 2026-08-04, after bisect job 556366.

---

## 1. Current state, in one paragraph

The V2X-ViT baseline trains cleanly for a full epoch (median loss 0.43–0.45)
and then, in the production harness, explodes during epoch 1 to a median of
~833 with a peak of 6.9e7. A 5,200-step bisect that crossed the same epoch
boundary using a simpler loop **did not reproduce it** — epoch 1 came out at
median 0.351, better than epoch 0. So the failure is real (observed twice at
full scale) but not yet reproducible on demand. **No fix is applied. No
60-epoch run is in flight.** Job **556564** (2 epochs through the real
`train.py` path) is queued to settle whether the production harness is the
trigger; it sits at priority 1 with an estimated start of 2026-08-05.

---

## 2. What the run is supposed to be

Corrected stage-1 protocol, taken from the reference's shipped Perfect-setting
config `v2x-vit/v2xvit/hypes_yaml/point_pillar_v2xvit.yaml`:

| setting | value | reference line |
|---|---|---|
| epochs | 60 | `:20` |
| batch size | 2 | `:19` |
| optimiser | Adam, lr 1e-3 | `:153` |
| schedule | multistep γ0.1, `[15, 50]` | `:159-161` |
| compression | **0** | `:84` |
| `backbone_fix` | **false** (end-to-end) | `:85` |
| wild_setting | `async: false`, `loc_err: false` | `:6,10` |

Oracle: clean **AP@0.7 0.712 / AP@0.5 0.882**.

These are already the package defaults (`trainer/default.yaml`,
`model/v2xvit.yaml`), so the launch passes no protocol override — only
`trainer.checkpoint_every=1` and `trainer.log_every=1`, which are operational.

### 2a. Corrections to earlier, wrong guidance (mine)

I previously recommended `compression=32` and `lr_steps=[15,65]`. **Both were
wrong.** They came from the *released checkpoint's* embedded config, which is
the **noisy/timedelay** model (`loc_err: true`, `async: true`) — a different
model from the Perfect-setting one that produced 0.712. Run 555051 was
launched on those wrong values because of me. The repo's `trainer/default.yaml`
already had the correct `[15, 50]`.

I also claimed the 0.712 model was trained with a frozen backbone. It was not:
`backbone_fix: false` in the shipped config, and the freeze helper's own
docstring says *"Fix the parameters of backbone during finetune on timedelay"*
(`point_pillar_transformer.py:48`).

---

## 3. The failure, as observed

### Run 555051 — `compression=32`, `lr_steps=[15,65]` (wrong protocol)
Healthy to ~step 340; `grad_norm` first crossed the clip of 35 at step 360;
loss 239 by step 440; 2203 by 560. Killed at 4h27m, 4 checkpoints kept.

### Run 555203 — 600-step smoke, `compression=0`, `[15,50]` (corrected)
Diverged **earlier**: healthy to ~290, first clip crossing at 295, loss 45 by
320, 719 by 345. So `compression` was not the cause — if anything the 32×
bottleneck slightly delayed it.

### Run 555923 — full corrected protocol, after the loss-fidelity fix
**Epoch 0 completed cleanly and was the best training seen: median 0.4340,
min 0.1009, max 16.40 over all 3,347 steps.** Then:

| epoch | median | min | max |
|---|---|---|---|
| 0 | 0.4340 | 0.1009 | 16.40 |
| 1 | **833.29** | 0.1303 | **68,753,640** |
| 2 | 536.32 | 140.06 | 78,814 |
| 3 | 657.95 | 196.08 | 359,790 |
| 4 | 744.07 | 261.02 | 14,175 |
| 5 | 815.65 | 213.34 | 6,927 |

Note epoch 1's **min is 0.1303**, lower than epoch 0's min — healthy steps and
catastrophic ones coexist in the same epoch. Killed at 6h31m, 7 checkpoints
kept (1020 MB preserved at `$HOME/v2xvit-results/555923`).

### Job 556366 — 5,200-step bisect across the epoch 0/1 boundary
**Did not reproduce.**

| epoch | n | median | max |
|---|---|---|---|
| 0 | 3,347 | 0.4532 | 16.34 |
| 1 | 1,853 | **0.3513** | **6.59** |

Zero batches above loss 1000. Global `max_abs_reg_target = 11.760`.

Epoch-0 medians agree between 555923 (0.434) and the bisect (0.453), so the
bisect is reproducing the same training — it simply does not blow up.

---

## 4. The hypothesis graveyard

Every one of these was proposed, then killed by measurement. Recorded so
nobody re-runs them.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Degenerate GT extents → `log(→0)` | **refuted** | 140,778 boxes scanned; min l/w/h = 3.633 / 1.789 / 1.301 m; zero below 0.1 m; zero exact zeros; zero negatives (job 555195) |
| 2 | `log` unguarded | **refuted** | `np.maximum(g[:,3:6], _EPS)` already at `preprocessing.py:312` |
| 3 | `compression: 32` | **refuted** | corrected run 555203 diverged *earlier* (step 295 vs 360) |
| 4 | Missing `backbone_fix` two-stage protocol | **refuted** | shipped config is `false`; and the two OPV2V CoBEVT checkpoints share **0 of 146** frozen-scope tensors bit-identically (job 555202), so the reference trains end-to-end per bench |
| 5 | Zero-GT frames → `n_pos=0` fallback | **refuted** | 0 of 6,694 train frames have zero GT; and the reference doesn't filter empty frames either (`voxel_postprocessor.py:116`) |
| 6 | A7 — our differentiable `grid_sample` warp vs a "discretised" reference warp | **refuted** | the reference uses the *same* bilinear `F.grid_sample` (`torch_transformation_utils.py:351-356`); `get_discretized_transformation_matrix` does **no rounding** — grep for `round\|floor\|ceil\|int(` in that file returns nothing. **Design-doc assumption A7 is factually wrong and should be retired.** |
| 7 | Unconditional force-match creating huge reg targets | **refuted as the mechanism** | bisect measured `max_abs_reg_target = 11.76` globally — benign. The *fidelity gap* is real (see §5) but is not producing bad targets on this data |
| 8 | Gradient clipping not firing | **refuted** | `clip_grad_norm_` **returns the pre-clip norm**, which is what we log. A logged 6e6 means the raw grad was 6e6 and *was* rescaled to 35. Clipping works |

A corollary of #8: a skip-step-on-nonfinite guard would never fire here.
**All 21,059 logged steps of 555923 were finite.** This is optimisation
divergence, not a numerical blow-up, and gradient guarding will not fix it.

---

## 5. Fidelity gaps found in the reference comparison

Real differences from the code that produced 0.712. Only the first is fixed.

1. **FIXED — classification loss.** Ours used batch-global normalisation and
   excluded `-1` "ignore" anchors. The reference normalises **per sample** then
   divides by batch size (`point_pillar_loss.py:106-108,124,137`) and has **no
   ignore band** — `pos_equal_one` is binary, so every non-positive anchor is a
   negative (`:100-101`). `cpbench/training/losses.py` now matches, verified by
   a golden test that transcribes the reference and asserts equality within
   `rtol=1e-5`, including the batch-with-an-empty-sample cases. Suite: 429
   passed. Committed as `d16c518`.
2. **OPEN — force-match guard.** The reference force-matches a GT to its best
   anchor only when `iou > 0` (`voxel_postprocessor.py:141-144`, comment:
   *"make sure all highest iou is larger than 0"*). Ours does it
   unconditionally (`preprocessing.py:297-300`). Measured benign on V2XSet
   (max target 11.76) but still a divergence from the reference.
3. **OPEN — no gradient clipping in the reference.** `grep clip_grad` over
   `tools/train.py` and `train_utils.py` returns nothing; we clip at 35.0.
4. **OPEN — `smooth_l1` beta.** Reference `WeightedSmoothL1Loss(beta=1/9)`
   (`point_pillar_loss.py:21`); we use `F.smooth_l1_loss` default `beta=1.0`.
5. **OPEN — Adam eps.** Reference `1e-10`; ours `1e-8`.
6. **OPEN — `max_boxes`.** Reference `max_num: 100`; our `BoxDecoder`
   uses 300.
7. **OPEN — NMS is configured but never applied.** `nms_iou: 0.15` is in
   `v2xvit.yaml:56`, `build_decoder` never passes it, and `BoxDecoder` does no
   suppression. A rotated-BEV NMS exists at `cpbench/utils/geometry.py:127`
   and is called from nowhere in non-test code. This *depresses* AP (duplicates
   become false positives), so it cannot inflate a result — but it is a
   protocol deviation and the first thing to test if trained AP lands low.

---

## 6. The open question

The bisect and the production run differ in exactly these ways:

| | bisect 556366 (stable) | run 555923 (diverged) |
|---|---|---|
| loader | direct `ds[j]` in main process | `DataLoader`, `num_workers=8`, `persistent_workers=True`, `worker_init_fn` |
| sampler | own `torch.randperm(generator=g)` | `shuffle=True` (global RNG) |
| harness | bare loop | full `Trainer` + logbook + per-epoch checkpointing |

So either the production harness triggers it — the 8-worker dataloader being
the prime suspect, since the seeded `worker_init_fn` was added recently — or
the bisect drew a luckier batch *ordering* (a different sampler gives different
sample pairings).

**Job 556564** runs 2 full epochs through the real `train.py` path to settle
this. Queued at priority 1, estimated start 2026-08-05T11:41.

---

## 7. Operational lessons (cost real time)

- **`sbatch` does not validate scripts.** A `sed`-assembled script with an
  unbalanced brace was accepted and queued; `bash -n` caught it after
  submission. Always `bash -n` *before* `sbatch`.
- **Never submit onto a node whose previous job was just cancelled.** Job
  556251 burned 3 h and produced nothing: I cancelled 555923 on `hpc-node09`,
  submitted immediately, Slurm reused the node, and the epilog `/local` cleanup
  deleted the new job's staging underneath its rsync
  (`mkstemp ... No such file or directory`). Scripts now check rsync's exit
  status and verify staged file counts against the source.
- **A vacuous green is worse than a red.** A label-scan predicate bug once
  matched **zero files** and reported `SCAN_GREEN=True`. Gates now require
  `SCANNED_SOMETHING` and `SAW_KNOWN_PICKLES`.
- **`metrics.csv` is written only at the end of `fit()`** (in-memory
  `_CsvSink`), so it is useless for live monitoring and **is lost entirely if a
  run is killed**. TensorBoard events are the only incremental record;
  `results/diag/loss_watch.py` converts them to a tailable text log.
- **`--constraint=l40` no longer pins the SKU.** `hpc-node12` and `hpc-node14`
  both advertise `l40` but report L40S and L40 respectively. It still keeps
  jobs off the Turing nodes (which killed job 549175 in 3 s), which is its
  essential job.
- **`nvidia-smi ... | head -1` reads the wrong GPU** on multi-GPU nodes. The
  `gpu=7 %, 551 MiB` figures in `progress.txt` are not the training GPU —
  profiling measured 19 GB allocated for this exact config.
- **Torch import from the NFS venv costs 15–20 min per job.** Do not stage
  `.venv-hpc` to `/local`; that was measured *slower* (>22 min for 5.4 GB) than
  importing in place. Stage data, not the interpreter.

---

## 8. What I would do next

1. Wait for **556564**. If it diverges, we have the failure in the production
   harness with per-step data, and the worker path is the prime suspect —
   testable by rerunning at `num_workers=0`.
2. If it stays clean, 555923's divergence is ordering-dependent, and the next
   move is to replay 555923's exact sampler seed and locate the batch.
3. Do **not** relaunch 60 epochs until one of those resolves. That is a
   ~52-hour bet on an unexplained failure.
4. Independently of the divergence, decide the **M3 eval-range** question
   (`docs/v2xvit_design.md` §7.1): grading on the oracle's y ±40 charges the
   model for **792 of 59,664** test GT boxes (1.327 %) that lie outside anchor
   coverage — bounded at ~0.0095 AP, which is roughly the difference between
   "reproduces the paper" and "one point short".
5. Retire assumption **A7** in `docs/v2xvit_design.md` and
   `configs/model/v2xvit.yaml` — it records a divergence from the reference
   that does not exist.
