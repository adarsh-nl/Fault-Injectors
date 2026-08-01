# Debug log

One entry per investigated failure. Format is fixed: **symptom → cause → fix →
verification**. An entry with no verification line is an open item, not a fix.

---

## 2026-07-31 — DL-001 — torch import stalls indefinitely on the login node

**Symptom.** `.venv-hpc/bin/python -c "import torch"` on `hpc-head1` did not
complete in 6 minutes. Two independent attempts, both hung. `python -c
"print('hello')"` returned instantly, so the interpreter itself was fine.

**Localisation.**

```
$ cat /proc/841026/status | grep State
State:  D (disk sleep)
$ cat /proc/841026/wchan
rpc_wait_bit_killable
$ ps -o pcpu 841026
0.5
```

`D` + `rpc_wait_bit_killable` is a blocked NFS RPC, not a CPU-bound import.
`ls -l /proc/<pid>/fd` showed it 3 minutes in still opening
`numpy/_core/__pycache__/_methods.cpython-313.pyc` — it *was* progressing,
just at NFS latency per file. `uptime` on the head node read
`load average: 10.69` with 4 interactive users.

**Cause.** `.venv-hpc` lives on the NFS-mounted `/home`. A torch import touches
several thousand small files, and on a contended login node each one costs an
NFS round trip. Nothing is broken; the filesystem is simply the wrong place to
import a multi-gigabyte package from.

**Fix.** Do not import torch on the login node at all. The keymatch diagnostic
moved to a CPU sbatch job (`results/diag/keymatch_diag.sbatch`,
`--partition=ps,main-cpu`, no GPU) which `rsync`s `.venv-hpc` into
`/local/$SLURM_JOB_ID` (node-local dual NVMe) and runs the interpreter from
there. A relocated venv needs `home =` in `pyvenv.cfg` repointed at
`/software/python/3.13.14/bin`; the interpreter is a symlink into `/software`,
which is shared and safe to follow.

**Verification — and a correction to the fix.** Two jobs were run as a
controlled comparison:

| job | approach | time to clear checkpoint staging |
|---|---|---|
| 554121 | `rsync .venv-hpc` → `/local`, then import | **> 22 min, never got past the rsync** (cancelled) |
| 554122 | import straight from NFS on the node | **< 1 min** to staging, then ~16 min in the import |

`.venv-hpc` is **5.4 GB**. Copying it costs more than the import it was meant
to accelerate. **Do not stage the venv.** Stage *data*; import the interpreter
in place. `results/diag/keymatch_diag_nocopy.sbatch` is the shape to copy.

**The import latency is not specific to the login node.** Job 554018
(`corabench`, 2026-07-31) shows 22:32 job start → 22:51:13 first Python log
line: **~19 minutes**. 554122 is consistent with that. So every job in this
repo pays roughly a **15–20 minute torch-import tax** before its first line of
work, on top of queue time. Budget `--time` accordingly, and never interpret a
silent first quarter-hour as a hang.

**Generalisation.** Any command that touches `.venv-hpc` broadly (`pytest`,
`pip`, `import torchvision`) hits the same wall. `~/CLAUDE.md`'s "login-node
commands must be <1 min" is not a courtesy rule — the filesystem enforces it.
`du -sh .venv-hpc` also exceeded 120 s. Even
`python -c "from cpbench.utils import load_config"` — no torch at all —
timed out at 60 s on `hpc-head1` during this pass.

**Corollary that saved this task.** A `.pth` written by `torch.save` is a ZIP
whose `data.pkl` member records every tensor's shape as arguments to
`torch._utils._rebuild_tensor_v2`. Stubbing that one symbol lets the standard
library read a checkpoint's complete key→shape layout **without importing
torch** — sub-second, on the login node, no job.
`results/diag/dump_official_keys.py` does exactly this, and it produced the
entire official side of the keymatch while 554122 was still importing.

---

## 2026-07-31 — DL-002 — `torch.load` default flipped to `weights_only=True`

**Symptom.** Not yet observed at runtime — found by reading, before it could
cost a job. Recorded here because it will fire on the first real checkpoint
load.

**Localisation.** `v2xvitbench/scripts/_cli.py:52` and
`w2cbench/scripts/_cli.py:52`:

```python
state = torch.load(checkpoint, map_location=device)
```

`cobevtbench/scripts/benchmark.py:83` and `corabench/scripts/common.py:235`
are the same shape.

**Cause.** PyTorch 2.6 flipped `torch.load`'s `weights_only` default from
`False` to `True`. `.venv-hpc` is torch 2.13.0, well past the flip. A pure
tensor `state_dict` still loads, so this repo's own checkpoints are fine —
but a released checkpoint saved as a training wrapper (`{'model': ...,
'optimizer': ...}` or anything carrying a numpy scalar) raises
`UnpicklingError` instead.

**Fix.** Not applied — this task is diagnosis only, and the call sites are
correct for this repo's own checkpoints. When the converter task starts, the
loaders need `weights_only=False` **and** an explicit comment saying the
checkpoint is trusted, because silently disabling the safety default is worse
than the error.

**Verification.** Open. The keymatch diagnostic
(`results/diag/keymatch_diag.py`) passes `weights_only=False` explicitly and
so is unaffected.

---

## 2026-07-31 — DL-003 — `--partition=ps,main-cpu` pends on node availability

**Symptom.** Job 554121 was accepted but immediately `PENDING` with

```
(Nodes required for job are DOWN, DRAINED or reserved for jobs in higher
 priority partitions)
```

**Cause.** Not an error. `sinfo -p ps,main-cpu` shows 6 nodes `mixed` and 3
`idle~` (powered down). The reason string is Slurm's generic message and reads
alarming; the job started on `hpc-node12` shortly after.

**Fix.** None needed. Recorded so the message is not mistaken for a
misconfigured partition next time.

**Verification.** `squeue` showed the job `RUNNING` on `hpc-node12` within a
minute of the pending message.

---

## 2026-08-01 — DL-004 — the keymatch harness cannot build `corabench`

**Symptom.** `results/diag/keymatch_raw.json` records `corabench`'s reimpl side
as `BLOCKED-BY-IMPORT`:

```
AttributeError: module 'corabench.scripts.common' has no attribute 'load'
```

**Cause.** A defect in the diagnostic, not in `corabench`. `keymatch_diag.py`'s
`build_reimpl()` assumes the `load(overrides)` → `build_model(cfg)` pair that
`cobevtbench`, `w2cbench` and `v2xvitbench` share. `corabench` exposes
`build_model(cfg, grid)` (`corabench/scripts/common.py:166`) and no `load`.
`corabench` imports and constructs fine; the label is misleading.

**Fix.** Not applied. CoRA has no released weights, so its verdict is
`NO-CHECKPOINT` and a reimpl key dump would be compared against nothing.
Re-running costs another ~20-minute job (DL-001) to produce a file no
question depends on. The misleading label is corrected in prose in
`results/diag/cora_keymatch.md`.

**Verification.** N/A — deliberately not fixed. Recorded so the
`BLOCKED-BY-IMPORT` string in the raw JSON is not later mistaken for evidence
that `corabench` is broken.

---

## 2026-08-01 — DL-005 — a Slurm job that has finished its work but will not exit

**Symptom.** Job 554122 printed its final line
(`WROTE .../keymatch_raw.json`) and then sat in `RUNNING` for a further ~24
minutes without producing output or exiting.

**Cause.** Not fully root-caused, and it did not need to be — the job's output
was already written and read. The suspect is the `trap 'rm -rf "$SCRATCH"'
EXIT` cleanup over `/local/554122`, which held ~400 MB of unpacked
checkpoints. Note the job had also just written into `~/.cache/torch` (F-08),
so an NFS flush on exit is an equally plausible culprit.

**Fix.** `scancel 554122` once its output file was confirmed complete on disk.

**Verification.** `results/diag/keymatch_raw.json` (25 KB) and all key dumps
were present and parsed correctly by `render_keymatch.py` after the cancel.

**Note.** Cancelling a job that has written its results is safe here *because
the results are written atomically at the end*, not streamed. Do not
generalise this to a training job, where cancelling loses everything since the
last checkpoint.

---

## 2026-08-01 — DL-006 — pillar decoration was 9 channels; OpenCOOD builds 10

**Symptom.** `pillar_vfe.pfn_layers.0.linear.weight` is `[64, 10]` in all three
released checkpoints; the reimplementation's is `[64, 9]`. Surfaced by the
keymatch diagnostic (audit F-12) as the block common to every operator.

**Cause.** `cpbench/data/preprocessing.py`'s `PillarVoxelizer` followed the
original PointPillars paper (Lang et al. 2019), whose decoration is 9 channels:
`[x, y, z, intensity, dx_mean, dy_mean, dz_mean, dx_center, dy_center]`.
OpenCOOD — which every released checkpoint and all four benchmarked papers use
— builds **10**. Its `f_center` is a full 3-vector, so there is a `dz_center`
as well. This was a reimplementation bug recorded as a deliberate convention
(`v2xvitbench` assumption **A6**), which is why it survived: the unit tests
asserted 9, so they confirmed the bug rather than catching it.

**Fix.** Added a 10th channel

```python
cz = 0.5 * (zmin + zmax)          # pillar geometric centre in z
features[i, :k, 9] = pv[:, 2] - cz
```

`cz` is the offset from the pillar's **geometric centre**, not the point mean —
the mean offset is already `dz_mean` and duplicating it would both waste a
channel and mismatch OpenCOOD. A pillar grid has exactly one voxel in z, so
OpenCOOD's `z_offset = voxel_z / 2 + zmin` collapses to the range midpoint,
which needs no new `GridSpec` field. On the released V2XSet/OPV2V geometry
(z −3..1) it is **−1.0**, consistent with the anchor `z_center` default.

Propagated to every hard-coded pillar-feature dim found by grep — 35 files
across all five packages, including two *functional* defaults that would
otherwise have crashed at the first matmul
(`w2cbench/models/encoder_lidar.py`, `lgcpbench/perception/native.py`, both
`in_channels: int = 9`) and four empty-pillar fallbacks
(`torch.zeros(0, max_points, 9)`) that would have failed only on empty scenes.
Assumption A6 retired in the config and both design docs, since it is
interpolated into `meta.json` on every run.

**Verification.** Job **554290**, `hpc-node14`, NVIDIA L40, `--constraint=l40`
— the same single-sample forward probe as job 554126, on the same real V2XSet
scene, so the two are directly comparable:

| | before (554126) | after (554290) |
|---|---|---|
| `features` | `[19480, 32, 9]` | **`[19480, 32, 10]`** |
| model params | 12 432 612 | **12 432 676** (**+64**) |
| model tensors | 281 | 281 |
| `cls` / `reg` / `fused` | `[1,2,48,176]` / `[1,14,48,176]` / `[1,256,48,176]` | unchanged |
| all finite | true | true |
| agent mask on | 62.25 % | 62.25 % |
| `cls` mean | −4.549 | −4.725 |

The **+64** is the load-bearing number: one extra input channel × 64 VFE output
channels, and *nothing else in the graph moved* (tensor count identical, every
output shape identical). That localises the change to the VFE exactly as
intended and rules out collateral edits.

`cls` mean drifting −4.549 → −4.725 is expected and not a regression: the head
bias is initialised at the focal prior (−4.595, `sigmoid` ≈ 0.01) and both runs
straddle it. The model is randomly initialised, and a 10th input channel
changes the random-init activation statistics feeding the head.

**Not verified here.** The unit test suite has **not** been run — it needs its
own compute job (DL-001) and was not part of this task's gate. The tests were
edited from 9 to 10; a regenerated test proves nothing on its own, which is
precisely how this bug survived. Run
`python -m pytest src/tests cpbench corabench/tests lgcpbench cobevtbench
w2cbench v2xvitbench --doctest-modules -q` in an sbatch job before relying on
them. Several edits were to doctests, so `--doctest-modules` matters.

**Consequence.** Any checkpoint trained before this change was fitted to
9-channel input and can no longer be loaded — `PillarVFE.linear` is now
`[64, 10]`. On disk that is
`results/cora_synthetic_clean_train/checkpoints/{best,last}.pt`, synthetic
smoke weights only. Not deleted; flagged.

---

## Open items carried out of this pass

- **DL-002** `weights_only` in the four checkpoint loaders. Downgraded in
  importance by F-12: with no released checkpoint loadable, the only things
  these loaders will ever open are this repo's own pure-tensor state dicts,
  which are unaffected. Fix it when a loader is next touched, not before.
- **DL-004** `corabench` reimpl key dump, deliberately not produced.
- ~~ResNet-34 not cached~~ — **closed**. Job 554122 downloaded
  `resnet34-b627a593.pth` from `hpc-node12`, which both warms the cache and
  proves compute nodes have outbound HTTPS (audit F-08).
- ~~DL-001 wall-clock for the staged import~~ — **closed**: staging is the
  wrong approach, see the corrected fix above.
