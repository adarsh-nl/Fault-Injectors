# Environment matrix

Measured 2026-07-31 on `hpc-head1`. The question this answers: **is the
adopted "four fully isolated environments" decision required by the
dependency facts, or was it over-caution?**

Answer: **it is required, and for a sharper reason than dependency conflicts
between the four benches.** See §4.

---

## 1. Environments that exist on this account

| Env | Path | Python | torch | numpy | scipy | spconv / cumm | numba | open3d | mmcv | OpenCOOD |
|---|---|---|---|---|---|---|---|---|---|---|
| `.venv-hpc` (**the repo's env**) | `Fault-Injectors/.venv-hpc` | **3.13.14** | **2.13.0+cu130** | **2.5.1** | 1.18.0 | — | — | — | — | — |
| `thesis` (conda) | `~/.conda/envs/thesis` | 3.10.20 | 2.5.1 | 2.2.6 | 1.15.2 | — | — | — | — | — |
| `dhd_env` (conda) | `~/.conda/envs/dhd_env` | 3.8.20 | 2.0.1+cu118 | **1.24.4** | 1.10.1 | — | — | — | — | — |
| `coop-perception` (venv) | `~/venvs/coop-perception` | 3.13.5 | — | — | — | — | — | — | — | — |
| `base` (conda, login default) | `/software/anaconda3/2025.06` | — | — | — | — | — | — | — | — | — |

`coop-perception` is a **downloader** venv (`gdown`, `huggingface_hub`,
`requests`, `bs4`) — it has no ML stack. It is not a candidate host for
anything.

**OpenCOOD is not installed anywhere on this account**, and no `inference.py`
exists in the repo or under `$HOME`. Confirms the brief.

## 2. What the four benches need

All four paper packages import the **same** stack and nothing else:

| Requirement | Source |
|---|---|
| Python ≥ 3.9 | `cpbench/utils/config.py` is deliberately 3.9-friendly |
| `torch`, `numpy`, `scipy`, `PyYAML` | everywhere |
| `einops` | `cobevtbench`, `v2xvitbench` attention blocks |
| `torchvision` | camera tracks only, imported **lazily** |
| `tensorboard` | optional — logbook degrades to CSV+JSON |
| MultiCorrupt backends | optional — missing files become stubs in `src.fault_injectors.MISSING_OPTIONAL` |

`requirements-hpc.lock.txt` (56 packages, verified exact against `.venv-hpc`)
is the authoritative record.

**There is no dependency conflict between `corabench`, `cobevtbench`,
`w2cbench`, `v2xvitbench` and `cpbench`.** They are five packages in one
source tree sharing one import graph, enforced by
`cpbench/tests/test_layering.py`. One env hosts all four today and does so
correctly.

## 3. What the *released checkpoints* need

This is the axis that actually forces isolation.

| Requirement | Why | Satisfiable in `.venv-hpc`? |
|---|---|---|
| `spconv` (+ `cumm`) | every released config uses `SpVoxelPreprocessor` for voxelisation | **no** — no spconv build targets Python 3.13 |
| `numba` | OpenCOOD's NMS / box utils | **no** — the historical pin is `numba 0.49`, which caps at Python 3.8 |
| `numpy < 2.0` | the released `config.yaml`s embed a pickle addressing `numpy.core.multiarray._reconstruct`; `numpy.core` was renamed `numpy._core` in numpy 2.0 | **no** — `.venv-hpc` is numpy 2.5.1 |
| unsafe YAML loader | same embedded pickle: `yaml.safe_load` refuses `!!python/object/apply:` | yes (loader choice, not a dependency) |
| `open3d` | OpenCOOD visualisation | not needed for eval |

`dhd_env` (Python 3.8.20, torch 2.0.1+cu118, **numpy 1.24.4**) is the only
environment on the account whose Python/numpy pair could host OpenCOOD:
`spconv-cu118` publishes cp38 wheels, and numpy 1.24 keeps the `numpy.core`
path the pickled configs need. torch 2.0.1+cu118 also covers every GPU
generation in this cluster including the Turing nodes that killed job 549175.

**`dhd_env` must not be modified** (brief). A *clone* of it is the honest
starting point if the OpenCOOD path is ever taken.

### 3a. An OpenCOOD env builder already exists in the repo, unrun

`lgcpbench/slurm/opencood_env.sbatch` builds `~/.conda/envs/opencood-py37`:
`python=3.7`, `torch==1.12.0+cu113`, `spconv-cu113`, then clones
`DerrickXuNu/OpenCOOD`, `python setup.py develop`, builds the two C
extensions, and verifies `SpVoxelPreprocessor` imports. It writes
`results/opencood_env.json` recording the resolved OpenCOOD SHA.

**It has never been run.** None of its three products exist:
`~/.conda/envs/opencood-py37`, `~/OpenCOOD`, `results/opencood_env.json`.

Preconditions checked from `hpc-head1`:

| Precondition | Status |
|---|---|
| Outbound HTTPS (PyPI, GitHub) | **OK** — `pypi.org` and `github.com` both return 200 from the login node. Not verified from a compute node. |
| `nvidia/cuda-11.8` module | **present** in `/software/Modulefiles` |
| `miniconda3` module | **absent** from the module list. The script does `module purge` and then `module load miniconda3 \|\| true`, relying on `command -v conda` still resolving afterwards. Slurm's default `--export=ALL` should carry the login shell's conda PATH through, but `module purge` in between makes this the script's most likely failure point. |
| Python 3.7 from conda defaults | not verified |

So the "legacy env" branch is **scripted but unproven**, and its first failure
is predictable and cheap to fix. That is a materially better position than the
brief's prior assumed — but it is still the slow path, and it produces
*OpenCOOD's* AP through *OpenCOOD's* pipeline, not a number that flows through
this repo's fault planes, taps or `results/cells` bundle.

## 4. Verdict on the isolation decision

Isolation is required, but the boundary is **not** "one env per bench". The
measured boundary is:

```
  [ .venv-hpc : py3.13 / torch 2.13 / numpy 2.5 ]      <- all four reimplementations
                        |
                        |  NO shared env is possible across this line
                        v
  [ opencood-env : py3.8 / numpy<2 / spconv / numba ]  <- released checkpoints,
                                                          their preprocessors,
                                                          their embedded configs
```

The conflict is **reimplementation stack vs released-checkpoint stack**, and
it is hard: Python 3.13 vs a `numba` pin that caps at 3.8 cannot be
reconciled by version juggling. Nothing to solve; the split is the design.

This is a **refinement of the adopted decision, not a contradiction of it.**
Keeping four separate envs costs little and buys per-bench pinning headroom,
so nothing about the adopted policy needs to change. But the *reason* to
record is the one above — if the four-env rule is ever justified by
"the benches conflict with each other", that justification is false and will
mislead a later decision.

The `results/cells/` integration layer is unaffected either way: a cell is a
JSON file, so any env can write one.

## 5. Consequences that follow directly

1. **Checkpoint conversion cannot be avoided by "just run OpenCOOD".** Running
   OpenCOOD means building the py3.8 stack, and even then the output would be
   OpenCOOD's AP, not this repo's — so the fault planes, taps and
   `results/cells` bundle would all be bypassed. That is a different project.
2. **A converter runs in `.venv-hpc`** — it only needs `torch.load` and a key
   remap, neither of which needs spconv. Whether that converter is worth
   writing is what `results/diag/*_keymatch.md` decides.
3. **Voxelisation must be this repo's own**, not `SpVoxelPreprocessor`. It
   already is (`cpbench` `PointPillarEncoder` + the packages' lidar datasets),
   so a converted checkpoint would be fed by a *different* voxeliser than it
   was trained with. Pillar-level differences change the encoder's input
   distribution, so **a converted encoder is not guaranteed to be numerically
   valid even with a perfect key match.** (The largest such difference,
   9-vs-10 feature decoration, was fixed on 2026-08-01 — assumption A6 is
   retired. `max_points_per_pillar` and pillar-selection order remain
   unverified against OpenCOOD.)
4. **The camera tracks need internet on the compute node.** `resnet34
   pretrained: true` downloads to `~/.cache/torch/hub/checkpoints/`, which
   currently holds only `efficientnet-b4`. Cache it from the login node before
   any camera job, or the job dies on a network call.

## 6. GPU pinning (unchanged, restated because it interacts with the above)

`--partition=ps,main-gpu` spans three GPU generations. `.venv-hpc`'s torch
2.13.0+**cu130** does not target sm_75, so a Turing node aborts the CUDA
context with an empty stderr (job 549175, 3 s). Every GPU job must carry
`--constraint=l40`. A hypothetical py3.8/torch 2.0.1+cu118 OpenCOOD env would
*not* have this restriction — cu118 covers sm_75 — which is a reason to keep
the constraint attached to the env, not to the repo.
