# cobevtbench — CoBEVT under physical faults

An implementation of **CoBEVT** (Xu et al., *Cooperative Bird's Eye View
Semantic Segmentation with Sparse Transformers*, CoRL 2022,
[arXiv:2207.02202](https://arxiv.org/abs/2207.02202)) built as a
fault-injection benchmark on top of this repository's `cpbench` core and
`src` fault toolkit.

Both tracks from the paper are here:

- **camera** (the headline) — ResNet → SinBEVT → FuseBEVT → decoder → BEV
  semantic segmentation, scored by IoU. The first package in this repo whose
  primary fault surface is the *image*.
- **lidar** (Table 2) — `cpbench` PointPillars → FuseBEVT → detection head,
  scored by AP, directly comparable with `corabench` and `lgcpbench`.

The `> github.com/DerrickXuNu/CoBEVT` code is the reference; this is a
re-implementation restructured so every intermediate tensor is reachable.

## The two-plane contract

Inherited from the rest of the repo, and the reason a robustness number here
means what it says:

1. **Corruption** is physical and upstream. Faults touch raw images, camera
   calibration, poses, LiDAR and the comm link — on the `CooperativeSample`,
   *before any tensor exists*. No model, loss or metric code is fault-aware.
2. **Measurement** is passive and read-only. Every intermediate tensor is a
   named observation point; `emit(taps, x, ...)` hands taps a detached tensor
   and returns `None`. Forward output is bit-identical with taps on or off.

There is no third plane: CoBEVT is pure feed-forward (unlike LGCP, which has
a control plane).

## Why the reference was restructured

Upstream builds each attention block as `nn.Sequential(Rearrange, PreNorm,
...)`, which makes the attention scores, the softmax and the 3-D
relative-position bias unreachable. Those are the most interesting tensors in
this paper — *"when a collaborator's pose is wrong, does attention
down-weight it, or integrate the corruption?"* is a question about the
softmax. So every FAX block here writes its residuals out explicitly and taps
each step. The maths is identical; a released checkpoint loads into it.

## Architecture

```
attention/     the paper's contribution, isolated (no cameras, no agents):
               window/grid partition, 3-D rel-pos bias, QKV, attention, FAX
fusion/        FuseBEVT, the STTF warp, camera-ray embeddings, compression
models/        ResNet backbone, SinBEVT, decoder, seg head, the two models
data/          camera + lidar datasets, agent-axis collate, BEV rasterisation
faults/        CalibrationErrorInjector, CameraDropoutInjector, build_bridge
training/      losses, track-agnostic Trainer, clean-only validators
evaluation/    Tester, clean-first benchmark runner, sweeps, dynamic+static merge
observation/   the ~120-template registry of named tensors
scripts/       train / evaluate / benchmark CLIs
```

## Quick start (no data needed)

```bash
pip install -r requirements-bench.txt      # torch, torchvision, einops

# smoke-train the camera track on synthetic cooperative scenes
python -m cobevtbench.scripts.train trainer=smoke

# the camera-dropout reproduction (paper section 7.4) as a fault sweep
python -m cobevtbench.scripts.benchmark faults=camera_dropout --max-frames 8

# the LiDAR track
python -m cobevtbench.scripts.train model=cobevt_lidar dataset=synthetic_lidar trainer=smoke
```

## Real experiments (paper reproduction)

```bash
# Camera Table 1 — two SEPARATELY trained models (assumption A8)
python -m cobevtbench.scripts.train dataset=opv2v_camera model=cobevt_camera_dynamic
python -m cobevtbench.scripts.train dataset=opv2v_camera model=cobevt_camera_static

# camera-dropout robustness (paper reports 44.3 IoU, all ego cameras off)
python -m cobevtbench.scripts.benchmark --checkpoint best.pt \
    dataset=opv2v_camera model=cobevt_camera_dynamic faults=camera_dropout

# compression ablation (524 KB → 8 KB, 60.4 → 54.8 IoU) is a config override
python -m cobevtbench.scripts.evaluate --checkpoint best.pt compression=16

# section 7.3 local/global ablation is a model swap
python -m cobevtbench.scripts.train model=ablation_local_only

# LiDAR Table 2 (AP@0.7 85.2)
python -m cobevtbench.scripts.train dataset=opv2v_lidar model=cobevt_lidar trainer=lidar
```

## The fault that is specific to CoBEVT

SinBEVT lifts image features to BEV by matching **ray directions** computed
from the camera intrinsics `K` and extrinsics `T` — no depth network. Those
matrices are therefore on the attention path, and `src/` had no injector for
them. `CalibrationErrorInjector` fills that gap:

```bash
python -m cobevtbench.scripts.benchmark faults=calibration_error --checkpoint best.pt
```

## Configuration

Plain-YAML composition (no Hydra dependency). Swap a group with `group=name`,
set a leaf with `a.b.c=value`:

```bash
python -m cobevtbench.scripts.benchmark \
    model=cobevt_camera_dynamic dataset=opv2v_camera \
    faults=pose_error taps=stats compression=32 seed=7
```

Config groups: `model/` (camera dynamic/static, lidar, three ablations),
`dataset/` (synthetic + opv2v, per track), `faults/` (ten families),
`taps/` (none/stats/attention/info_quality), `trainer/`
(default/paper/lidar/smoke).

## Results bundle

`results/<experiment_name>_<train|bench>/`:

```
metrics.csv            per-epoch (train) or per-condition (bench) rows
meta.json              seeds, git commit, environment, assumptions A1–A11
fault_statistics.csv   flip / SDC / fault-success rate per condition
injection_summary.csv  every physically injected fault
taps.csv, taps/        per-tensor stats and dumps (when taps active)
checkpoints/           best.pt (mIoU / AP@0.7), last.pt
training.log           the run's console log
```

## Paper assumptions (A1–A11)

Every place the paper text and the released code disagree is a config flag,
recorded in `meta.json`. The defaults follow the **code** (it produced the
released weights); the alternative reading is always one override away.

| | ambiguity | default |
|---|---|---|
| A1 | 60 ep / Adam (paper) vs 151 / AdamW (config) | config; `trainer=paper` for the text |
| A2 | backbone taps: layer1/2/3 (paper) vs layer2/3/4 (code) | code (`id_pick=[1,2,3]`) |
| A3 | seg head Conv1×1 (paper) vs 3×3 (code) | 3×3 |
| A4 | decoder upsample bilinear (paper) vs nearest (code) | nearest |
| A5 | FAX cross-attention has a rel-pos bias (paper Eq. 4) vs not (code) | no bias |
| A6 | camera axis reduced by mean after projection | mean |
| A7 | reference allocates a dead second seg head | exactly one head built |
| A8 | dynamic + static are two models merged at inference | two configs + `merge.py` |
| A9 | attention heads derived as `dim // dim_head` | derived, asserted |
| A10 | no released LiDAR model | reconstructed from Appendix C.3 |
| A11 | agent pooling: unweighted mean over padding | `mean`; `pool=masked_mean` isolates info loss |

## Extending

- **A new fault** on the camera track is a sample-stage injector plus a
  `configs/faults/*.yaml`; `build_bridge` picks it up.
- **A new observation point** is one `_loc(...)` in `observation/locations.py`;
  the registry cross-check tests force the code and registry to agree.
- **FuseBEVT is modality-agnostic** — `(B, L, C, H, W)` + mask → `(B, C, H, W)`
  — which is also exactly the OpenCOOD intermediate-fusion contract.

## Testing

```bash
python -m pytest cobevtbench --doctest-modules -q     # CPU, no downloads, < 30 s
```

Note: `lgcpbench/configs/model/cobevt.yaml` also exists — that is LGCP *using*
CoBEVT as an orchestrated OpenCOOD backbone, a different thing from this
package, which studies CoBEVT as the subject. They share no code (the sibling
rule in `cpbench/tests/test_layering.py` forbids it).
