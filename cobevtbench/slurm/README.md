# Running CoBEVT-Bench on the UT EEMCS HPC

Cluster docs: <https://hpc.wiki.utwente.nl/slurm:start> ·
partitions and specifics: <https://hpc.wiki.utwente.nl/eemcs-hpc:specifics#partitions>

## Ground rules

- Log in to a **head node** only: `hpc-head1.ewi.utwente.nl` or
  `hpc-head2.ewi.utwente.nl`. Compile and submit there; never `ssh` into a
  compute node.
- Use your AD account in lowercase (`nanjaiyalathaa`).
- Partitions for the PS resources: `ps,main-gpu` (GPU) and `ps,main-cpu`
  (CPU-only). CoBEVT training needs a GPU; the fault sweep can run either.
- Datasets are **read-only**, and every path derives from the
  `CPBENCH_DATA_ROOT` environment variable — set it once for the machine
  (on this cluster: `/datasets/eemcs/ps/cv`) instead of passing
  `dataset.root=` per job. Layout and precedence: `cpbench/utils/paths.py`.

  ```bash
  export CPBENCH_DATA_ROOT=/path/to/your/datasets
  ```
- Results go to `$HOME`; per-node scratch is `/local/$USER/$SLURM_JOB_ID` and
  is deleted at the end of each job.
- Jobs are non-interactive by preference. These scripts take no input.

## Setup

The scripts self-bootstrap a venv at `$REPO/.venv-hpc` on first run
(`torch torchvision einops pyyaml numpy matplotlib scipy tensorboard`). Two
things are worth doing once on a **head node** before the first GPU job:

```bash
# 1. clone and let the venv build (runs on the head node, has egress)
git clone <repo> ~/Fault-Injectors && cd ~/Fault-Injectors
python3 -m venv .venv-hpc && .venv-hpc/bin/pip install -r requirements-bench.txt

# 2. cache the pretrained ResNet34 -- compute nodes may have no internet
python3 -c "import torchvision; torchvision.models.resnet34(weights='DEFAULT')"
```

The second step matters: CoBEVT's backbone starts from ImageNet weights, and
`train_camera.sbatch` sets `TORCH_HOME=$HOME/.cache/torch` so the compute node
reads that cache instead of hanging on a socket timeout. If your compute
nodes do have egress, you can skip it.

## Jobs

```bash
# smoke test everything on synthetic data, CPU, no downloads
sbatch --partition=ps,main-cpu cobevtbench/slurm/benchmark_array.sbatch

# train the two camera models (SEPARATE jobs -- assumption A8)
sbatch cobevtbench/slurm/train_camera.sbatch dataset=opv2v_camera model=cobevt_camera_dynamic
sbatch cobevtbench/slurm/train_camera.sbatch dataset=opv2v_camera model=cobevt_camera_static

# train the LiDAR model
sbatch cobevtbench/slurm/train_lidar.sbatch dataset=opv2v_lidar

# fault sweep over a trained checkpoint (one array task per fault family)
CKPT=$HOME/cobevt-results/cobevt_camera_dynamic_opv2v_camera_clean_train/checkpoints/best.pt \
  sbatch cobevtbench/slurm/benchmark_array.sbatch dataset=opv2v_camera model=cobevt_camera_dynamic
```

Override any config key positionally, e.g. `compression=16`, `seed=7`,
`trainer=paper`. Environment overrides: `REPO`, `VENV`, `RESULTS`, `CKPT`,
`TORCH_HOME`.

## Reading the output

Each run writes `$RESULTS/<experiment_name>_<train|bench>/`:

| file | what |
|---|---|
| `metrics.csv` | one row per epoch (train) or condition (bench) |
| `meta.json` | seeds, git commit, environment, the A1–A11 assumptions |
| `fault_statistics.csv` | flip / SDC / fault-success rate per condition |
| `injection_summary.csv` | every physically injected fault |
| `taps.csv` | per-tensor stats, when `taps=stats` |
| `checkpoints/` | `best.pt` (by mIoU / AP@0.7) and `last.pt` |
| `training.log` | the run's console log |

The array's per-family SLURM logs are `slurm-cobevt-bench-<A>_<task>.out`.

## Known limitations

- The camera track is two separately trained models merged at inference. The
  benchmark array evaluates one `model=` at a time; merge them offline with
  `cobevtbench.evaluation.merge` for the paper's combined IoU.
- `taps=attention` dumps tensors and can fill disk fast; it sets `every_n`
  and `max_dumps`, but budget the space. `taps=stats` is the safe default for
  a full sweep.
- No real OPV2V adapter ships here — `dataset=opv2v_camera` needs a
  `src.datasets` adapter wired to the data under `$CPBENCH_DATA_ROOT`.
  Synthetic datasets run out of the box.
