# Running LGCP-Bench on the UT EEMCS HPC

Cluster docs: <https://hpc.wiki.utwente.nl/eemcs-hpc> · SLURM basics:
<https://hpc.wiki.utwente.nl/slurm:start> · status:
<http://hpc-status.ewi.utwente.nl/slurm/>

## Ground rules

- Submit only from the **head nodes** (`hpc-head1.ewi.utwente.nl` /
  `hpc-head2.ewi.utwente.nl`), AD account in lowercase. Never ssh into a
  compute node.
- Partitions: `ps,main-cpu` for the control-plane and native-backbone sweeps
  (they need no GPU), `ps,main-gpu` for the OpenCOOD paper models.
- Storage: code and results in `/home/<user>`, node-local scratch in `/local`
  (cleared after the job).
- Datasets: every path derives from the `CPBENCH_DATA_ROOT` environment
  variable — set it once for the machine rather than passing `dataset.root=`
  per job. On this cluster the shared CV tree is `/datasets/eemcs/ps/cv`;
  the expected subdirectories are listed in `cpbench/utils/paths.py`.

  ```bash
  export CPBENCH_DATA_ROOT=/path/to/your/datasets
  ```
- Everything is config-driven and non-interactive; no job needs a terminal.

## Two environments, and why

| Environment | Python | Used for |
|---|---|---|
| `.venv-hpc` | 3.9+ | core, control plane, native backbone, all CPU sweeps |
| `opencood-py37` (conda) | **3.7** | Where2comm / CoBEVT / CoAlign + weights |

OpenCOOD pins `numba==0.49.0`, which does not build on modern Python, so it is
locked to 3.7 regardless of the torch version, and needs `spconv` + CUDA.
`lgcpbench`'s core deliberately does not depend on any of it — only
`lgcpbench/perception/opencood/` does — so the two coexist rather than one
constraining the other.

`benchmark_array.sbatch` picks the right interpreter from the arguments: ask
for `model=where2comm|cobevt|coalign` and it activates the conda env (failing
early with instructions if it is missing); anything else uses `.venv-hpc`.

## Setup

```bash
# once, on a head node
cd ~ && git clone https://github.com/adarsh-nl/Fault-Injectors.git
cd Fault-Injectors
python3 -m venv .venv-hpc
.venv-hpc/bin/pip install --upgrade pip
.venv-hpc/bin/pip install torch pyyaml numpy matplotlib tensorboard

# only if you need the paper models (takes ~30 min)
sbatch lgcpbench/slurm/opencood_env.sbatch
```

`opencood_env.sbatch` writes `results/opencood_env.json` recording the
resolved OpenCOOD commit, spconv package and torch version, so a results
directory can be traced back to the exact revision that produced it.

## Jobs

```bash
# fault sweep, CPU, native backbone, synthetic data (whole-pipeline smoke test)
sbatch lgcpbench/slurm/benchmark_array.sbatch faults=comm_stress

# fault sweep on OPV2V with a paper model
sbatch --partition=ps,main-gpu --gres=gpu:1 \
  lgcpbench/slurm/benchmark_array.sbatch \
  model=where2comm dataset=opv2v faults=pose_error \
  model.hypes_yaml=$HOME/OpenCOOD/opencood/hypes_yaml/point_pillar_where2comm.yaml \
  model.checkpoint=$HOME/weights/where2comm.pth

# Fig. 7 scaling curve: 5-30 CAVs, control plane only, no GPU, minutes
sbatch lgcpbench/slurm/simulate_array.sbatch
```

The array is over `Delta_g` (0.05 / 0.075 / 0.1 / 0.125 — the Table II
values). Conditions are independent by construction: each rebuilds the dataset
around a fresh fault bridge and never touches the model, so the grid
parallelises perfectly and one failed condition does not lose the rest.

## Reading the output

Per condition, under `$RESULTS/<experiment_name>/`:

| File | Contents |
|---|---|
| `config.yaml`, `meta.json` | resolved config, environment, git commit, assumptions B1–B12 |
| `metrics.csv` / `.json` | one row per condition, all metric families |
| `fault_statistics.csv` | injected fault count per condition |
| `injection_summary.csv` | per-record audit trail |
| `control_plane.csv` | RSU decisions (with `taps=stats`) |
| `training.log` | full run log |

**Check the log for two warnings.** The evaluator flags `orphan_rate ≈ 1.0`
and `ap50 == 0.0` explicitly, because both are what an untrained or
unloaded backbone produces — technically valid rows that would otherwise look
like findings. If you see them with `model=where2comm`, the checkpoint did not
load.

## Known limitations

- **Detection AP needs trained weights.** The native backbone is untrained;
  its AP is meaningless (assumption B11). System metrics — communication
  volume, latency decomposition, schedule health, area coverage — are
  meaningful either way.
- **The OpenCOOD path has never been executed against real weights.** It is
  written against upstream sources and tested against structural stubs. Treat
  the first cluster run as integration testing.
- **Absolute latency depends on assumption B4.** The paper's per-model MFLOPs
  are whole-map inference costs; LGCP fuses ~0.3% of a map per area. Charging
  the full cost makes a 30-CAV run fusion-bound at ~165 ms; scaling by area
  share (`FusionLatencyModel.area_scaled`) reconciles with the paper's
  reported sub-deadline latencies. Both readings are available; the choice is
  recorded, not hidden.
