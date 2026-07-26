# Running v2xvitbench on the UT EEMCS HPC

Head nodes: `hpc-head1.ewi.utwente.nl` / `hpc-head2.ewi.utwente.nl` (never
run on compute nodes directly; submit through SLURM). Partitions:
`ps,main-gpu` for training and real benchmarks, `ps,main-cpu` for synthetic
smoke runs.

V2XSet is read-only, and its path derives from the `CPBENCH_DATA_ROOT`
environment variable — `configs/dataset/v2xset.yaml` says
`root: ${data_root}/opencood/v2xset` and never a literal. Set the variable
once for the machine (on this cluster the shared CV tree is
`/datasets/eemcs/ps/cv`):

```bash
export CPBENCH_DATA_ROOT=/path/to/your/datasets
```

Layout and precedence are documented in `cpbench/utils/paths.py`;
`dataset.root=` still overrides this one dataset.

Every script builds `$REPO/.venv-hpc` on first use from
`requirements.txt` + `requirements-bench.txt`, loads CUDA 12.4 (11.8
fallback) and exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic
mode. Override paths with environment variables: `REPO`, `VENV`, `RESULTS`,
`CPBENCH_DATA_ROOT`, and `CKPT` for the benchmark scripts.

```bash
# 1. clean training (the paper's schedule; ~60 epochs)
sbatch v2xvitbench/slurm/train.sbatch dataset=v2xset

# 2. the fault families, one array task each
CKPT=$HOME/v2xvit-results/v2xvit_v2xset_none_train/checkpoints/best.pt \
  sbatch v2xvitbench/slurm/benchmark_array.sbatch dataset=v2xset

# 3. the severity curves (pose / latency / correction-matrix)
CKPT=... sbatch v2xvitbench/slurm/curve_array.sbatch dataset=v2xset

# smoke everything first, on the CPU partition, no data or checkpoint needed
sbatch --partition=ps,main-cpu v2xvitbench/slurm/train.sbatch \
       model=v2xvit_tiny trainer=smoke
sbatch --partition=ps,main-cpu v2xvitbench/slurm/benchmark_array.sbatch \
       model=v2xvit_tiny max_frames=8
```

Results land in `$RESULTS/<experiment_name>/` as the standard bundle
(`meta.json`, `config.yaml`, `metrics.csv`, `metrics.json`,
`fault_statistics.csv`, `injection_summary.csv`, `taps.csv`, `training.log`,
`tensorboard/`, `checkpoints/`). Positional overrides pass straight through:
anything after the script name reaches the CLI unchanged.
