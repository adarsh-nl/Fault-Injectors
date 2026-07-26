# Running w2cbench on the UT EEMCS cluster

Submit from a head node (`hpc-head1.ewi.utwente.nl` or `hpc-head2`), never from
a compute node. Partitions: `ps,main-gpu` for training and benchmarking,
`ps,main-cpu` for the synthetic smoke run. Results go to `$HOME` and
node-local scratch to `/local/$USER`.

Dataset paths all derive from one environment variable — set it once for the
machine and no command needs an absolute path (on this cluster the shared CV
tree is `/datasets/eemcs/ps/cv`):

```bash
export CPBENCH_DATA_ROOT=/path/to/your/datasets
```

See `cpbench/utils/paths.py` for the expected subdirectory layout and the
precedence rules; `dataset.root=` still overrides a single dataset.

| script | what it does | typical wall time |
|---|---|---|
| `train.sbatch` | one model, clean data, both tracks | 12–36 h |
| `benchmark_array.sbatch` | six fault families in parallel | 2–8 h each |
| `curve_array.sbatch` | the AP-vs-bandwidth curve under four conditions | 8–16 h each |

Every script accepts config overrides positionally, exactly as the CLI does, so
a command from the package README pastes in unchanged:

```bash
sbatch w2cbench/slurm/train.sbatch dataset=opv2v_lidar trainer=paper
```

Environment variables: `REPO` (default `$HOME/Fault-Injectors`), `VENV`,
`RESULTS`, and `CKPT` for the benchmark scripts.

## Before you read any results

**An untrained model shows no compression, and that is expected.** The
detection head initialises at the focal prior `sigmoid(−4.59) = 0.010051`,
which sits just *above* the released selection threshold of `0.01`. Selection
saturates, the whole map is transmitted, and the bandwidth column looks like
the model is not compressing at all. It is a training diagnostic, not a bug —
the benchmark CLI warns when run without `--checkpoint`.

**The `protocol` family needs `rounds=3`.** `RequestLossInjector` is provably a
no-op at the default `rounds=1`, because nothing consumes a request map in a
single round. `benchmark_array.sbatch` sets it automatically for that index;
if you run the family by hand, add `model.communication.rounds=3` or it will
report perfect robustness by construction.

**Camera results are not a reproduction.** There is no released Where2comm
camera model and no published camera table (assumption A14), so camera numbers
are internal comparisons — clean versus faulted under an identical model.

## Determinism

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is exported by every script because
`torch.use_deterministic_algorithms` requires it; without it a deterministic
run raises at the first matmul. The selector's bandwidth curriculum (A17) draws
through `torch.rand`, so `seed_everything` covers it and two runs of one config
train on the same schedule.

## Why the benchmark is an array rather than one long job

Conditions are independent by construction: each rebuilds the dataset around a
fresh fault bridge and never mutates the model weights. One failed family does
not lose the rest, and the families run in parallel. Merge the per-family
`metrics.csv` files afterwards — they share a column schema.

## First run on a fresh checkout

The scripts build `$VENV` from `requirements.txt` and `requirements-bench.txt`
on first use. Two notes:

* Match the CUDA module to the torch wheel. The scripts try `nvidia/cuda-12.4`
  then fall back to `11.8`; pin whichever your wheel was built against.
* The camera track fetches pretrained ResNet weights once. Pre-warm
  `$TORCH_HOME` from a head node — a compute node without egress fails loudly
  at model construction, which is better than hanging but still a wasted queue.

Smoke the whole pipeline on CPU before queuing anything expensive:

```bash
sbatch --partition=ps,main-cpu w2cbench/slurm/benchmark_array.sbatch
```

That runs the synthetic dataset with no checkpoint. Metrics are meaningless;
the sweep shape, the fault plumbing and the results bundle are all exercised.
