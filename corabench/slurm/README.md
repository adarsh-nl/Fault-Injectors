# Running corabench on the UT EEMCS HPC

Cluster docs: <https://hpc.wiki.utwente.nl/eemcs-hpc> · SLURM basics:
<https://hpc.wiki.utwente.nl/slurm:start>

## Ground rules

- Log in only to the **head nodes** (`hpc-head1.ewi.utwente.nl` or
  `hpc-head2.ewi.utwente.nl`) with your AD account (lowercase). Compile and
  submit there; **never** ssh into compute nodes.
- Partitions: use `ps,main-gpu` for GPU jobs (as in the templates),
  `ps,main-cpu` for CPU-only sweeps.
- Storage: home `/home/<user>` (code, results), node-local scratch `/local`
  (temporary only), and the dataset tree — see below.

## Where the data is: `CPBENCH_DATA_ROOT`

Every dataset path in this repository derives from one environment variable.
Set it once for the machine and no config, script or command line needs an
absolute path:

```bash
export CPBENCH_DATA_ROOT=/path/to/your/datasets     # on this cluster: /datasets/eemcs/ps/cv
```

The resolver (`cpbench/utils/paths.py`) expects this layout underneath:

| subdirectory | dataset |
|---|---|
| `opencood/opv2v` | OPV2V |
| `opencood/v2xset` | V2XSet |
| `air-thu/dair-v2x-c` | DAIR-V2X-C |
| `huggingface/griffin` | Griffin |

Precedence: an explicit `data_root=` on the command line, then
`$CPBENCH_DATA_ROOT`, then the `data_root` key in `configs/config.yaml`, then
the built-in default. A single dataset can still be pointed elsewhere with
`dataset.root=/somewhere/else` — that always wins for that one path. The
resolved value is recorded in each run's `config.yaml`.
- Jobs must run non-interactively — everything here is config-driven and
  needs no terminal interaction. Status page:
  <http://hpc-status.ewi.utwente.nl/slurm/>.

## One-time setup (on a head node)

```bash
cd ~ && git clone https://github.com/adarsh-nl/Fault-Injectors.git
cd Fault-Injectors
python3 -m venv .venv-hpc
.venv-hpc/bin/pip install --upgrade pip
.venv-hpc/bin/pip install torch torchvision pyyaml tensorboard matplotlib numpy
# optional, for the fused CSSM kernel (needs a CUDA toolchain):
# .venv-hpc/bin/pip install mamba-ssm && use model.cssm.backend=cuda
```

## Train (paper setting: 30 epochs, batch 2, 1 GPU)

```bash
sbatch corabench/slurm/train.sbatch dataset=opv2v
```

## Benchmark a trained checkpoint over all fault sweeps (job array)

```bash
sbatch corabench/slurm/benchmark_array.sbatch \
    CKPT=$HOME/cora-results/<experiment>/checkpoints/best.pt \
    dataset=opv2v
```

Each array task writes its own `results/<experiment>_bench/` bundle
(metrics.csv, fault_statistics.csv, injection_summary.csv, tensorboard/).
Merge across tasks by concatenating the CSVs — rows carry the condition.

## Monitoring

```bash
squeue -u $USER          # queue state
scontrol show job <id>   # details
tail -f slurm-cora-train-<id>.out
```
