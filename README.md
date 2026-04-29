# GreenDyGNN — SC26 Artifact

Companion code and data for *GreenDyGNN: Runtime-Adaptive Energy-Efficient
Communication for Distributed GNN Training* (SC26 submission).

## Layout

```
README.md
LICENSE
requirements.txt
code/
  train_default.py       Default DGL baseline (no caching)
  train_bgl.py           BGL baseline (prefetch during sampling)
  train_rapidgnn.py      RapidGNN baseline (epoch-level static cache)
  train_greendygnn.py    GreenDyGNN (intra-epoch adaptive cache + RL)
  greendygnn_agent.py    Dueling Double-DQN cache controller

  model.py               Shared 2-layer GraphSAGE
  cache.py               Double-buffered feature cache
  prefetcher.py          Async sampler + cache builder + feature resolver
  sampler.py             DGL distributed neighbor sampler wrapper
  presample.py           Cache warmup
  energy_monitor.py      NVML (GPU) + Intel RAPL (CPU) energy
  metrics.py             Per-step / per-epoch JSON profiler
  helpers.py             GPU clock control + run-summary printer

  launch.py              Vanilla DGL distributed launcher (upstream)
  congestion.py          tc netem driver (15-25 ms time-varying)
  parse_results.py       Log -> metrics.json
  gen_figures.py         Reads data/paper_data.json -> figures/*.pdf

  run_benchmark.sh       Single entry-point benchmark driver
  smoke_test.sh          Dependency/parser/figure-generation smoke test
  kill_all.sh            Emergency cluster cleanup
  ip_config.txt          Edit this with your 4 cluster node IPs
  ip_config.example      Commented example showing coordinator/node order
data/
  paper_data.json        All measurements used in the paper
  sim_params.json        Calibrated simulator parameters reported in the paper
figures/                 Generated PDFs (8 figures, Figs 4-11)
results/                 Reproduction outputs (created by run_benchmark.sh)
```

## Hardware

- 4-node cluster connected via 25 Gbps Ethernet
- Each node: Intel Xeon CPU (RAPL), one NVIDIA GPU (P100 or newer with NVML),
  Linux with `tc netem`, SSH key authentication between nodes, sudo for `tc`
- Reported runs: Chameleon Cloud bare-metal nodes with 2 P100 GPUs per node.

## Software

| Package    | Version | Source                                    |
| ---------- | ------- | ----------------------------------------- |
| Python     | 3.10+   | python.org                                |
| PyTorch    | 2.0+    | pytorch.org                               |
| DGL        | 1.1+    | dgl.ai (distributed mode)                 |
| pynvml     | 11.5+   | PyPI                                      |
| matplotlib | 3.7+    | PyPI                                      |
| numpy      | 1.24+   | PyPI                                      |
| METIS      | 5.1     | github.com/KarypisLab/METIS               |

## Reproducing the paper

### Quick path — inspect archived results (no cluster needed)

The `figures/` directory contains the generated PDF figures from the archived
measurements in `data/paper_data.json`. This path supports artifact inspection
without requiring access to the 4-node cluster.

```bash
python3 code/gen_figures.py
./code/smoke_test.sh
```

### Full path — re-run the benchmark on a 4-node cluster

1. **Provision** 4 GPU nodes (e.g., Chameleon Cloud bare metal) and configure
   SSH key authentication between them.

2. **Install** the software stack on every node:
   ```bash
   conda create -n greendygnn python=3.10 -y
   conda activate greendygnn
   pip install -r requirements.txt
   ```

3. **Partition** the three datasets with METIS into 4 parts and place them on
   shared storage (NFS), then export the root path:
   ```bash
   export DATASET_ROOT=/path/to/Dataset
   # Expects:
   #   $DATASET_ROOT/OGBN-Products/data/ogbn-products.json
   #   $DATASET_ROOT/Reddit/data/reddit.json
   #   $DATASET_ROOT/OGBN-Papers/data/ogbn-papers100M.json
   ```
   Use `dgl.distributed.partition_graph()` with the METIS algorithm.

4. **Edit** `code/ip_config.txt` with the 4 node IPs (one per line, no comments),
   replacing the documentation-only placeholder addresses. The first IP is the
   coordinator; the remaining three receive congestion injection. See
   `code/ip_config.example` for the annotated format.

5. **Run** the full benchmark:
   ```bash
   ./code/run_benchmark.sh --with-congestion         # 36 runs under congestion
   ./code/run_benchmark.sh                            # 36 clean baseline runs
   ```
   Per-run logs land in `code/logs/benchmark_<timestamp>/<method>/<dataset>/B<batch>/`.
   With `--with-congestion`, epochs 0-2 run clean as warmup, epochs 3 through
   the penultimate epoch repeat a 7-epoch pattern of 15-25 ms one-way delay on
   one or two non-coordinator nodes, and the final epoch is forced clean.
   The `tc netem` rule targets DGL RPC port 30050; PyTorch rendezvous port
   29500 is not delayed.

6. **Regenerate** the figures from the archived or updated data:
   ```bash
   python3 code/gen_figures.py
   ```

7. **Analyze** the generated `metrics.json` files and compare the aggregate
   energy, runtime, convergence, and cache metrics with the archived
   measurements in `data/paper_data.json` and the reference PDFs in `figures/`.

## Simulator Calibration

The offline simulator calibration protocol from the paper sweeps RPC delay
values `0,2,4,6,8` ms and cache windows `1..128`, then records a clean power
baseline. The fitted parameters reported in the paper are committed in
`data/sim_params.json` for reviewer inspection:
`alpha_rpc=4.67 ms`, `beta=1.40e-9 s/byte`, `gamma_c=2.01e-10 s/byte/ms`,
and `R^2=0.75`.

The submitted trainer does not require a separate binary RL checkpoint:
`code/greendygnn_agent.py` instantiates the lightweight Double-DQN controller,
calibrates live fetch/step-time baselines during the warmup epochs, and enables
adaptation from epoch 3 onward.

## Run-time

| Phase                         | Time                          |
| ----------------------------- | ----------------------------- |
| Setup (install + partition)   | ~60 min                       |
| Congestion benchmark (36 runs)| ~180 min on P100-class GPUs   |
| Clean benchmark (36 runs)     | ~150 min                      |
| Figure generation             | < 10 s                        |

## Selective runs

```bash
# Single dataset / batch / method:
./code/run_benchmark.sh --with-congestion \
    --datasets reddit --batch-sizes 2000 --methods greendygnn

# Ablation (handled inside the trainer):
python3 code/train_greendygnn.py --no_rl              # static W=16
python3 code/train_greendygnn.py --no_cost_weights    # uniform allocation
```

## Datasets

| Dataset           | Nodes  | Edges  | Features | Classes |
| ----------------- | ------ | ------ | -------- | ------- |
| Reddit            | 233 K  | 114 M  | 602      | 41      |
| OGBN-Products     | 2.4 M  | 61.9 M | 100      | 47      |
| OGBN-Papers100M   | 111 M  | 1.6 B  | 128      | 172     |

All three are from the Open Graph Benchmark (https://ogb.stanford.edu).
