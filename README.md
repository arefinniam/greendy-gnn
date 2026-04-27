# GreenDyGNN — SC26 Artifact

Companion code and data for *GreenDyGNN: Runtime-Adaptive Energy-Efficient
Communication for Distributed GNN Training* (SC26 submission).

## Layout

```
artifacts/
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
    gen_figures.py         Reads ../data/paper_data.json -> ../figures/*.pdf

    run_benchmark.sh       Single entry-point benchmark driver
    kill_all.sh            Emergency cluster cleanup
    ip_config.txt          Edit this with your 4 cluster node IPs
  data/
    paper_data.json        All measurements used in the paper
  figures/                 Generated PDFs (8 figures, Figs 4-11)
  results/                 Reproduction outputs (created by run_benchmark.sh)
```

## Hardware

- 4-node cluster connected via 25 Gbps Ethernet
- Each node: Intel Xeon CPU (RAPL), one NVIDIA GPU (P100 or newer with NVML),
  Linux with `tc netem`, passwordless SSH between nodes, sudo for `tc`
- Reported runs: Chameleon Cloud bare-metal nodes

## Software

| Package    | Version | Source                                    |
| ---------- | ------- | ----------------------------------------- |
| Python     | 3.10+   | python.org                                |
| PyTorch    | 2.0+    | pytorch.org                               |
| DGL        | 1.1+    | dgl.ai (distributed mode)                 |
| pynvml     | 11.5+   | PyPI                                      |
| matplotlib | 3.7+    | PyPI                                      |
| METIS      | 5.1     | github.com/KarypisLab/METIS               |

## Reproducing the paper

### Quick path — figures only (no cluster needed)

```bash
cd code
python3 gen_figures.py
# writes 8 PDFs to ../figures/
```

This regenerates every paper figure (Fig. 4-11) from the archived
measurements in `data/paper_data.json`.

### Full path — re-run the benchmark on a 4-node cluster

1. **Provision** 4 GPU nodes (e.g., Chameleon Cloud bare metal) and configure
   passwordless SSH between them.

2. **Install** the software stack on every node:
   ```bash
   conda create -n greendygnn python=3.10 -y
   conda activate greendygnn
   pip install torch dgl pynvml matplotlib numpy
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

4. **Edit** `code/ip_config.txt` with the 4 node IPs (one per line), replacing
   the documentation-only placeholder addresses. The first IP is the
   coordinator; the remaining three receive congestion injection.

5. **Run** the full benchmark:
   ```bash
   cd code
   ./run_benchmark.sh --with-congestion              # 36 runs under congestion
   ./run_benchmark.sh                                 # 36 clean baseline runs
   ```
   Per-run logs land in `code/logs/benchmark_<timestamp>/<method>/<dataset>/B<batch>/`.

6. **Re-generate** the figures from the new data (after updating
   `data/paper_data.json` with the parsed results):
   ```bash
   python3 gen_figures.py
   ```

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
./run_benchmark.sh --with-congestion \
    --datasets reddit --batch-sizes 2000 --methods greendygnn

# Ablation (handled inside the trainer):
python3 train_greendygnn.py --no_rl              # static W=16
python3 train_greendygnn.py --no_cost_weights    # uniform allocation
```

## Datasets

| Dataset           | Nodes  | Edges  | Features | Classes |
| ----------------- | ------ | ------ | -------- | ------- |
| Reddit            | 233 K  | 114 M  | 602      | 41      |
| OGBN-Products     | 2.4 M  | 61.9 M | 100      | 47      |
| OGBN-Papers100M   | 111 M  | 1.6 B  | 128      | 172     |

All three are from the Open Graph Benchmark (https://ogb.stanford.edu).

## Archival note

For the SC26 artifact-freeze stage, create a DOI-backed release of this
repository through Zenodo, FigShare, Dryad, or another DOI-providing archive.
