#!/usr/bin/env python3
"""GreenDyGNN trainer: intra-epoch window-based cache rebuilds driven by an RL agent.

Ablation flags:
  --no_rl            disable RL controller, use static W=16
  --no_cost_weights  disable per-owner cost-aware cache selection
"""

import argparse, os, threading, time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl, dgl.distributed

from prefetcher import BatchPrefetcher, SharedBuffer, BackgroundSampler
from cache import FeatureCache
from energy_monitor import AccurateEnergyMonitor, CPUEnergyMonitor
from model import DistSAGE
from helpers import set_gpu_frequency, print_summary
from sampler import DistSampler
from presample import presample_and_cache
from greendygnn_agent import GreenDyGNNAgent
from metrics import TrainingProfiler, compute_accuracy


def main(args):
    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    train_nid = dgl.distributed.node_split(
        g.ndata["train_mask"], g.get_partition_book(), force_even=True)

    device = th.device(f"cuda:{g.rank() % args.num_gpus}") if args.num_gpus else th.device("cpu")
    if args.num_gpus:
        th.cuda.set_device(device)
    dev_idx = device.index if args.num_gpus else None

    if args.n_classes == 0:
        sl = g.ndata["labels"][train_nid[:min(10000, train_nid.numel())]]
        v = th.logical_and(~th.isnan(sl), sl >= 0)
        lm = th.max(sl[v]).long()
        th.distributed.all_reduce(lm, op=th.distributed.ReduceOp.MAX)
        args.n_classes = int(lm.item()) + 1

    pid = g.rank()
    nparts = th.distributed.get_world_size()
    num_owners = nparts - 1

    label = "greendygnn"
    if args.no_rl:
        label += "_no_rl"
    if args.no_cost_weights:
        label += "_no_cw"

    agent = GreenDyGNNAgent(num_owners=num_owners, initial_w=args.window_size)
    profiler = TrainingProfiler(label, pid, output_dir=args.out_dir)

    gpu_mon = AccurateEnergyMonitor(device_index=dev_idx, tick=0.05)
    cpu_mon = CPUEnergyMonitor(verbose=False)
    gpu_mon.start(); cpu_mon.start()

    os.makedirs(args.out_dir, exist_ok=True)
    set_gpu_frequency("min", dev_idx)
    dist_lock = threading.Lock()

    lp = g.local_partition
    if lp:
        inner = lp.ndata["inner_node"].bool()
        ids = lp.ndata["_ID"]
        if not isinstance(ids, th.Tensor):
            ids = th.tensor(ids, dtype=th.long)
        lid = ids[inner]
    else:
        lid = th.empty(0, dtype=th.long)
    lmask = th.zeros(g.num_nodes(), dtype=th.bool)
    if lid.numel() > 0:
        lmask[lid] = True

    sbuf = SharedBuffer(capacity=500)
    sampler = DistSampler(g, train_nid, args.fan_out, args.batch_size)
    bg = BackgroundSampler(sampler, sbuf, g, lmask, start_batch_id=0,
                           dist_lock=dist_lock, num_epochs=args.num_epochs)
    bg.start()
    bpe = len(sampler)
    tot = bpe * args.num_epochs

    ew = args.window_size if args.window_size > 0 else bpe
    sim_cache, cache, ps_time, _ = presample_and_cache(
        args, g, sbuf, device, dist_lock, max_batches=max(1, min(2 * ew, bpe)))
    set_gpu_frequency("default", dev_idx)

    W = args.window_size
    agent.current_w = W
    print(f"Part {pid}: {label} {args.graph_name} W={W} cache={args.cache_size} bpe={bpe}")

    model = DistSAGE(g.ndata["features"].shape[1], args.num_hidden,
                     args.n_classes, args.num_layers, F.relu, args.dropout).to(device)
    ddp = th.nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if args.num_gpus else None)
    opt = optim.Adam(ddp.parameters(), lr=args.lr)
    lfn = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    pf = BatchPrefetcher(g, device, cache, W, sbuf, tot, bpe,
                         max_batches=args.prefetch_buffer_size,
                         synchronous_cache=getattr(args, 'sync_cache', False),
                         n_classes=args.n_classes, window_hot_nodes={},
                         dist_lock=dist_lock, initial_data=sim_cache)

    for epoch in range(args.num_epochs):
        te = time.time()
        e_losses, e_accs = [], []

        if epoch == 2:
            agent.calibrate_baseline()
        if epoch >= 3 and not args.no_rl:
            agent.epoch_progress = epoch / args.num_epochs
            owner_misses = cache.get_owner_miss_counts()
            if owner_misses:
                agent.update_owner_misses(owner_misses, pid)
                agent.update_owner_congestion_from_misses()
            new_W, _ = agent.select_action()
            if new_W != W:
                W = new_W
                pf.window_size = W
            if not args.no_cost_weights:
                pf.cost_weights = agent.compute_cost_weights(pid, nparts)
            else:
                pf.cost_weights = None

        pf.start_epoch(epoch)
        with ddp.join():
            for step in range(bpe):
                t0 = time.perf_counter()
                inp, lab, blk = pf.get()
                ft = cache.get_last_fetch_time()
                if ft > 0:
                    agent.record_fetch_time(ft)
                out = ddp(blk, inp)
                loss = lfn(out, lab)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                st = time.perf_counter() - t0
                agent.record_step_time(st)

                acc = compute_accuracy(out.detach(), lab)
                hr = cache.get_step_remote_hit_rate()
                agent.record_hit_rate(hr)
                gj = gpu_mon.get_total_gpu_energy()
                cj = cpu_mon.get_total_cpu_energy()
                profiler.record_step(epoch, step, loss.item(), acc, st,
                                     max(0, ft), gj, cj,
                                     cache_hit_pct=hr, extra={"W": W})
                e_losses.append(loss.item())
                e_accs.append(acc)

                if (step + 1) % args.log_every == 0:
                    print(f"Part {pid} Ep{epoch:02d} S{step+1:3d}: "
                          f"L={loss.item():.4f} A={acc:.3f} W={W}")

        et = time.time() - te
        gj = gpu_mon.get_total_gpu_energy()
        cj = cpu_mon.get_total_cpu_energy()
        profiler.record_epoch(epoch, et, gj, cj,
                              avg_loss=np.mean(e_losses),
                              avg_accuracy=np.mean(e_accs))
        print(f"Part {pid} Ep{epoch:02d}: {et:.2f}s GPU={gj:.1f}J "
              f"loss={np.mean(e_losses):.4f} acc={np.mean(e_accs):.3f}")

    gj = gpu_mon.get_total_gpu_energy()
    cj = cpu_mon.get_total_cpu_energy()
    print_summary(cache, pid, args, ps_time, gj, cj)
    profiler.save()
    gpu_mon.stop(); cpu_mon.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a, t, d in [
        ("--graph_name", str, None), ("--ip_config", str, None),
        ("--part_config", str, None), ("--n_classes", int, 0),
        ("--backend", str, "gloo"), ("--num_gpus", int, 1),
        ("--num_epochs", int, 30), ("--num_hidden", int, 16),
        ("--num_layers", int, 2), ("--fan_out", str, "10,25"),
        ("--batch_size", int, 1000), ("--log_every", int, 20),
        ("--lr", float, 0.003), ("--dropout", float, 0.5),
        ("--local_rank", int, None), ("--cache_size", int, 100000),
        ("--window_size", int, 16), ("--prefetch_buffer_size", int, 100),
        ("--presample_batches", int, 2000), ("--out_dir", str, "logs"),
    ]:
        p.add_argument(a, type=t, default=d)
    p.add_argument("--sync_cache", action="store_true")
    p.add_argument("--no_rl", action="store_true", help="Ablation: disable RL")
    p.add_argument("--no_cost_weights", action="store_true",
                   help="Ablation: disable cost-aware cache selection")
    main(p.parse_args())
