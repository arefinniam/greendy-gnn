#!/usr/bin/env python3
"""Default DGL baseline: on-demand feature fetching, no caching or prefetching."""

import argparse, time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl, dgl.distributed

from model import DistSAGE
from energy_monitor import AccurateEnergyMonitor, CPUEnergyMonitor
from metrics import TrainingProfiler, compute_accuracy


def main(args):
    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata["train_mask"], pb, force_even=True)

    device = th.device(f"cuda:{g.rank() % args.num_gpus}") if args.num_gpus else th.device("cpu")
    if args.num_gpus:
        th.cuda.set_device(device)
    dev_idx = device.index if args.num_gpus else None

    n_classes = args.n_classes
    if n_classes == 0:
        labels = g.ndata["labels"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels

    pid = g.rank()
    profiler = TrainingProfiler("default_dgl", pid, output_dir=args.out_dir)

    gpu_mon = AccurateEnergyMonitor(device_index=dev_idx, tick=0.05)
    cpu_mon = CPUEnergyMonitor(verbose=False)
    gpu_mon.start(); cpu_mon.start()

    sampler = dgl.dataloading.NeighborSampler(
        [int(f) for f in args.fan_out.split(",")])
    dataloader = dgl.distributed.DistNodeDataLoader(
        g, train_nid, sampler, batch_size=args.batch_size,
        shuffle=True, drop_last=False)

    in_feats = g.ndata["features"].shape[1]
    model = DistSAGE(in_feats, args.num_hidden, n_classes,
                     args.num_layers, F.relu, args.dropout).to(device)
    ddp = th.nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if args.num_gpus else None)
    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(ddp.parameters(), lr=args.lr)

    print(f"Part {pid}: Default DGL {args.graph_name} B={args.batch_size}")

    for epoch in range(args.num_epochs):
        tic = time.time()
        e_losses, e_accs = [], []

        with ddp.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                t0 = time.perf_counter()
                ft0 = time.perf_counter()
                batch_inputs = g.ndata["features"][input_nodes]
                batch_labels = g.ndata["labels"][seeds].long()
                fetch_time = time.perf_counter() - ft0

                blocks = [b.to(device) for b in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                pred = ddp(blocks, batch_inputs)
                loss = loss_fcn(pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                st = time.perf_counter() - t0

                acc = compute_accuracy(pred.detach(), batch_labels)
                gj = gpu_mon.get_total_gpu_energy()
                cj = cpu_mon.get_total_cpu_energy()
                profiler.record_step(epoch, step, loss.item(), acc, st,
                                     fetch_time, gj, cj)
                e_losses.append(loss.item())
                e_accs.append(acc)

                if (step + 1) % args.log_every == 0:
                    print(f"Part {pid} Ep{epoch:02d} S{step+1:3d}: "
                          f"L={loss.item():.4f} A={acc:.3f}")

        et = time.time() - tic
        gj = gpu_mon.get_total_gpu_energy()
        cj = cpu_mon.get_total_cpu_energy()
        profiler.record_epoch(epoch, et, gj, cj,
                              avg_loss=np.mean(e_losses),
                              avg_accuracy=np.mean(e_accs))
        print(f"Part {pid} Ep{epoch:02d}: {et:.2f}s GPU={gj:.1f}J "
              f"loss={np.mean(e_losses):.4f} acc={np.mean(e_accs):.3f}")

    gpu_mon.stop(); cpu_mon.stop()
    gj = gpu_mon.get_total_gpu_energy()
    cj = cpu_mon.get_total_cpu_energy()
    print(f"Part {pid}: Total GPU energy: {gj:.2f}J CPU: {cj:.2f}J")
    profiler.save()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a, t, d in [
        ("--graph_name", str, None), ("--ip_config", str, None),
        ("--part_config", str, None), ("--n_classes", int, 0),
        ("--backend", str, "gloo"), ("--num_gpus", int, 1),
        ("--num_epochs", int, 10), ("--num_hidden", int, 16),
        ("--num_layers", int, 2), ("--fan_out", str, "10,25"),
        ("--batch_size", int, 1000), ("--log_every", int, 20),
        ("--lr", float, 0.003), ("--dropout", float, 0.5),
        ("--local_rank", int, None), ("--out_dir", str, "logs"),
    ]:
        p.add_argument(a, type=t, default=d)
    main(p.parse_args())
