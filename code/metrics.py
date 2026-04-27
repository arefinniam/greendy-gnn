#!/usr/bin/env python3
"""Unified per-step/per-epoch profiler shared by all training methods.

Records loss, accuracy, step time, fetch time, GPU/CPU energy, and fetch-energy
breakdown into a JSON file that drives the figure pipeline. Used by
`train_default`, `train_bgl`, `train_rapidgnn`, and `train_greendygnn`.

Usage:
    profiler = TrainingProfiler("greendygnn", part_id=0, output_dir="logs/run1")
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            ...
            profiler.record_step(epoch, step, loss, acc, step_time, fetch_time,
                                 gpu_energy, cpu_energy, cache_hit_pct=...)
        profiler.record_epoch(epoch, epoch_time, gpu_energy, cpu_energy)
    profiler.save()
"""

import json, os, time
import torch as th


def compute_accuracy(pred, labels):
    """Compute classification accuracy from logits and labels.

    Args:
        pred: (N, C) logits or (N,) predicted class indices
        labels: (N,) ground truth labels (values < 0 are ignored)
    Returns:
        float accuracy in [0, 1]
    """
    with th.no_grad():
        if pred.dim() > 1:
            pred = pred.argmax(dim=1)
        labels = labels.long()
        valid = labels >= 0
        if valid.sum() == 0:
            return 0.0
        return float((pred[valid] == labels[valid]).float().mean())


class TrainingProfiler:
    """Per-step and per-epoch metrics recorder.

    Outputs a JSON file with two arrays:
      "steps": [{epoch, step, global_step, loss, accuracy, step_time_s,
                  fetch_time_s, gpu_energy_j, cpu_energy_j, fetch_energy_j,
                  wall_time_s, cache_hit_pct, ...}, ...]
      "epochs": [{epoch, epoch_time_s, gpu_energy_j, cpu_energy_j,
                   wall_time_s, avg_loss, avg_accuracy, ...}, ...]
    """

    def __init__(self, method_name, part_id, output_dir="logs"):
        self.method = method_name
        self.part_id = part_id
        self.output_dir = output_dir
        self.start_time = time.time()
        self.steps = []
        self.epoch_summaries = []
        self.cum_fetch_time = 0.0
        self.step_count = 0

    def record_step(self, epoch, step, loss, accuracy, step_time_s,
                    fetch_time_s, gpu_energy_cum_j, cpu_energy_cum_j,
                    cache_hit_pct=None, extra=None):
        """Record one training step's metrics."""
        self.cum_fetch_time += max(0, fetch_time_s)
        self.step_count += 1
        wall = time.time() - self.start_time

        # Fetch energy = fetch_time * average GPU power
        # Average GPU power = cumulative_gpu_energy / cumulative_wall_time
        avg_power = gpu_energy_cum_j / wall if wall > 0.1 else 35.0
        fetch_energy_cum = self.cum_fetch_time * avg_power

        rec = {
            "method": self.method,
            "part": self.part_id,
            "epoch": epoch,
            "step": step,
            "global_step": self.step_count,
            "loss": round(float(loss), 6),
            "accuracy": round(float(accuracy), 6),
            "step_time_s": round(step_time_s, 6),
            "fetch_time_s": round(fetch_time_s, 6),
            "gpu_energy_j": round(gpu_energy_cum_j, 2),
            "cpu_energy_j": round(cpu_energy_cum_j, 2),
            "fetch_energy_j": round(fetch_energy_cum, 2),
            "wall_time_s": round(wall, 3),
        }
        if cache_hit_pct is not None:
            rec["cache_hit_pct"] = round(float(cache_hit_pct), 1)
        if extra:
            rec.update(extra)
        self.steps.append(rec)

    def record_epoch(self, epoch, epoch_time_s, gpu_energy_cum_j,
                     cpu_energy_cum_j, avg_loss=None, avg_accuracy=None,
                     extra=None):
        """Record one epoch's summary metrics."""
        rec = {
            "method": self.method,
            "part": self.part_id,
            "epoch": epoch,
            "epoch_time_s": round(epoch_time_s, 3),
            "gpu_energy_j": round(gpu_energy_cum_j, 2),
            "cpu_energy_j": round(cpu_energy_cum_j, 2),
            "wall_time_s": round(time.time() - self.start_time, 3),
        }
        if avg_loss is not None:
            rec["avg_loss"] = round(float(avg_loss), 6)
        if avg_accuracy is not None:
            rec["avg_accuracy"] = round(float(avg_accuracy), 6)
        if extra:
            rec.update(extra)
        self.epoch_summaries.append(rec)

    def save(self, filename=None):
        """Save all metrics to JSON."""
        if not filename:
            filename = f"{self.method}_part{self.part_id}_profile.json"
        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            "method": self.method,
            "part_id": self.part_id,
            "total_steps": self.step_count,
            "total_wall_time_s": round(time.time() - self.start_time, 3),
            "total_fetch_time_s": round(self.cum_fetch_time, 3),
            "steps": self.steps,
            "epochs": self.epoch_summaries,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=1)
        print(f"PROFILE_SAVED|{self.method}|part{self.part_id}|{path}|"
              f"steps={self.step_count}|wall={data['total_wall_time_s']:.1f}s")
        return path
