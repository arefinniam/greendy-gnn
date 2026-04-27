import torch as th
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

def select_hot_nodes(freq_tensor: th.Tensor, n_hot: int, **kwargs) -> th.Tensor:
    idxs = th.nonzero(freq_tensor > 0, as_tuple=True)[0]
    if idxs.numel() == 0:
        return th.empty(0, dtype=th.long, device=freq_tensor.device)
    
    scores = freq_tensor[idxs].float()
    k = min(n_hot, idxs.numel())
    return idxs[th.topk(scores, k).indices]



@dataclass
class BatchCacheMetrics:
    batch_id: int
    window_start: Optional[int]
    num_inputs: int
    cache_hits: int
    remote_cache_hits: int
    remote_misses: int
    local_misses: int
    miss_owner_counts: Dict[int, int] = field(default_factory=dict)
    step_time_s: float = 0.0
    gpu_energy_j: float = 0.0
    cpu_energy_j: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RebuildMetrics:
    window_start: int
    window_size: int
    unique_remote_nodes: int
    total_remote_accesses: int
    reused_nodes: int
    new_nodes: int
    owner_counts: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

class FeatureCache:
    def __init__(self, g, n_hot, device, dist_lock=None, track_detailed_metrics=False):
        self.part_id = None
        self.n_hot = n_hot
        self.dist_lock = dist_lock
        self.track_detailed_metrics = track_detailed_metrics
        
        # Cache feature metadata once to avoid repeated DistTensor lookups
        self.feat_dim = g.ndata["features"].shape[1]
        self.feat_dtype = g.ndata["features"].dtype
        
        # Double buffering: [Buffer 0, Buffer 1]
        self.cache_nodes = [
            th.tensor([], dtype=th.long, device=device),
            th.tensor([], dtype=th.long, device=device)
        ]
        self.cache_features = [
            th.empty((0, self.feat_dim), device=device),
            th.empty((0, self.feat_dim), device=device)
        ]
        self.cache_idx = [
            th.full((g.num_nodes(),), -1, dtype=th.int64, device=device),
            th.full((g.num_nodes(),), -1, dtype=th.int64, device=device)
        ]
        
        self.active_idx = 0  # 0 or 1
        
        self.pb = None
        try:
            self.pb = g.get_partition_book()
        except AttributeError:
            self.pb = None
        self.local_owner = None
        if self.pb is not None:
            self.local_owner = getattr(self.pb, "partid", None)
        if self.local_owner is None:
            self.local_owner = g.rank()
        self.current_window_start: Optional[int] = None
        self._batch_metrics: Dict[int, BatchCacheMetrics] = {}
        self._rebuild_metrics: List[RebuildMetrics] = []
        self._metrics_lock = threading.Lock()

        # Basic counters
        self.total_fetches = 0
        self.cache_hits = 0
        self.remote_cache_hits = 0
        self.remote_misses = 0
        self.local_misses = 0
        
        # Step-level counters (avoid per-batch hasattr checks)
        self._step_remote_hits = 0
        self._step_remote_misses = 0
        
        # Per-owner miss tracking (drives GreenDyGNN's cost-aware allocation).
        self._owner_miss_counts = {}  # {part_id: count} per window
        self._last_fetch_time_s = 0.0  # time spent in last DistTensor fetch
        self._total_fetch_time_s = 0.0
        self._fetch_count = 0
        
    def get_write_buffer_indices(self):
        """Return (nodes, features, idx_map) for the pending (write) buffer"""
        write_idx = 1 - self.active_idx
        return (self.cache_nodes[write_idx], 
                self.cache_features[write_idx], 
                self.cache_idx[write_idx])

    def set_write_buffer_state(self, nodes, features):
        """Update the state of the pending buffer"""
        write_idx = 1 - self.active_idx
        self.cache_nodes[write_idx] = nodes
        self.cache_features[write_idx] = features
        
        # Update index map for pending buffer
        self.cache_idx[write_idx].fill_(-1)
        if len(nodes) > 0:
            self.cache_idx[write_idx][nodes] = th.arange(len(nodes), device=nodes.device)

    def swap_buffers(self):
        """Make the pending buffer active"""
        self.active_idx = 1 - self.active_idx

    def get_features(self, input_nodes, g, device, remote_mask, batch_idx):
        if self.part_id is None:
            self.part_id = g.rank()
            
        # input_nodes and remote_mask are already on CPU from BackgroundSampler
        input_nodes_cpu = input_nodes
        remote_mask_cpu = remote_mask
        
        n_inputs = input_nodes_cpu.numel()
        
        # Allocate output tensor on target device
        out = th.empty((n_inputs, self.feat_dim), dtype=self.feat_dtype, device=device)
        
        self.total_fetches += n_inputs
        
        # Use ACTIVE buffer - capture index locally for consistency
        idx = self.active_idx
        active_nodes = self.cache_nodes[idx]
        active_features = self.cache_features[idx]
        active_idx_map = self.cache_idx[idx]
        
        has_cache = (active_nodes.numel() > 0)
        
        if has_cache:
            input_nodes_dev = input_nodes_cpu.to(device)
            pos = active_idx_map[input_nodes_dev]
            hit_mask_dev = (pos >= 0)
            
            if hit_mask_dev.any():
                out[hit_mask_dev] = active_features[pos[hit_mask_dev]]
                # Lightweight counters — defer int() to reduce GPU→CPU syncs
                n_hits = hit_mask_dev.sum()
                remote_hits_dev = hit_mask_dev & remote_mask_cpu.to(device)
                n_remote_hits = remote_hits_dev.sum()
                cache_hit_count = int(n_hits)
                remote_hit_count = int(n_remote_hits)
                self.cache_hits += cache_hit_count
                self.remote_cache_hits += remote_hit_count
                self._step_remote_hits += remote_hit_count
            else:
                cache_hit_count = 0
                remote_hit_count = 0
            
            miss_mask_cpu = (~hit_mask_dev).cpu()
        else:
            miss_mask_cpu = th.ones(n_inputs, dtype=th.bool)
            cache_hit_count = 0
            remote_hit_count = 0
        
        # Combined fetch: merge all misses into single DistTensor pull
        if miss_mask_cpu.any():
            remote_miss_cpu = miss_mask_cpu & remote_mask_cpu
            has_remote_miss = remote_miss_cpu.any()
            if has_remote_miss:
                remote_miss_count = int(remote_miss_cpu.sum())
                self.remote_misses += remote_miss_count
                self._step_remote_misses += remote_miss_count
                # Per-owner miss tracking for cost-aware caching
                if self.pb is not None:
                    try:
                        miss_nids = input_nodes_cpu[remote_miss_cpu]
                        owners = self.pb.nid2partid(miss_nids)
                        for pid in owners.unique().tolist():
                            c = int((owners == pid).sum())
                            self._owner_miss_counts[pid] = self._owner_miss_counts.get(pid, 0) + c
                    except Exception:
                        pass
            
            local_miss_cpu = miss_mask_cpu & (~remote_mask_cpu)
            if local_miss_cpu.any():
                self.local_misses += int(local_miss_cpu.sum())
            
            miss_nodes = input_nodes_cpu[miss_mask_cpu]
            import time as _time
            _t0 = _time.perf_counter()
            if self.dist_lock:
                with self.dist_lock:
                    miss_feats_cpu = g.ndata["features"][miss_nodes]
            else:
                miss_feats_cpu = g.ndata["features"][miss_nodes]
            _t1 = _time.perf_counter()
            self._last_fetch_time_s = _t1 - _t0
            self._total_fetch_time_s += self._last_fetch_time_s
            self._fetch_count += 1
            out[miss_mask_cpu.to(device)] = miss_feats_cpu.to(device, non_blocking=True)

        return out

    def _record_batch_metrics(
        self,
        batch_idx: int,
        num_inputs: int,
        cache_hits: int,
        remote_cache_hits: int,
        remote_misses: int,
        local_misses: int,
        miss_owner_counts: Dict[int, int],
    ) -> None:
        if batch_idx is None:
            return
        record = BatchCacheMetrics(
            batch_id=int(batch_idx),
            window_start=self.current_window_start,
            num_inputs=int(num_inputs),
            cache_hits=int(cache_hits),
            remote_cache_hits=int(remote_cache_hits),
            remote_misses=int(remote_misses),
            local_misses=int(local_misses),
            miss_owner_counts=miss_owner_counts.copy(),
        )
        with self._metrics_lock:
            self._batch_metrics[record.batch_id] = record

    def attach_runtime_metrics(
        self,
        batch_id: int,
        *,
        step_time_s: Optional[float] = None,
        gpu_energy_j: Optional[float] = None,
        cpu_energy_j: Optional[float] = None,
    ) -> None:
        with self._metrics_lock:
            record = self._batch_metrics.get(batch_id)
            if record is None:
                record = BatchCacheMetrics(
                    batch_id=int(batch_id),
                    window_start=self.current_window_start,
                    num_inputs=0,
                    cache_hits=0,
                    remote_cache_hits=0,
                    remote_misses=0,
                    local_misses=0,
                )
                self._batch_metrics[batch_id] = record
            if step_time_s is not None:
                record.step_time_s = float(step_time_s)
            if gpu_energy_j is not None:
                record.gpu_energy_j = float(max(0.0, gpu_energy_j))
            if cpu_energy_j is not None:
                record.cpu_energy_j = float(max(0.0, cpu_energy_j))

    def consume_batch_metrics(self) -> List[BatchCacheMetrics]:
        with self._metrics_lock:
            records = list(self._batch_metrics.values())
            self._batch_metrics.clear()
        return records

    def record_rebuild_event(
        self,
        *,
        window_start: int,
        window_size: int,
        unique_remote_nodes: int,
        total_remote_accesses: int,
        reused_nodes: int,
        new_nodes: int,
        owner_counts: Dict[int, int],
    ) -> None:
        self.current_window_start = window_start
        with self._metrics_lock:
            self._rebuild_metrics.append(
                RebuildMetrics(
                    window_start=window_start,
                    window_size=window_size,
                    unique_remote_nodes=unique_remote_nodes,
                    total_remote_accesses=total_remote_accesses,
                    reused_nodes=reused_nodes,
                    new_nodes=new_nodes,
                    owner_counts=owner_counts.copy(),
                )
            )

    def consume_rebuild_metrics(self) -> List[RebuildMetrics]:
        with self._metrics_lock:
            records = list(self._rebuild_metrics)
            self._rebuild_metrics.clear()
        return records
    
    def get_owner_miss_counts(self):
        """Per-owner miss counts since last call (resets internal state)."""
        counts = dict(self._owner_miss_counts)
        self._owner_miss_counts.clear()
        return counts

    def get_avg_fetch_time(self):
        """Return average DistTensor fetch time in seconds."""
        if self._fetch_count == 0:
            return 0.0
        return self._total_fetch_time_s / self._fetch_count

    def get_last_fetch_time(self):
        return self._last_fetch_time_s

    def get_stats(self):
        """Return basic statistics"""
        total_requests = max(1, self.total_fetches)
        total_remote_requests = max(1, self.remote_cache_hits + self.remote_misses)
        
        return {
            'total_fetches': self.total_fetches,
            'cache_hits': self.cache_hits,
            'remote_cache_hits': self.remote_cache_hits,
            'remote_misses': self.remote_misses,
            'local_misses': self.local_misses,
            'cache_hit_rate': self.cache_hits / total_requests,
            'remote_cache_hit_rate': self.remote_cache_hits / total_remote_requests,
            'remote_miss_rate': self.remote_misses / total_requests,
            'local_miss_rate': self.local_misses / total_requests
        }
    
    def get_step_remote_hit_rate(self):
        """
        Get remote cache hit rate for current step and reset step counters.
        Returns hit rate as percentage (0-100).
        """
        total = self._step_remote_hits + self._step_remote_misses
        hit_rate = (self._step_remote_hits / total * 100) if total > 0 else 0.0
        
        # Reset for next step
        self._step_remote_hits = 0
        self._step_remote_misses = 0
        
        return hit_rate
