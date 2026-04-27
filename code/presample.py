"""Pre-sample batches into memory for cache warm-up."""

import time
import torch as th
from cache import FeatureCache


def presample_and_cache(args, g, shared_buffer, device, dist_lock=None,
                        max_batches=2000):
    """Consume initial batches from SharedBuffer for cache initialization.

    Returns:
        sim_cache: list of (input_nodes, seeds, blocks, remote_mask, labels)
        cache: initialized FeatureCache
        presample_time: wall-clock seconds
        batch_count: number of batches consumed
    """
    t0 = time.time()
    sim_cache = []
    pid = g.rank()

    print(f"Part {pid} Pre-sampling up to {max_batches} batches...")

    for _ in range(max_batches):
        item = shared_buffer.get()
        if item is None:
            shared_buffer.put(None)
            break
        sim_cache.append(item)

    presample_time = time.time() - t0
    cache = FeatureCache(g, n_hot=args.cache_size, device=device,
                         dist_lock=dist_lock)
    batch_count = len(sim_cache)

    print(f"Part {pid} Pre-sampling done: {batch_count} batches in "
          f"{presample_time:.2f}s")

    return sim_cache, cache, presample_time, batch_count
