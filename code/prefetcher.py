import threading, queue, time, torch as th
from collections import deque

class SharedBuffer:
    """Thread-safe buffer for passing batches between Sampler and Prefetcher"""
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = deque()
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.finished = False

    def put(self, item):
        with self.lock:
            while len(self.buffer) >= self.capacity and not self.finished:
                self.not_full.wait()
            if self.finished:
                return
            self.buffer.append(item)
            self.not_empty.notify()

    def get(self):
        with self.lock:
            while not self.buffer and not self.finished:
                self.not_empty.wait()
            if not self.buffer and self.finished:
                return None
            item = self.buffer.popleft()
            self.not_full.notify()
            return item

    def peek_n(self, n):
        """Return the next n items without removing them"""
        with self.lock:
            # Wait until we have enough items or finished
            while len(self.buffer) < n and not self.finished:
                self.not_empty.wait()
            
            count = min(n, len(self.buffer))
            return list(self.buffer)[:count]

    def size(self):
        with self.lock:
            return len(self.buffer)

    def mark_finished(self):
        with self.lock:
            self.finished = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

class BackgroundSampler(threading.Thread):
    """Producer thread that runs the DGL sampler and fills the SharedBuffer"""
    def __init__(self, sampler, buffer, g, local_mask, start_batch_id=0, dist_lock=None, initial_iter=None, num_epochs=1):
        super().__init__(daemon=True)
        self.sampler = sampler
        self.buffer = buffer
        self.g = g
        self.local_mask = local_mask
        self.current_batch_id = start_batch_id
        self.dist_lock = dist_lock
        self.initial_iter = initial_iter
        self.num_epochs = num_epochs
        self.running = True

    def run(self):
        try:
            # If initial_iter is provided we finish it as the remainder of epoch 0,
            # then iterate from epoch 1. Otherwise iterate all epochs from scratch.
            if self.initial_iter:
                self._consume_iterator(self.initial_iter)
            start_epoch = 1 if self.initial_iter else 0
            for epoch in range(start_epoch, self.num_epochs):
                if not self.running: break
                iterator = iter(self.sampler)
                self._consume_iterator(iterator)
                
        except Exception as e:
            print(f"BackgroundSampler error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.buffer.mark_finished()

    def _consume_iterator(self, iterator):
        while self.running:
            try:
                if self.dist_lock:
                    with self.dist_lock:
                        input_nodes, seeds, blocks = next(iterator)
                else:
                    input_nodes, seeds, blocks = next(iterator)
            except StopIteration:
                break
            except Exception as e:
                print(f"BackgroundSampler iterator error: {e}")
                break
            
            # Process batch
            remote_mask = ~self.local_mask[input_nodes]
            
            # Access labels safely
            if self.dist_lock:
                with self.dist_lock:
                    labels = self.g.ndata["labels"][seeds].cpu()
            else:
                labels = self.g.ndata["labels"][seeds].cpu()
            
            # Store as tuple
            batch_data = (input_nodes, seeds, blocks, remote_mask, labels)
            self.buffer.put(batch_data)
            self.current_batch_id += 1

    def stop(self):
        self.running = False

class BatchPrefetcher:
    def __init__(self, g, device, cache, window_size, shared_buffer, total_batches, batches_per_epoch, max_batches=8, synchronous_cache=False, n_classes=None, window_hot_nodes=None, dist_lock=None, initial_data=None):
        self.g, self.device, self.cache = g, device, cache
        self.n_classes = n_classes
        self.part_id = g.rank()
        self.window_size = window_size
        self.shared_buffer = shared_buffer # Replaces presample_db. Can be SharedBuffer or list.
        self.total_batches = total_batches
        self.batches_per_epoch = batches_per_epoch
        self.synchronous_cache = synchronous_cache
        self.window_hot_nodes = window_hot_nodes or {}
        self.dist_lock = dist_lock
        self.initial_data = initial_data if isinstance(initial_data, list) else None
        self.initial_data_iter = iter(initial_data) if initial_data else None
        
        try:
            self.pb = g.get_partition_book()
        except AttributeError:
            self.pb = None
        
        if not self.synchronous_cache:
            self.buffer = queue.Queue(maxsize=max_batches)
            self.stop_event = threading.Event()
            
            # Parallel Cache Building Primitives
            self.next_window_ready = threading.Event()
            self.build_next_window = threading.Event()
            self.next_window_start_batch = 0
            self.cache_builder_thread = None
        
        self.cost_weights = None  # {part_id: weight} for cost-weighted hot-node selection
        self.cache_valid_from_batch = 0
        self.cache_valid_until_batch = 0
        self.current_batch_idx = 1 # 1-based index
        self.total_popped = 0 # Track total batches consumed for offset calc
        
    def _sanitize_labels(self, labels):
        """Sanitize labels: handle NaN values and clamp to valid range"""
        if labels.is_floating_point():
            labels = th.nan_to_num(labels, nan=-1.0)
        
        labels = labels.long()
        if self.n_classes is not None:
            labels = th.clamp(labels, min=-1, max=self.n_classes-1)
        else:
            labels = th.clamp(labels, min=-1, max=1000)
        
        return labels
        
    def start_epoch(self, epoch):
        self.epoch = epoch
        
        if epoch == 0:
            self.current_batch_idx = 1
        else:
            self.current_batch_idx = epoch * self.batches_per_epoch + 1

        # Initial cache update (blocking for first window)
        # Initial cache update (blocking for first window)
        if self.current_batch_idx < self.cache_valid_from_batch or \
           self.current_batch_idx >= self.cache_valid_until_batch:
            self._update_cache_sync(self.current_batch_idx)
        
        if not self.synchronous_cache:
            self.stop_event.clear()
            
            # Start worker thread (it exits after batches_per_epoch)
            threading.Thread(target=self._worker, daemon=True).start()
            
            # Start cache builder thread ONLY if not already running
            if self.cache_builder_thread is None or not self.cache_builder_thread.is_alive():
                self.next_window_ready.clear()
                self.build_next_window.clear()
                # Queue a build for the next window immediately.
                self.next_window_start_batch = self.current_batch_idx + self.window_size
                self.build_next_window.set()
                self.cache_builder_thread = threading.Thread(target=self._cache_builder, daemon=True)
                self.cache_builder_thread.start()
    
    def _update_cache_sync(self, start_batch):
        """Synchronous cache update (used for first window or sync mode)"""
        self._build_cache_for_window(start_batch, is_sync=True)
        self.cache.swap_buffers()
        self.cache_valid_from_batch = start_batch
        self.cache_valid_until_batch = start_batch + self.window_size

    def _cache_builder(self):
        """Background thread to build cache for the NEXT window"""
        while not self.stop_event.is_set():
            if not self.build_next_window.wait(timeout=0.1):
                continue
            
            if self.stop_event.is_set():
                break
                
            self.build_next_window.clear()
            start_batch = self.next_window_start_batch
            
            self._build_cache_for_window(start_batch, is_sync=False)
            
            self.next_window_ready.set()
    
    def _worker(self):
        # Worker thread for async mode
        for step in range(self.batches_per_epoch):
            if self.stop_event.is_set(): 
                break
            
            # Check if we crossed window boundary
            if self.current_batch_idx >= self.cache_valid_until_batch:
                if not self.next_window_ready.wait(timeout=30.0):
                    print(f"[Prefetcher] Warning: Cache builder timed out for batch {self.current_batch_idx}")
                
                self.next_window_ready.clear()
                self.cache.swap_buffers()
                
                self.cache_valid_from_batch = self.current_batch_idx
                self.cache_valid_until_batch = self.current_batch_idx + self.window_size
                
                self.next_window_start_batch = self.cache_valid_until_batch
                self.build_next_window.set()
            
            # Get batch from shared buffer (or initial data)
            batch_data = None
            if self.initial_data_iter:
                try:
                    batch_data = next(self.initial_data_iter)
                except StopIteration:
                    self.initial_data_iter = None
            
            if batch_data is None:
                batch_data = self.shared_buffer.get()
            
            if batch_data is None: 
                break
                
            self.total_popped += 1
            
            input_nodes, seeds, blocks, remote_mask, labels = batch_data
            
            feats = self.cache.get_features(input_nodes, self.g, self.device, remote_mask, self.current_batch_idx)
            labels = self._sanitize_labels(labels).to(self.device, non_blocking=True)
            blocks = [b.to(self.device, non_blocking=True) for b in blocks]
            
            self.buffer.put((feats, labels, blocks))
            self.current_batch_idx += 1
        
    def _build_cache_for_window(self, start_batch, is_sync=False):
        """Build cache for window using SharedBuffer peek"""
        t0 = time.time()
        
        precomputed_hot_nodes = None
        if self.window_hot_nodes and start_batch in self.window_hot_nodes:
             precomputed_hot_nodes = self.window_hot_nodes[start_batch]
        
        t1 = time.time()
        t2 = t1
        t3 = t1
        
        if precomputed_hot_nodes is not None:
            hot_nodes = precomputed_hot_nodes
            unique_remote_nodes = len(hot_nodes)
            total_accesses = unique_remote_nodes 
        else:
            # JIT Hot Node Calculation using SharedBuffer
            if isinstance(self.shared_buffer, list):
                # Autotuning mode: use list slicing
                # start_batch is 1-based
                start_idx = start_batch - 1
                end_idx = start_idx + self.window_size
                window_items = self.shared_buffer[start_idx:end_idx]
            else:
                # Streaming mode: check initial_data first, then buffer
                window_items = []
                start_idx_in_buffer = 0
                
                # Try to fill from initial_data if available
                if self.initial_data:
                    # start_batch is 1-based index
                    curr_batch_idx = start_batch 
                    start_list_idx = curr_batch_idx - 1 # 0-based index in list
                    
                    if start_list_idx < len(self.initial_data):
                         end_list_idx = min(start_list_idx + self.window_size, len(self.initial_data))
                         window_items = self.initial_data[start_list_idx:end_list_idx]
                
                # If the window isn't filled from initial_data, peek from SharedBuffer
                # at the correct global offset (head of SharedBuffer corresponds to
                # batch index max(total_initial, total_popped) + 1).
                items_needed = self.window_size - len(window_items)
                if items_needed > 0:
                    total_initial = len(self.initial_data) if self.initial_data else 0
                    global_idx_needed = (start_batch - 1) + len(window_items)
                    effective_head_idx = max(total_initial, self.total_popped)
                    buffer_offset = max(0, global_idx_needed - effective_head_idx)
                    items = self.shared_buffer.peek_n(buffer_offset + items_needed)
                    window_items.extend(items[buffer_offset : buffer_offset + items_needed])
            
            t1 = time.time()
            
            all_input_nodes = []
            for data in window_items:
                inp, mask = data[0], data[3]
                if mask.any():
                    all_input_nodes.append(inp[mask])
            t2 = time.time()
            
            unique_remote_nodes = 0
            total_accesses = 0
            hot_nodes = th.tensor([], dtype=th.long)
            
            if all_input_nodes:
                all_remote = th.cat(all_input_nodes)
                total_accesses = all_remote.numel()
                if total_accesses > 0:
                    unique_nodes, node_counts = th.unique(all_remote, return_counts=True)
                    unique_remote_nodes = unique_nodes.numel()
                    k = min(self.cache.n_hot, unique_nodes.numel())
                    if k > 0:
                        # Cost-weighted scoring when cost_weights are provided
                        if self.cost_weights and self.pb is not None:
                            try:
                                owners = self.pb.nid2partid(unique_nodes.cpu())
                                scores = node_counts.float()
                                for pid, weight in self.cost_weights.items():
                                    mask = (owners == pid)
                                    if mask.any():
                                        scores[mask] *= weight
                                _, top_indices = th.topk(scores, k)
                            except Exception:
                                _, top_indices = th.topk(node_counts, k)
                        else:
                            _, top_indices = th.topk(node_counts, k)
                        hot_nodes = unique_nodes[top_indices]
            t3 = time.time()

        # Phase 2: Cache update
        pending_nodes, pending_features, pending_idx_map = self.cache.get_write_buffer_indices()
        active_nodes = self.cache.cache_nodes[self.cache.active_idx]
        active_features = self.cache.cache_features[self.cache.active_idx]
        
        delta_nodes_cpu = th.tensor([], dtype=th.long)
        reused_count = 0
        new_count = 0
        
        if len(hot_nodes) > 0:
            # hot_nodes is already on CPU (from th.unique on CPU tensors)
            hot_nodes_cpu = hot_nodes
            selected_count = len(hot_nodes_cpu)
            delta_nodes_cpu = hot_nodes_cpu
            
            if len(active_nodes) > 0:
                old_nodes_cpu = active_nodes.cpu()
                
                overlap_mask_cpu = th.isin(hot_nodes_cpu, old_nodes_cpu)
                reused_count = int(overlap_mask_cpu.sum())
                new_count = selected_count - reused_count
                
                # Allocate pending features buffer if size changed
                if pending_features.shape[0] != selected_count:
                    pending_features = th.empty((selected_count, self.cache.feat_dim), 
                                               dtype=active_features.dtype, device=self.device)
                
                if reused_count > 0:
                    # Transfer overlap mask to device once, derive both position sets
                    overlap_mask_dev = overlap_mask_cpu.to(self.device)
                    overlap_nodes_dev = hot_nodes_cpu[overlap_mask_cpu].to(self.device)
                    
                    active_idx_map = self.cache.cache_idx[self.cache.active_idx]
                    old_positions = active_idx_map[overlap_nodes_dev]
                    new_positions = th.nonzero(overlap_mask_dev, as_tuple=True)[0]
                    
                    pending_features[new_positions] = active_features[old_positions]
                    
                    new_nodes_cpu = hot_nodes_cpu[~overlap_mask_cpu]
                    if len(new_nodes_cpu) > 0:
                        new_positions_new = th.nonzero(~overlap_mask_dev, as_tuple=True)[0]
                        if self.dist_lock:
                            with self.dist_lock:
                                pending_features[new_positions_new] = self.g.ndata["features"][new_nodes_cpu].to(self.device)
                        else:
                            pending_features[new_positions_new] = self.g.ndata["features"][new_nodes_cpu].to(self.device)
                else:
                    # No overlap - fetch all from DistTensor
                    if self.dist_lock:
                        with self.dist_lock:
                            pending_features = self.g.ndata["features"][hot_nodes_cpu].to(self.device)
                    else:
                        pending_features = self.g.ndata["features"][hot_nodes_cpu].to(self.device)
            else:
                # No active cache - fetch all from DistTensor
                reused_count = 0
                new_count = selected_count
                if self.dist_lock:
                    with self.dist_lock:
                        pending_features = self.g.ndata["features"][hot_nodes_cpu].to(self.device)
                else:
                    pending_features = self.g.ndata["features"][hot_nodes_cpu].to(self.device)
            
            self.cache.set_write_buffer_state(hot_nodes_cpu.to(self.device), pending_features)
        else:
            empty_nodes = th.tensor([], dtype=th.long, device=self.device)
            empty_feats = th.empty((0, self.cache.feat_dim), device=self.device)
            self.cache.set_write_buffer_state(empty_nodes, empty_feats)
            
        self.cache.current_window_start = start_batch

    def get(self):
        if not self.synchronous_cache:
            return self.buffer.get()
        else:
            # Synchronous mode
            if isinstance(self.shared_buffer, list):
                 # Autotuning fallback if get() called (unlikely)
                 return None
            
            if self.initial_data_iter:
                try:
                    batch_data = next(self.initial_data_iter)
                except StopIteration:
                    self.initial_data_iter = None
                    batch_data = self.shared_buffer.get()
            else:
                batch_data = self.shared_buffer.get()

            if batch_data is None:
                raise StopIteration("No more batches")
            
            self.total_popped += 1
            
            input_nodes, seeds, blocks, remote_mask, labels = batch_data
            
            if self.current_batch_idx >= self.cache_valid_until_batch:
                self._update_cache_sync(self.current_batch_idx)
            
            feats = self.cache.get_features(input_nodes, self.g, self.device, remote_mask, self.current_batch_idx)
            labels = self._sanitize_labels(labels).to(self.device, non_blocking=True)
            blocks = [b.to(self.device, non_blocking=True) for b in blocks]
            
            self.current_batch_idx += 1
            return (feats, labels, blocks)

    def get_batch(self, batch_id):
        """Synchronous batch retrieval for autotuning"""
        if isinstance(self.shared_buffer, list):
             idx = batch_id - 1
             if 0 <= idx < len(self.shared_buffer):
                 return self.shared_buffer[idx]
             return None
        else:
             return None

    def consume_cache_metrics(self):
        return {
            "batches": self.cache.consume_batch_metrics(),
            "rebuilds": self.cache.consume_rebuild_metrics(),
        }

    def stop(self):
        if hasattr(self, 'stop_event'):
            self.stop_event.set()