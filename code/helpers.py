"""GPU frequency control and post-run summary printing."""

import subprocess


def set_gpu_frequency(mode="default", device_index=None):
    """Set GPU clock via nvidia-smi. mode='min' locks to lowest graphics clock;
    mode='default' resets. Used to suppress GPU energy draw during presampling.
    """
    try:
        gpu_flag = f"-i {device_index}" if device_index is not None else ""
        if mode == "min":
            result = subprocess.run(
                f"nvidia-smi {gpu_flag} --query-supported-clocks=gr "
                "--format=csv,noheader,nounits".split(),
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                clocks = [int(c.strip()) for c in result.stdout.strip().split('\n')
                          if c.strip().isdigit()]
                if clocks:
                    min_clock = min(clocks)
                    subprocess.run(
                        f"nvidia-smi {gpu_flag} -lgc {min_clock},{min_clock}".split(),
                        capture_output=True, timeout=5)
                    print(f"[GPU Freq] Set GPU to minimum frequency: {min_clock} MHz")
                    return min_clock
        else:
            subprocess.run(f"nvidia-smi {gpu_flag} -rgc".split(),
                           capture_output=True, timeout=5)
            print("[GPU Freq] Reset GPU to default frequency")
    except Exception as e:
        print(f"[GPU Freq] Warning: Could not set GPU frequency: {e}")
    return None


def print_summary(cache, part_id, args, presample_time, total_gpu_energy, total_cpu_energy):
    """Print a one-partition summary line that parse_results.py consumes."""
    stats = cache.get_stats()
    print(f"Part {part_id}: Remote cache hit rate {stats['remote_cache_hit_rate']*100:.1f}% "
          f"(hits: {stats['remote_cache_hits']}, misses: {stats['remote_misses']}), "
          f"WS={args.window_size}, Cache={args.cache_size}")
    print(f"Part {part_id}: Total presample time: {presample_time:.2f}s")
    print(f"Part {part_id}: Total GPU energy consumed: {total_gpu_energy:.2f}J")
    print(f"Part {part_id}: Total CPU energy consumed: {total_cpu_energy:.2f}J")
    print(f"Part {part_id}: Total energy consumed: {total_gpu_energy + total_cpu_energy:.2f}J")
