import time
import threading
import subprocess
from contextlib import contextmanager

class AccurateEnergyMonitor:
    """Basic GPU energy monitoring"""
    def __init__(self, device_index=None, tick=0.1):
        self.gpu_energy_j = 0.0
        self.start_time = None
        self.last_time = None
        self.device_index = device_index
        self.tick = tick
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._init_gpu()
    
    def _init_gpu(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            if self.device_index is None:
                import torch as th
                if th.cuda.is_available():
                    self.device_index = th.cuda.current_device()
                else:
                    self.device_index = 0
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.nvml = pynvml
            self.gpu_ok = True
        except Exception as e:
            print(f"[GPU Monitor Init Error] device_index={self.device_index}: {type(e).__name__}: {e}")
            self.gpu_ok = False
    
    def start(self):
        with self._lock:
            self.start_time = time.time()
            self.last_time = self.start_time
            self.gpu_energy_j = 0.0
            self._running = True
            if self.gpu_ok:
                self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self._thread.start()
    
    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.gpu_ok and self.last_time:
            self._update_gpu_energy()
    
    def _monitor_loop(self):
        """Polling loop that integrates power over time"""
        self._last_t = time.monotonic()
        self._last_p = self._read_power_watts()
        
        while self._running:
            time.sleep(self.tick)
            if not self._running:
                break
            
            # Integrate using left-Riemann (simple and robust)
            t = time.monotonic()
            p = self._read_power_watts()
            dt = max(0.0, t - self._last_t)
            
            with self._lock:
                self.gpu_energy_j += self._last_p * dt
            
            self._last_t = t
            self._last_p = p
    
    def _read_power_watts(self):
        """Read current power in Watts"""
        if not self.gpu_ok:
            return 0.0
        try:
            power_mw = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0
        except Exception:
            return 0.0
    
    def _update_gpu_energy(self):
        """Legacy method - now handled by polling loop"""
        pass
    
    @contextmanager
    def tag(self, name: str):
        yield
    
    def get_total_gpu_energy(self):
        if self.gpu_ok:
            self._update_gpu_energy()
        with self._lock:
            return self.gpu_energy_j


class CPUEnergyMonitor:
    """Monitor CPU energy consumption using Intel RAPL"""
    
    def __init__(self, verbose=True):
        self.rapl_domains = []
        self.start_energy = {}
        self.end_energy = {}
        self.is_monitoring = False
        self.verbose = verbose
        self.cpu_ok = False
        self.baseline_energy = {}  # For continuous delta measurement
        
        # Discover all RAPL domains
        self._discover_rapl_domains()
        self.cpu_ok = len(self.rapl_domains) > 0
        
        if not self.cpu_ok and verbose:
            print(f"[CPU Monitor Init] No RAPL domains found. RAPL may require permissions or be unavailable.")
        
    def _discover_rapl_domains(self):
        """Discover all available RAPL energy domains"""
        import os
        import glob
        
        # Only read top-level package domains (intel-rapl:X).
        # Skip sub-domains (intel-rapl:X:Y like dram) to avoid double-counting
        # and inconsistent availability across nodes.
        possible_patterns = [
            '/sys/class/powercap/intel-rapl:*/energy_uj',
        ]
        
        candidate_paths = []
        for pattern in possible_patterns:
            candidate_paths.extend(glob.glob(pattern))
        
        seen_paths = set()
        for path in candidate_paths:
            try:
                # Skip duplicates
                if path in seen_paths:
                    continue
                
                # Check if path exists and is readable (no sudo)
                if not os.path.exists(path):
                    continue
                if not os.access(path, os.R_OK):
                    continue
                
                seen_paths.add(path)
                
                # Get domain name
                name_path = path.replace('energy_uj', 'name')
                if os.path.exists(name_path) and os.access(name_path, os.R_OK):
                    try:
                        with open(name_path, 'r') as f:
                            domain_name = f.read().strip()
                    except:
                        domain_name = os.path.basename(os.path.dirname(path))
                else:
                    domain_name = os.path.basename(os.path.dirname(path))
                
                # Get max energy range for wraparound handling
                max_energy_path = path.replace('energy_uj', 'max_energy_range_uj')
                max_energy = None
                if os.path.exists(max_energy_path) and os.access(max_energy_path, os.R_OK):
                    try:
                        with open(max_energy_path, 'r') as f:
                            max_energy = int(f.read().strip())
                    except:
                        pass
                
                # Validate domain returns actual energy data before adding
                test_energy = self._read_energy_static(path)
                if test_energy is None:
                    continue
                
                self.rapl_domains.append({
                    'path': path,
                    'name': domain_name,
                    'max_energy_uj': max_energy
                })
                if self.verbose:
                    print(f"Found RAPL domain: {domain_name} at {path} (current={test_energy})")
                    
            except Exception as e:
                continue  # Skip domains that don't exist or aren't accessible
            
    @staticmethod
    def _read_energy_static(path):
        """Read energy value once (used during discovery to validate domains)"""
        try:
            with open(path, 'r') as f:
                val = f.read().strip()
                if not val:
                    return None
                return int(val)
        except (IOError, ValueError, PermissionError):
            return None

    def _read_energy(self, path):
        """Read energy value from RAPL with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(path, 'r') as f:
                    energy_uj = int(f.read().strip())
                return energy_uj
            except (IOError, ValueError, PermissionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.05)  # Brief delay before retry
                    continue
                else:
                    if self.verbose:
                        print(f"Error reading {path}: {e}")
                    return None
        return None
    
    def start(self):
        """Start monitoring - record initial energy values"""
        if not self.rapl_domains:
            if self.verbose:
                print("Warning: No RAPL domains available for CPU energy monitoring")
            return
            
        self.is_monitoring = True
        self.start_energy = {}
        self.baseline_energy = {}  # For continuous measurement
        
        for domain in self.rapl_domains:
            energy = self._read_energy(domain['path'])
            if energy is not None:
                self.start_energy[domain['name']] = energy
                self.baseline_energy[domain['name']] = energy
                if self.verbose:
                    print(f"CPU Energy Monitor: {domain['name']} start = {energy} µJ")
            else:
                if self.verbose:
                    print(f"Warning: Could not read start energy for {domain['name']}")
                
    def stop(self):
        """Stop monitoring - record final energy values"""
        if not self.is_monitoring:
            if self.verbose:
                print("Warning: CPU energy monitor was not started")
            return
            
        self.end_energy = {}
        
        for domain in self.rapl_domains:
            energy = self._read_energy(domain['path'])
            if energy is not None:
                self.end_energy[domain['name']] = energy
                if self.verbose:
                    print(f"CPU Energy Monitor: {domain['name']} end = {energy} µJ")
            else:
                if self.verbose:
                    print(f"Warning: Could not read end energy for {domain['name']}")
                
        self.is_monitoring = False
        
    def get_total_cpu_energy(self):
        """
        Calculate total CPU energy consumed in Joules.
        If monitoring has stopped, uses captured end_energy.
        Otherwise, reads current values for continuous delta measurement.
        """
        if not self.cpu_ok:
            return 0.0
        
        if not self.baseline_energy:
            return 0.0
        
        # If we've stopped, use the captured end_energy values
        if not self.is_monitoring and self.end_energy:
            total_energy_uj = 0
            for domain_name in self.baseline_energy:
                if domain_name in self.end_energy:
                    baseline = self.baseline_energy[domain_name]
                    end = self.end_energy[domain_name]
                    
                    # Find the domain to get max_energy_uj
                    max_energy_uj = None
                    for domain in self.rapl_domains:
                        if domain['name'] == domain_name:
                            max_energy_uj = domain.get('max_energy_uj')
                            break
                    
                    # Handle wraparound
                    if end < baseline:
                        if max_energy_uj:
                            energy_diff = (max_energy_uj - baseline) + end
                        else:
                            continue  # Skip if we can't handle wraparound
                    else:
                        energy_diff = end - baseline
                    
                    total_energy_uj += energy_diff
            
            return total_energy_uj / 1e6
        
        # Otherwise, read current values (for continuous monitoring during training)
        total_energy_uj = 0
        
        for domain in self.rapl_domains:
            domain_name = domain['name']
            if domain_name not in self.baseline_energy:
                continue
            
            current_energy = self._read_energy(domain['path'])
            if current_energy is None:
                continue
            
            baseline = self.baseline_energy[domain_name]
            
            # Handle counter wraparound using actual max_energy_range_uj
            if current_energy < baseline:
                # Counter wrapped around
                max_energy_uj = domain.get('max_energy_uj')
                if max_energy_uj:
                    energy_diff = (max_energy_uj - baseline) + current_energy
                else:
                    # Fallback: skip this reading
                    continue
            else:
                energy_diff = current_energy - baseline
                
            total_energy_uj += energy_diff
                
        # Convert microjoules to joules
        total_energy_j = total_energy_uj / 1e6
        return total_energy_j
    
    def set_verbose(self, verbose):
        """Set verbose mode on/off"""
        self.verbose = verbose
    
    def get_energy_breakdown(self):
        """Get energy breakdown by domain"""
        if not self.start_energy or not self.end_energy:
            return {}
            
        breakdown = {}
        
        for domain_name in self.start_energy:
            if domain_name in self.end_energy:
                start = self.start_energy[domain_name]
                end = self.end_energy[domain_name]
                
                if end < start:
                    # Handle wraparound
                    energy_diff = (2**32 - start) + end
                else:
                    energy_diff = end - start
                    
                breakdown[domain_name] = energy_diff / 1e6  # Convert to Joules
                
        return breakdown
