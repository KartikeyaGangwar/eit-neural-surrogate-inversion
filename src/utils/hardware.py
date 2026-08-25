"""
Hardware Profiler and High-Resolution Execution Timer
=====================================================
Provides centralized machine specification detection, device introspection
(CPU vs GPU, cores, VRAM, RAM), and high-precision execution wall-clock timing
for computational benchmarks across all pipeline stages.

No EIDORS dependency. Runs on local machines and Google Colab.
"""

from __future__ import annotations
import os
import sys
import time
import platform
import psutil
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import torch


def get_hardware_specs() -> Dict[str, Any]:
    """
    Probe and return complete host system hardware specifications.
    
    Returns:
        Dict containing OS, CPU model, cores, RAM, GPU details, and CUDA version.
    """
    cuda_avail = torch.cuda.is_available()
    specs = {
        "os_platform": f"{platform.system()} {platform.release()} ({platform.version()})",
        "architecture": platform.machine(),
        "processor": platform.processor() or "Unknown CPU",
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores": psutil.cpu_count(logical=True) or 1,
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "cuda_available": cuda_avail,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_avail else None,
        "gpu_count": torch.cuda.device_count() if cuda_avail else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution)",
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if cuda_avail else 0.0,
    }
    return specs


def format_time(seconds: float) -> str:
    """Format elapsed seconds into a human-readable duration string."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000.0:.2f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    elif seconds < 3600.0:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"
    else:
        hrs = int(seconds // 3600)
        rem = seconds % 3600
        mins = int(rem // 60)
        secs = rem % 60
        return f"{hrs}h {mins}m {secs:.2f}s"


def print_hardware_header(
    phase_title: str,
    device: Optional[Union[torch.device, str]] = None,
    phase_type: str = "GPU",
    workers: Optional[int] = None,
) -> None:
    """
    Print an authoritative hardware profiling banner for manuscript traceability.
    
    Args:
        phase_title: Name of the current pipeline phase or benchmark.
        device:      Target PyTorch device instance or string ('cuda' / 'cpu').
        phase_type:  Primary computing resource ('CPU', 'GPU', 'Multi-Worker CPU').
        workers:     Number of parallel CPU worker processes if applicable.
    """
    specs = get_hardware_specs()
    dev_str = str(device) if device is not None else ("cuda" if specs["cuda_available"] else "cpu")
    
    print("=" * 80)
    print(f"  {phase_title.upper()}")
    print("=" * 80)
    print(f"  Execution Target:  {phase_type}")
    print(f"  Compute Device:    {dev_str.upper()} ({specs['gpu_name'] if 'cuda' in dev_str.lower() else specs['processor']})")
    
    if "cuda" in dev_str.lower() and specs["cuda_available"]:
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"  GPU Hardware:      {specs['gpu_name']} ({specs['gpu_vram_gb']:.2f} GB total VRAM | CUDA {specs['cuda_version']})")
        print(f"  Initial VRAM Used: {vram_alloc:.2f} MB")
    else:
        w_str = f" | {workers} Parallel Worker(s)" if workers else ""
        print(f"  Host CPU:          {specs['processor']} ({specs['physical_cores']} Cores / {specs['logical_cores']} Threads{w_str})")
        print(f"  Host System RAM:   {specs['total_ram_gb']:.2f} GB ({specs['available_ram_gb']:.2f} GB Available)")
        
    print(f"  Host Platform:     {specs['os_platform']}")
    print("-" * 80)


class PrecisionTimer:
    """
    High-precision wall-clock and CPU execution timer with CUDA synchronization.
    """

    def __init__(self, name: str = "Operation", synchronize_cuda: bool = True) -> None:
        self.name = name
        self.synchronize_cuda = synchronize_cuda and torch.cuda.is_available()
        self.t_start_wall: float = 0.0
        self.t_end_wall: float = 0.0
        self.t_start_cpu: float = 0.0
        self.t_end_cpu: float = 0.0
        self.is_running: bool = False

    def start(self) -> PrecisionTimer:
        """Start or restart the precision timer."""
        if self.synchronize_cuda:
            torch.cuda.synchronize()
        self.t_start_cpu = time.process_time()
        self.t_start_wall = time.perf_counter()
        self.is_running = True
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed wall-clock seconds."""
        if self.synchronize_cuda:
            torch.cuda.synchronize()
        self.t_end_wall = time.perf_counter()
        self.t_end_cpu = time.process_time()
        self.is_running = False
        return self.elapsed_wall

    @property
    def elapsed_wall(self) -> float:
        """Elapsed wall-clock time in seconds."""
        if self.is_running:
            if self.synchronize_cuda:
                torch.cuda.synchronize()
            return time.perf_counter() - self.t_start_wall
        return self.t_end_wall - self.t_start_wall

    @property
    def elapsed_cpu(self) -> float:
        """Elapsed CPU execution time in seconds."""
        if self.is_running:
            return time.process_time() - self.t_start_cpu
        return self.t_end_cpu - self.t_start_cpu

    def __enter__(self) -> PrecisionTimer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def summary(self, n_items: Optional[int] = None, item_label: str = "item") -> str:
        """Return formatted summary of elapsed wall-clock, CPU time, and throughput."""
        wall_s = self.elapsed_wall
        cpu_s = self.elapsed_cpu
        wall_str = format_time(wall_s)
        
        if n_items is not None and n_items > 0:
            per_item_ms = (wall_s / n_items) * 1000.0
            throughput = n_items / wall_s if wall_s > 0 else 0.0
            return (
                f"[{self.name}] Total Time: {wall_str} (CPU: {cpu_s:.2f}s) | "
                f"Avg: {per_item_ms:.2f} ms/{item_label} ({throughput:.2f} {item_label}s/sec)"
            )
        return f"[{self.name}] Total Time: {wall_str} (CPU Time: {cpu_s:.2f}s)"
