"""
Benchmark script using hydra.

Run it with python run_benchmark.py --config-name baseline to get baseline performance.

This script benchmarks MRI nufft with finufft and cufinufft backends for qMRI problems.
It initializes data, sets up the necessary configurations, and runs benchmarks while monitoring performance.

Usage:
    python run_benchmark.py --config-name config_name

Output:
    Performance metrics and results saved in CSV files.
    
"""

import csv
import logging
import os
import time
import warnings

from pathlib import Path

import hydra
import numpy as np

from hydra_callbacks.logger import PerfLogger
from hydra_callbacks.monitor import ResourceMonitorService

from pygrog.linop import SubspaceNUFFT
from pygrog.benchmark import generate_mrf_case

from omegaconf import DictConfig

AnyShape = tuple[int, int, int]

# Check for CUPY availability for GPU support
CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False

# Initialize logger
logger = logging.getLogger(__name__)

# Suppress specific warnings from mrinufft module
warnings.filterwarnings(
    "ignore",
    "Samples will be rescaled to .*",
    category=UserWarning,
    module="mrinufft",
)


def get_data(cfg):
    """Initialize all the data for the benchmark."""
    # Initialize trajectory
    if cfg.trajectory.endswith(".bin"):
        trajectory, params = read_trajectory(
            str(Path(__file__).parent / cfg.trajectory)
        )
    else:
        eval(trajectory.name)(**trajectory.kwargs)

    cpx_type = np.dtype(cfg.data.dtype)
    if cpx_type == np.complex64:
        trajectory = trajectory.astype(np.float32)
    C = cfg.data.n_coils
    XYZ = tuple(params["img_size"])
    K = np.prod(trajectory.shape[:-1])

    # Load or generate data
    if data_file := getattr(cfg.data, "file", None):
        data = np.load(data_file)
        if data.shape != XYZ:
            logger.warning("mismatched shape between data and trajectory file.")
    else:
        data = (1j * np.random.rand(*XYZ)).astype(cpx_type)
        data += np.random.rand(*XYZ).astype(cpx_type)

    # Generate k-space data
    ksp_data = 1j * np.random.randn(C, K).astype(cpx_type)
    ksp_data += np.random.randn(C, K).astype(cpx_type)

    # Initialize sensitivity maps
    smaps = None
    if cfg.data.n_coils > 1:
        smaps_true = get_smaps(XYZ, C)
        if cfg.data.smaps:
            smaps = smaps_true
        else:
            # Expand the data to multicoil
            data = data[None, ...] * smaps_true

    return (data, ksp_data, trajectory, smaps, XYZ, C)


@hydra.main(
    config_path="perf",
    config_name="benchmark_config",
    version_base=None,
)
def main_app(cfg: DictConfig) -> None:
    """Run the benchmark."""

    # Initialize the NUFFT operator
    nufftKlass = get_operator(cfg.backend.name)
    data, ksp_data, trajectory, smaps, shape, n_coils = get_data(cfg)
    logger.debug(
        f"{data.shape}, {ksp_data.shape}, {trajectory.shape}, {n_coils}, {shape}"
    )

    # Set up resource monitoring
    monit = ResourceMonitorService(
        interval=cfg.monitor.interval, gpu_monit=cfg.monitor.gpu
    )
    kwargs = {}
    if "stacked" in cfg.backend.name:
        kwargs["z_index"] = "auto"
    nufft = nufftKlass(
        trajectory,
        shape,
        n_coils=n_coils,
        smaps=smaps,
        **kwargs,
    )
    run_config = {
        "backend": cfg.backend.name,
        "eps": cfg.backend.eps,
        "upsampfac": cfg.backend.upsampfac,
        "n_coils": nufft.n_coils,
        "shape": nufft.shape,
        "n_samples": nufft.n_samples,
        "dim": len(nufft.shape),
        "sense": nufft.uses_sense,
    }
    trajectory_name = cfg.trajectory.split("/")[-1].split("_")[0]
    result_file = f"{cfg.backend.name}_{cfg.backend.upsampfac}_{trajectory_name}_{cfg.backend.eps}_{cfg.data.n_coils}.csv"

    # Run benchmark tasks
    for task in cfg.task:
        tic = time.perf_counter()
        toc = tic
        i = -1
        while toc - tic < cfg.max_time:
            i += 1
            nufft = nufftKlass(
                trajectory,
                shape,
                n_coils=n_coils,
                smaps=smaps,
                eps=cfg.backend.eps,
                upsampfac=cfg.backend.upsampfac,
            )
            with (
                monit,
                PerfLogger(logger, name=f"{cfg.backend.name}_{task}, #{i}") as perflog,
            ):
                if task == "forward":
                    nufft.op(data)
                elif task == "adjoint":
                    nufft.adj_op(ksp_data)
                elif task == "grad":
                    nufft.data_consistency(data, ksp_data)
                else:
                    raise ValueError(f"Unknown task {task}")
            toc = time.perf_counter()
            values = monit.get_values()
            monit_values = {
                "task": task,
                "run": i,
                "run_time": perflog.get_timer(f"{cfg.backend.name}_{task}, #{i}"),
                "mem_avg": np.mean(values["rss_GiB"]),
                "mem_peak": np.max(values["rss_GiB"]),
                "cpu_avg": np.mean(values["cpus"]),
                "cpu_peak": np.max(values["cpus"]),
            }
            if cfg.monitor.gpu:
                gpu_keys = [k for k in values.keys() if "gpu" in k]
                for k in gpu_keys:
                    monit_values[f"{k}_avg"] = np.mean(values[k])
                    monit_values[f"{k}_peak"] = np.max(values[k])

            # Save benchmark results to CSV file
            with open(result_file, "a") as f:
                row_dict = run_config | monit_values
                writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                f.seek(0, os.SEEK_END)
                if not f.tell():
                    writer.writeheader()
                writer.writerow(row_dict)
    del nufft
    if CUPY_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main_app()
