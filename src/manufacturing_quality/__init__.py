"""Manufacturing quality and process analytics package."""

from .analysis import (
    compute_oee,
    detect_spc_breaches,
    fit_defect_model,
    identify_bottlenecks,
    recommend_actions,
)
from .data_simulation import build_manufacturing_dataset
from .real_benchmark import run_steel_faults_benchmark

__all__ = [
    "build_manufacturing_dataset",
    "compute_oee",
    "detect_spc_breaches",
    "fit_defect_model",
    "identify_bottlenecks",
    "recommend_actions",
    "run_steel_faults_benchmark",
]
