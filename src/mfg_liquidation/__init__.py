"""Heterogeneous mean-field optimal-liquidation research code."""

from .metrics import compute_execution_metrics
from .references import solve_homogeneous_references
from .solver import (
    FixedPointInfo,
    compute_aggregated_flow,
    kinematic_residual,
    normalize_rho,
    solve_multi_population_fixed_point,
    terminal_condition_residual,
)

__all__ = [
    "FixedPointInfo",
    "compute_aggregated_flow",
    "compute_execution_metrics",
    "kinematic_residual",
    "normalize_rho",
    "solve_homogeneous_references",
    "solve_multi_population_fixed_point",
    "terminal_condition_residual",
]

__version__ = "1.0.0"
