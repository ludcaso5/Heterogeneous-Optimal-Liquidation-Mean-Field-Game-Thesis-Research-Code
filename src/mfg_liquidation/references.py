"""Homogeneous reference scenarios used for heterogeneous comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .solver import compute_aggregated_flow, normalize_rho, solve_multi_population_fixed_point


def solve_homogeneous_references(
    qbar0_list: Sequence[np.ndarray],
    gamma_list: Sequence[float],
    rho_list: Sequence[float],
    mathbb_v: np.ndarray,
    impact_matrix: np.ndarray,
    a_terminal_list: Sequence[np.ndarray],
    sigma_matrix: np.ndarray,
    *,
    horizon: float,
    n_steps: int,
    relaxation_lambda: float,
    epsilon: float,
    max_iter: int,
) -> dict:
    """Construct the matched homogeneous reference for every population. """

    p = len(qbar0_list)
    d = len(np.asarray(qbar0_list[0]))
    rho = normalize_rho(rho_list)

    reference_qbar = np.zeros((p, n_steps + 1, d), dtype=float)
    reference_vbar = np.zeros((p, n_steps + 1, d), dtype=float)
    reference_mu = np.zeros((p, n_steps + 1, d), dtype=float)
    reference_info = []
    reference_t_grid = None

    for i in range(p):
        qbar0_reference = [np.asarray(qbar0_list[i], dtype=float).copy() for _ in range(p)]
        gamma_reference = [float(gamma_list[i])] * p
        a_reference = [np.asarray(a_terminal_list[i], dtype=float).copy() for _ in range(p)]

        t_grid, qbar, vbar, info = solve_multi_population_fixed_point(
            qbar0_reference,
            gamma_reference,
            rho,
            mathbb_v,
            impact_matrix,
            a_reference,
            sigma_matrix,
            horizon=horizon,
            n_steps=n_steps,
            relaxation_lambda=relaxation_lambda,
            epsilon=epsilon,
            max_iter=max_iter,
            require_convergence=True,
        )

        if reference_t_grid is None:
            reference_t_grid = t_grid

        reference_qbar[i] = qbar[0]
        reference_vbar[i] = vbar[0]
        reference_mu[i] = compute_aggregated_flow(vbar, rho)
        reference_info.append(info)

    return {
        "t_grid": reference_t_grid,
        "qbar": reference_qbar,
        "vbar": reference_vbar,
        "mu": reference_mu,
        "info": reference_info,
    }
