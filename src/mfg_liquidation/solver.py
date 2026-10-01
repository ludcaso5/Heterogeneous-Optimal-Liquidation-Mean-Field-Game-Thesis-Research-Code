"""Numerical solver for the heterogeneous mean-field liquidation model.

The implementation follows the discrete forward-backward system used in the
thesis.  The sparse linear system is factorized once for a fixed scenario and
reused across fixed-point iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class FixedPointInfo:
    """Diagnostics returned by the fixed-point solver."""

    n_iter: int
    error: float
    converged: bool


def normalize_rho(rho_raw: Sequence[float]) -> np.ndarray:
    """Validate and normalize population masses so they sum to one."""

    rho = np.asarray(rho_raw, dtype=float)
    if rho.ndim != 1:
        raise ValueError("Population masses must be a one-dimensional sequence.")
    if np.any(rho < 0.0):
        raise ValueError("Population masses must be non-negative.")

    total = float(rho.sum())
    if total <= 0.0:
        raise ValueError("The sum of population masses must be strictly positive.")
    return rho / total


def compute_aggregated_flow(vbar: np.ndarray, rho_list: Sequence[float]) -> np.ndarray:
    """Return the aggregate trading flow ``mu = sum_k rho_k * vbar_k``."""

    return np.tensordot(normalize_rho(rho_list), np.asarray(vbar, dtype=float), axes=(0, 0))


def _validate_inputs(
    qbar0_list: Sequence[np.ndarray],
    gamma_list: Sequence[float],
    rho_list: Sequence[float],
    mathbb_v: np.ndarray,
    impact_matrix: np.ndarray,
    a_terminal_list: Sequence[np.ndarray],
    sigma_matrix: np.ndarray,
    horizon: float,
    n_steps: int,
    relaxation_lambda: float,
    epsilon: float,
    max_iter: int,
) -> tuple[int, int, np.ndarray]:
    """Validate solver inputs and return ``(P, d, normalized_rho)``."""

    if horizon <= 0.0:
        raise ValueError("The liquidation horizon must be strictly positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be strictly positive.")
    if not 0.0 < relaxation_lambda <= 1.0:
        raise ValueError("relaxation_lambda must belong to (0, 1].")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be strictly positive.")

    p = len(qbar0_list)
    if p == 0:
        raise ValueError("At least one population is required.")

    d = len(np.asarray(qbar0_list[0], dtype=float))
    if d == 0:
        raise ValueError("At least one asset is required.")

    if any(len(np.asarray(q0, dtype=float)) != d for q0 in qbar0_list):
        raise ValueError("All initial mean-inventory vectors must have the same length.")
    if len(gamma_list) != p or len(rho_list) != p or len(a_terminal_list) != p:
        raise ValueError(
            "gamma_list, rho_list and a_terminal_list must match the number of populations."
        )

    expected_shape = (d, d)
    for name, matrix in (
        ("mathbb_v", mathbb_v),
        ("impact_matrix", impact_matrix),
        ("sigma_matrix", sigma_matrix),
    ):
        if np.asarray(matrix).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")

    for i, matrix in enumerate(a_terminal_list):
        if np.asarray(matrix).shape != expected_shape:
            raise ValueError(f"a_terminal_list[{i}] must have shape {expected_shape}.")

    return p, d, normalize_rho(rho_list)


def _build_factorized_system(
    qbar0_list: Sequence[np.ndarray],
    gamma_list: Sequence[float],
    mathbb_v: np.ndarray,
    impact_matrix: np.ndarray,
    a_terminal_list: Sequence[np.ndarray],
    sigma_matrix: np.ndarray,
    horizon: float,
    n_steps: int,
) -> dict:
    """Build and factorize the sparse linear system for a fixed scenario."""

    p = len(qbar0_list)
    d = len(np.asarray(qbar0_list[0]))
    dt = horizon / n_steps

    qbar_size = p * n_steps * d
    n_rows = 2 * p * n_steps * d + p * d
    total_size = qbar_size + p * (n_steps + 1) * d

    matrix = lil_matrix((n_rows, total_size), dtype=float)
    rhs_base = np.zeros(n_rows, dtype=float)

    terminal_matrices = [
        4.0 * (mathbb_v @ np.asarray(a_terminal_list[i], dtype=float))
        for i in range(p)
    ]
    flow_impact_term = 2.0 * (mathbb_v @ impact_matrix)

    row = 0

    for i in range(p):
        qbar0 = np.asarray(qbar0_list[i], dtype=float)
        for k in range(1, n_steps + 1):
            for j in range(d):
                idx_q_k = (i * n_steps + k - 1) * d + j
                idx_v_km1 = qbar_size + (i * (n_steps + 1) + k - 1) * d + j
                matrix[row + j, idx_q_k] = 1.0
                matrix[row + j, idx_v_km1] = -dt

                if k > 1:
                    idx_q_km1 = (i * n_steps + k - 2) * d + j
                    matrix[row + j, idx_q_km1] = -1.0

            if k == 1:
                rhs_base[row : row + d] = qbar0
            row += d

    dynamic_start = row

    for i in range(p):
        c_i = 2.0 * float(gamma_list[i]) * (mathbb_v @ sigma_matrix)
        qbar0 = np.asarray(qbar0_list[i], dtype=float)

        for k in range(1, n_steps + 1):
            for j in range(d):
                idx_v_k = qbar_size + (i * (n_steps + 1) + k) * d + j
                idx_v_km1 = qbar_size + (i * (n_steps + 1) + k - 1) * d + j
                matrix[row + j, idx_v_k] += 1.0
                matrix[row + j, idx_v_km1] += -1.0

                for j2 in range(d):
                    if k > 1:
                        idx_q_km1 = (i * n_steps + k - 2) * d + j2
                        matrix[row + j, idx_q_km1] += -dt * c_i[j, j2]
                    else:
                        rhs_base[row + j] += dt * c_i[j, j2] * qbar0[j2]

            row += d

    for i in range(p):
        g_i = terminal_matrices[i]
        for j in range(d):
            idx_v_n = qbar_size + (i * (n_steps + 1) + n_steps) * d + j
            matrix[row + j, idx_v_n] = 1.0
            for j2 in range(d):
                idx_q_n = (i * n_steps + n_steps - 1) * d + j2
                matrix[row + j, idx_q_n] += g_i[j, j2]
        row += d

    if row != n_rows:
        raise RuntimeError("Internal sparse-system row count is inconsistent.")

    return {
        "lu": splu(csc_matrix(matrix)),
        "rhs_base": rhs_base,
        "dynamic_start": dynamic_start,
        "flow_impact_term": np.asarray(flow_impact_term, dtype=float),
        "qbar_size": qbar_size,
        "p": p,
        "d": d,
        "dt": dt,
        "n_steps": n_steps,
    }


def _unpack_solution(solution: np.ndarray, system: dict, qbar0_list) -> tuple[np.ndarray, np.ndarray]:
    """Convert the flattened sparse-system solution to qbar and vbar arrays."""

    p = system["p"]
    d = system["d"]
    n_steps = system["n_steps"]
    qbar_size = system["qbar_size"]

    qbar = np.zeros((p, n_steps + 1, d), dtype=float)
    vbar = np.zeros((p, n_steps + 1, d), dtype=float)

    for i in range(p):
        qbar[i, 0] = np.asarray(qbar0_list[i], dtype=float)

        for k in range(1, n_steps + 1):
            start = (i * n_steps + k - 1) * d
            qbar[i, k] = solution[start : start + d]

        for k in range(n_steps + 1):
            start = qbar_size + (i * (n_steps + 1) + k) * d
            vbar[i, k] = solution[start : start + d]

    return qbar, vbar


def solve_multi_population_fixed_point(
    qbar0_list: Sequence[np.ndarray],
    gamma_list: Sequence[float],
    rho_list: Sequence[float],
    mathbb_v: np.ndarray,
    impact_matrix: np.ndarray,
    a_terminal_list: Sequence[np.ndarray],
    sigma_matrix: np.ndarray,
    *,
    horizon: float = 1.0,
    n_steps: int = 2000,
    relaxation_lambda: float = 0.35,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    require_convergence: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FixedPointInfo]:
    """Solve the fixed point on the aggregate flow."""

    p, d, rho = _validate_inputs(
        qbar0_list,
        gamma_list,
        rho_list,
        np.asarray(mathbb_v, dtype=float),
        np.asarray(impact_matrix, dtype=float),
        a_terminal_list,
        np.asarray(sigma_matrix, dtype=float),
        horizon,
        n_steps,
        relaxation_lambda,
        epsilon,
        max_iter,
    )

    system = _build_factorized_system(
        qbar0_list,
        gamma_list,
        np.asarray(mathbb_v, dtype=float),
        np.asarray(impact_matrix, dtype=float),
        a_terminal_list,
        np.asarray(sigma_matrix, dtype=float),
        horizon,
        n_steps,
    )

    vbar_guess = np.zeros((p, n_steps + 1, d), dtype=float)
    for i in range(p):
        vbar_guess[i] = -np.outer(
            np.ones(n_steps + 1), np.asarray(qbar0_list[i], dtype=float) / horizon
        )

    mu_guess = np.tensordot(rho, vbar_guess, axes=(0, 0))
    error = np.inf
    n_iter = 0
    qbar = None
    vbar = None

    while n_iter < max_iter:
        rhs = system["rhs_base"].copy()
        dynamic_flow = -system["dt"] * (
            mu_guess[:-1] @ system["flow_impact_term"].T
        )

        block_size = n_steps * d
        for i in range(p):
            start = system["dynamic_start"] + i * block_size
            rhs[start : start + block_size] += dynamic_flow.reshape(-1)

        solution = system["lu"].solve(rhs)
        qbar, vbar = _unpack_solution(solution, system, qbar0_list)

        mu_new = np.tensordot(rho, vbar, axes=(0, 0))
        error = float(np.max(np.abs(mu_new - mu_guess)))

        vbar_guess = relaxation_lambda * vbar + (1.0 - relaxation_lambda) * vbar_guess
        mu_guess = np.tensordot(rho, vbar_guess, axes=(0, 0))

        n_iter += 1
        if error < epsilon:
            break

    info = FixedPointInfo(
        n_iter=n_iter,
        error=error,
        converged=bool(error < epsilon),
    )

    if require_convergence and not info.converged:
        raise RuntimeError(
            "Fixed-point iteration did not converge: "
            f"error={info.error:.6e}, epsilon={epsilon:.6e}, max_iter={max_iter}."
        )

    t_grid = np.linspace(0.0, horizon, n_steps + 1)
    return t_grid, qbar, vbar, info


def terminal_condition_residual(
    qbar: np.ndarray,
    vbar: np.ndarray,
    mathbb_v: np.ndarray,
    a_terminal_list: Sequence[np.ndarray],
) -> float:
    """Return the largest absolute terminal-condition residual."""

    residuals = []
    for i, a_terminal in enumerate(a_terminal_list):
        residual = vbar[i, -1] + 4.0 * (
            np.asarray(mathbb_v, dtype=float)
            @ np.asarray(a_terminal, dtype=float)
            @ qbar[i, -1]
        )
        residuals.append(np.max(np.abs(residual)))
    return float(np.max(residuals))


def kinematic_residual(t_grid: np.ndarray, qbar: np.ndarray, vbar: np.ndarray) -> float:
    """Return the largest discrete kinematic residual."""

    dt = np.diff(np.asarray(t_grid, dtype=float))
    residual = qbar[:, 1:, :] - qbar[:, :-1, :] - dt[None, :, None] * vbar[:, :-1, :]
    return float(np.max(np.abs(residual)))
