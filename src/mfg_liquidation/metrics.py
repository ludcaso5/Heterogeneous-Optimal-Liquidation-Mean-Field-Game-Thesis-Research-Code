"""Economic diagnostics used in the thesis."""

from __future__ import annotations

import numpy as np


def integrate_trapezoid(y, x, axis=0):
    """Trapezoidal integration with a compatibility fallback."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def compute_execution_metrics(
    t_grid: np.ndarray,
    inventory: np.ndarray,
    speed: np.ndarray,
    mu: np.ndarray,
    alpha_diag: np.ndarray,
    eta_diag: np.ndarray,
    liquidity: np.ndarray,
    sigma_matrix: np.ndarray,
) -> dict:
    """Compute the three diagnostics and auxiliary execution quantities.
    For each asset ``j`` the permanent price displacement is
    ``Delta S_perm,j(t) = alpha_j * integral_0^t mu_j(s) ds``.
    The unit permanent-impact index and unit temporary cost are then
    ``- integral v_j Delta S_perm,j dt / integral |v_j| dt``
    and
    ``integral eta_j v_j^2 / V_j dt / integral |v_j| dt``.
    Price-risk exposure is normalized by the exposure obtained by keeping the
    initial inventory unchanged over the entire horizon.
    """

    t_grid = np.asarray(t_grid, dtype=float)
    inventory = np.asarray(inventory, dtype=float)
    speed = np.asarray(speed, dtype=float)
    mu = np.asarray(mu, dtype=float)
    alpha_diag = np.asarray(alpha_diag, dtype=float)
    eta_diag = np.asarray(eta_diag, dtype=float)
    liquidity = np.asarray(liquidity, dtype=float)
    sigma_matrix = np.asarray(sigma_matrix, dtype=float)

    if inventory.ndim == 1:
        inventory = inventory[:, None]
    if speed.ndim == 1:
        speed = speed[:, None]
    if mu.ndim == 1:
        mu = mu[:, None]

    if np.any(liquidity <= 0.0):
        raise ValueError("Liquidity values must be strictly positive.")

    permanent_price_shift = np.zeros_like(mu, dtype=float)
    if len(t_grid) > 1:
        dt = np.diff(t_grid)[:, None]
        cumulative_flow = np.cumsum(0.5 * (mu[:-1] + mu[1:]) * dt, axis=0)
        permanent_price_shift[1:] = alpha_diag[None, :] * cumulative_flow

    permanent_impact_integrand = -speed * permanent_price_shift
    permanent_impact = integrate_trapezoid(
        permanent_impact_integrand, t_grid, axis=0
    )

    temporary_cost_integrand = eta_diag[None, :] * speed**2 / liquidity[None, :]
    temporary_cost = integrate_trapezoid(temporary_cost_integrand, t_grid, axis=0)
    volume = integrate_trapezoid(np.abs(speed), t_grid, axis=0)

    permanent_impact_unit = np.full_like(permanent_impact, np.nan, dtype=float)
    temporary_cost_unit = np.full_like(temporary_cost, np.nan, dtype=float)
    traded = volume > 1e-14
    permanent_impact_unit[traded] = permanent_impact[traded] / volume[traded]
    temporary_cost_unit[traded] = temporary_cost[traded] / volume[traded]

    risk_integrand = np.einsum(
        "ti,ij,tj->t", inventory, sigma_matrix, inventory
    )
    risk_exposure = float(integrate_trapezoid(risk_integrand, t_grid))

    horizon = float(t_grid[-1] - t_grid[0])
    q0 = inventory[0]
    risk_reference = horizon * float(q0 @ sigma_matrix @ q0)
    risk_normalized = (
        risk_exposure / risk_reference if risk_reference > 1e-14 else np.nan
    )

    return {
        "permanent_price_shift": permanent_price_shift,
        "permanent_impact": permanent_impact,
        "permanent_impact_unit": permanent_impact_unit,
        "permanent_impact_integrand": permanent_impact_integrand,
        "temporary_cost": temporary_cost,
        "temporary_cost_unit": temporary_cost_unit,
        "temporary_cost_integrand": temporary_cost_integrand,
        "volume": volume,
        "risk_exposure": risk_exposure,
        "risk_reference": risk_reference,
        "risk_normalized": risk_normalized,
        "risk_integrand": risk_integrand,
    }
