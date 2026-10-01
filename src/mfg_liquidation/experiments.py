"""High-level experiment runners for the thesis scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .parameters import (
    ALPHA,
    EPSILON,
    ETA,
    HORIZON,
    LIQUIDITY,
    MAX_ITER,
    N_STEPS,
    RELAXATION_LAMBDA,
    SIGMA,
    ThesisScenario,
)
from .references import solve_homogeneous_references
from .reporting import build_thesis_table, population_metrics, table_to_tsv
from .solver import solve_multi_population_fixed_point


def common_market_objects():
    """Return the one-asset matrices used in the main numerical chapter."""

    liquidity = np.array([LIQUIDITY], dtype=float)
    eta_diag = np.array([ETA], dtype=float)
    alpha_diag = np.array([ALPHA], dtype=float)
    mathbb_v = np.diag(liquidity / (4.0 * eta_diag))
    impact_matrix = np.diag(alpha_diag)
    sigma_matrix = np.array([[SIGMA**2]], dtype=float)
    return liquidity, eta_diag, alpha_diag, mathbb_v, impact_matrix, sigma_matrix


def run_thesis_scenario(scenario: ThesisScenario) -> dict:
    """Run one Chapter-5 scenario and its matched references when required."""

    (
        liquidity,
        eta_diag,
        alpha_diag,
        mathbb_v,
        impact_matrix,
        sigma_matrix,
    ) = common_market_objects()

    qbar0_list = [np.array([value], dtype=float) for value in scenario.qbar0]
    a_terminal_list = [np.diag([value]) for value in scenario.a_terminal]

    t_grid, qbar, vbar, info = solve_multi_population_fixed_point(
        qbar0_list,
        scenario.gamma,
        scenario.rho,
        mathbb_v,
        impact_matrix,
        a_terminal_list,
        sigma_matrix,
        horizon=HORIZON,
        n_steps=N_STEPS,
        relaxation_lambda=RELAXATION_LAMBDA,
        epsilon=EPSILON,
        max_iter=MAX_ITER,
        require_convergence=True,
    )

    metrics = population_metrics(
        t_grid,
        qbar,
        vbar,
        scenario.rho,
        alpha_diag,
        eta_diag,
        liquidity,
        sigma_matrix,
    )

    references = None
    reference_metrics = None
    if scenario.compare_to_homogeneous_reference:
        references = solve_homogeneous_references(
            qbar0_list,
            scenario.gamma,
            scenario.rho,
            mathbb_v,
            impact_matrix,
            a_terminal_list,
            sigma_matrix,
            horizon=HORIZON,
            n_steps=N_STEPS,
            relaxation_lambda=RELAXATION_LAMBDA,
            epsilon=EPSILON,
            max_iter=MAX_ITER,
        )
        reference_metrics = []
        for i in range(len(scenario.qbar0)):
            reference_metrics.extend(
                population_metrics(
                    references["t_grid"],
                    references["qbar"][i : i + 1],
                    references["vbar"][i : i + 1],
                    [1.0],
                    alpha_diag,
                    eta_diag,
                    liquidity,
                    sigma_matrix,
                )
            )

    table = build_thesis_table(
        qbar,
        metrics,
        reference_qbar=None if references is None else references["qbar"],
        reference_metrics=reference_metrics,
    )

    return {
        "scenario": scenario,
        "t_grid": t_grid,
        "qbar": qbar,
        "vbar": vbar,
        "info": info,
        "metrics": metrics,
        "references": references,
        "reference_metrics": reference_metrics,
        "table": table,
    }


def save_scenario_table(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """Save a machine-readable CSV and a Word-friendly TSV table."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = result["scenario"].key

    csv_path = output_dir / f"{key}_table.csv"
    tsv_path = output_dir / f"{key}_word.tsv"

    result["table"].to_csv(csv_path, encoding="utf-8-sig")
    tsv_path.write_text(table_to_tsv(result["table"]), encoding="utf-8")
    return csv_path, tsv_path
