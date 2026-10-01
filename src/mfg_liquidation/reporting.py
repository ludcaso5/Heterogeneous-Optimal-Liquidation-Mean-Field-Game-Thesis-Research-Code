"""Formatting and result-table helpers for the thesis notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_execution_metrics
from .solver import compute_aggregated_flow


ROW_LABELS = (
    "q̄_T",
    "VOLUME",
    "EXPOSITION AU RISQUE",
    "COÛT TEMPORAIRE",
    "INDICE D’IMPACT PERMANENT",
)


def format_integer_fr(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{int(np.rint(float(value))):,}".replace(",", " ")


def format_decimal_fr(value: float, decimals: int = 4) -> str:
    if not np.isfinite(value):
        return "N/A"
    x = float(value)
    if abs(x) < 0.5 * 10 ** (-decimals):
        x = 0.0
    return f"{x:.{decimals}f}".replace(".", ",")


def population_metrics(
    t_grid,
    qbar,
    vbar,
    rho,
    alpha_diag,
    eta_diag,
    liquidity,
    sigma_matrix,
):
    """Return metric dictionaries for every population in one equilibrium."""

    mu = compute_aggregated_flow(vbar, rho)
    out = []
    for i in range(qbar.shape[0]):
        out.append(
            compute_execution_metrics(
                t_grid,
                qbar[i],
                vbar[i],
                mu,
                alpha_diag,
                eta_diag,
                liquidity,
                sigma_matrix,
            )
        )
    return out


def _column_values(qbar_i, metrics_i):
    return {
        "q̄_T": format_integer_fr(float(qbar_i[-1, 0])),
        "VOLUME": format_integer_fr(float(metrics_i["volume"][0])),
        "EXPOSITION AU RISQUE": format_decimal_fr(float(metrics_i["risk_normalized"])),
        "COÛT TEMPORAIRE": format_decimal_fr(float(metrics_i["temporary_cost_unit"][0])),
        "INDICE D’IMPACT PERMANENT": format_decimal_fr(
            float(metrics_i["permanent_impact_unit"][0])
        ),
    }


def build_thesis_table(
    qbar,
    metrics,
    *,
    reference_qbar=None,
    reference_metrics=None,
) -> pd.DataFrame:
    """Build the results table"""

    columns = {}
    p = qbar.shape[0]

    for i in range(p):
        columns[f"SOUS-POPULATION {i + 1}"] = _column_values(qbar[i], metrics[i])

    if reference_qbar is not None and reference_metrics is not None:
        for i in range(p):
            columns[f"POPULATION {i + 1} HOMOGÈNE"] = _column_values(
                reference_qbar[i], reference_metrics[i]
            )

    table = pd.DataFrame(columns).reindex(ROW_LABELS)
    table.index.name = ""
    return table


def table_to_tsv(table: pd.DataFrame, include_headers: bool = False) -> str:
    """Return tab-separated text """

    lines = []
    if include_headers:
        lines.append("\t".join([""] + list(table.columns)))
    for index, row in table.iterrows():
        prefix = [str(index)] if include_headers else []
        lines.append("\t".join(prefix + [str(value) for value in row]))
    return "\n".join(lines)
