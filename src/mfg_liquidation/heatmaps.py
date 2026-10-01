"""Heatmap sensitivity analysis for Chapter 5.

The parameter ranges, numerical settings and visual conventions reproduce the
latest heatmap notebook supplied with the thesis project.  Raw result CSV files
are never modified by the figure-regeneration functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .metrics import compute_execution_metrics
from .parameters import ALPHA, ETA, LIQUIDITY, SIGMA
from .solver import compute_aggregated_flow, normalize_rho, solve_multi_population_fixed_point


HEATMAP_HORIZON = 1.0
HEATMAP_N_STEPS = 1000
HEATMAP_RELAXATION_LAMBDA = 0.35
HEATMAP_EPSILON = 1e-8
HEATMAP_MAX_ITER = 250
HEATMAP_GRID_SIZE = 50

BASE_PARAMS = {
    "gamma": [5e-6, 5e-6],
    "A_terminal": [1e-7, 1e-7],
    "rho_raw": [0.50, 0.50],
    "qbar0": [100_000.0, 100_000.0],
}


@dataclass(frozen=True)
class HeatmapScenario:
    key: str
    title: str
    parameter: str
    high: float
    low: float
    x_label: str
    y_label: str


@dataclass(frozen=True)
class HeatmapSpec:
    key: str
    result_column: str
    title: str
    cbar_label: str
    positive_is_better: bool


HEATMAP_SCENARIOS = (
    HeatmapScenario(
        "aversion_risque_gamma",
        "Aversion au risque",
        "gamma",
        1e-4,
        1e-7,
        r"$\gamma$ — sous-population 2",
        r"$\gamma$ — sous-population 1",
    ),
    HeatmapScenario(
        "aversion_inventaire_terminal_A",
        "Pénalisation de l'inventaire terminal",
        "A_terminal",
        1e-5,
        1e-10,
        r"$A$ — sous-population 2",
        r"$A$ — sous-population 1",
    ),
    HeatmapScenario(
        "inventaire_initial_E0",
        "Inventaire initial",
        "qbar0",
        1_000_000.0,
        1_000.0,
        r"$\bar{q}_0$ — sous-population 2",
        r"$\bar{q}_0$ — sous-population 1",
    ),
)

HEATMAP_SPECS = (
    HeatmapSpec(
        "indice_unitaire_impact_permanent",
        "delta_pop1_permanent_impact_unit",
        "Indice unitaire d'impact permanent",
        "Δ indice unitaire d'impact permanent",
        True,
    ),
    HeatmapSpec(
        "cout_temporaire_unitaire",
        "delta_pop1_temporary_cost_unit",
        "Coût temporaire unitaire",
        "Δ coût temporaire unitaire",
        False,
    ),
    HeatmapSpec(
        "exposition_risque_normalisee",
        "delta_pop1_risk_normalized",
        "Exposition au risque de prix normalisée",
        "Δ exposition au risque de prix normalisée",
        False,
    ),
    HeatmapSpec(
        "inventaire_final_residuel",
        "delta_pop1_abs_final_inventory",
        "Distance à zéro de l’inventaire terminal",
        r"$\Delta |\bar{q}_T|$",
        False,
    ),
)


def geometric_grid(high: float, low: float, n_values: int = HEATMAP_GRID_SIZE) -> list[float]:
    """Return a decreasing geometric grid containing exactly ``n_values`` values."""

    high = float(high)
    low = float(low)
    if high <= 0.0 or low <= 0.0:
        raise ValueError("Heatmap bounds must be strictly positive.")
    if high <= low:
        raise ValueError("high must be strictly greater than low.")
    return np.geomspace(high, low, num=n_values).tolist()


def _market_objects():
    liquidity = np.array([LIQUIDITY], dtype=float)
    eta_diag = np.array([ETA], dtype=float)
    alpha_diag = np.array([ALPHA], dtype=float)
    mathbb_v = np.diag(liquidity / (4.0 * eta_diag))
    impact_matrix = np.diag(alpha_diag)
    sigma_matrix = np.array([[SIGMA**2]], dtype=float)
    return liquidity, eta_diag, alpha_diag, mathbb_v, impact_matrix, sigma_matrix


def _params_copy(base):
    return {key: list(values) for key, values in base.items()}


def make_params_for_cell(scenario: HeatmapScenario, pop1_value: float, pop2_value: float):
    params = _params_copy(BASE_PARAMS)
    params[scenario.parameter] = [float(pop1_value), float(pop2_value)]
    return params


def _metric_row(t_grid, qbar_i, vbar_i, mu, pop: int, prefix: str):
    liquidity, eta_diag, alpha_diag, _, _, sigma_matrix = _market_objects()
    metrics = compute_execution_metrics(
        t_grid, qbar_i, vbar_i, mu, alpha_diag, eta_diag, liquidity, sigma_matrix
    )
    return {
        f"{prefix}_pop{pop}_permanent_impact_unit": float(metrics["permanent_impact_unit"][0]),
        f"{prefix}_pop{pop}_temporary_cost_unit": float(metrics["temporary_cost_unit"][0]),
        f"{prefix}_pop{pop}_risk_normalized": float(metrics["risk_normalized"]),
        f"{prefix}_pop{pop}_risk_exposure": float(metrics["risk_exposure"]),
        f"{prefix}_pop{pop}_risk_reference": float(metrics["risk_reference"]),
        f"{prefix}_pop{pop}_volume": float(metrics["volume"][0]),
        f"{prefix}_pop{pop}_final_inventory": float(qbar_i[-1, 0]),
        f"{prefix}_pop{pop}_abs_final_inventory": float(abs(qbar_i[-1, 0])),
    }


def solve_interaction(params: dict) -> dict:
    liquidity, eta_diag, alpha_diag, mathbb_v, impact_matrix, sigma_matrix = _market_objects()
    del liquidity, eta_diag, alpha_diag

    qbar0_list = [np.array([value], dtype=float) for value in params["qbar0"]]
    gamma_list = [float(value) for value in params["gamma"]]
    rho_list = [float(value) for value in params["rho_raw"]]
    a_terminal_list = [np.diag([float(value)]) for value in params["A_terminal"]]

    t_grid, qbar, vbar, info = solve_multi_population_fixed_point(
        qbar0_list,
        gamma_list,
        rho_list,
        mathbb_v,
        impact_matrix,
        a_terminal_list,
        sigma_matrix,
        horizon=HEATMAP_HORIZON,
        n_steps=HEATMAP_N_STEPS,
        relaxation_lambda=HEATMAP_RELAXATION_LAMBDA,
        epsilon=HEATMAP_EPSILON,
        max_iter=HEATMAP_MAX_ITER,
        require_convergence=True,
    )
    mu = compute_aggregated_flow(vbar, rho_list)

    row = {}
    for i in range(2):
        row.update(_metric_row(t_grid, qbar[i], vbar[i], mu, i + 1, "interaction"))
    row.update(
        interaction_n_iter=info.n_iter,
        interaction_err=info.error,
        interaction_converged=info.converged,
    )
    return {"t_grid": t_grid, "qbar": qbar, "vbar": vbar, "mu": mu, "row": row}


_REFERENCE_CACHE: dict[tuple, dict] = {}


def solve_homogeneous_reference_for_population(gamma_value, a_value, qbar0_value):
    """Solve the two-copy homogeneous reference used by the heatmap analysis."""

    key = (
        round(float(gamma_value), 14),
        round(float(a_value), 14),
        round(float(qbar0_value), 8),
    )
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]

    liquidity, eta_diag, alpha_diag, mathbb_v, impact_matrix, sigma_matrix = _market_objects()
    qbar0_pair = [np.array([qbar0_value], dtype=float) for _ in range(2)]
    gamma_pair = [float(gamma_value)] * 2
    a_pair = [np.diag([float(a_value)]) for _ in range(2)]
    rho_pair = [0.5, 0.5]

    t_grid, qbar, vbar, info = solve_multi_population_fixed_point(
        qbar0_pair,
        gamma_pair,
        rho_pair,
        mathbb_v,
        impact_matrix,
        a_pair,
        sigma_matrix,
        horizon=HEATMAP_HORIZON,
        n_steps=HEATMAP_N_STEPS,
        relaxation_lambda=HEATMAP_RELAXATION_LAMBDA,
        epsilon=HEATMAP_EPSILON,
        max_iter=HEATMAP_MAX_ITER,
        require_convergence=True,
    )
    mu = compute_aggregated_flow(vbar, rho_pair)
    metrics = compute_execution_metrics(
        t_grid, qbar[0], vbar[0], mu, alpha_diag, eta_diag, liquidity, sigma_matrix
    )

    result = {
        "info": info,
        "permanent_impact_unit": float(metrics["permanent_impact_unit"][0]),
        "temporary_cost_unit": float(metrics["temporary_cost_unit"][0]),
        "risk_normalized": float(metrics["risk_normalized"]),
        "risk_exposure": float(metrics["risk_exposure"]),
        "risk_reference": float(metrics["risk_reference"]),
        "volume": float(metrics["volume"][0]),
        "final_inventory": float(qbar[0, -1, 0]),
        "abs_final_inventory": float(abs(qbar[0, -1, 0])),
    }
    _REFERENCE_CACHE[key] = result
    return result


_REFERENCE_METRICS = (
    "permanent_impact_unit",
    "temporary_cost_unit",
    "risk_normalized",
    "risk_exposure",
    "risk_reference",
    "volume",
    "final_inventory",
    "abs_final_inventory",
)


def build_homogeneous_reference_row(params: dict) -> dict:
    row = {}
    for pop in (1, 2):
        i = pop - 1
        reference = solve_homogeneous_reference_for_population(
            params["gamma"][i], params["A_terminal"][i], params["qbar0"][i]
        )
        row.update(
            {f"reference_pop{pop}_{metric}": reference[metric] for metric in _REFERENCE_METRICS}
        )
        row.update(
            {
                f"reference_pop{pop}_n_iter": reference["info"].n_iter,
                f"reference_pop{pop}_err": reference["info"].error,
                f"reference_pop{pop}_converged": reference["info"].converged,
            }
        )
    return row


def _delta_metrics(interaction_row, reference_row):
    metrics = (
        "permanent_impact_unit",
        "temporary_cost_unit",
        "risk_normalized",
        "risk_exposure",
        "volume",
        "final_inventory",
        "abs_final_inventory",
    )
    out = {}
    for pop in (1, 2):
        for metric in metrics:
            inter = interaction_row.get(f"interaction_pop{pop}_{metric}", np.nan)
            ref = reference_row.get(f"reference_pop{pop}_{metric}", np.nan)
            out[f"delta_pop{pop}_{metric}"] = (
                inter - ref if np.isfinite(inter) and np.isfinite(ref) else np.nan
            )
    return out


def solve_one_cell(params: dict) -> dict:
    interaction_row = solve_interaction(params)["row"]
    reference_row = build_homogeneous_reference_row(params)
    rho = normalize_rho(params["rho_raw"])
    out = {**interaction_row, **reference_row, **_delta_metrics(interaction_row, reference_row)}
    out.update(
        gamma_1=float(params["gamma"][0]),
        gamma_2=float(params["gamma"][1]),
        A_1=float(params["A_terminal"][0]),
        A_2=float(params["A_terminal"][1]),
        rho_raw_1=float(params["rho_raw"][0]),
        rho_raw_2=float(params["rho_raw"][1]),
        rho_norm_1=float(rho[0]),
        rho_norm_2=float(rho[1]),
        qbar0_1=float(params["qbar0"][0]),
        qbar0_2=float(params["qbar0"][1]),
    )
    return out


def run_scenario_grid(scenario: HeatmapScenario) -> tuple[pd.DataFrame, list[float], list[float]]:
    y_values = geometric_grid(scenario.high, scenario.low)
    x_values = geometric_grid(scenario.high, scenario.low)
    rows = []

    for iy, pop1_value in enumerate(y_values):
        for ix, pop2_value in enumerate(x_values):
            params = make_params_for_cell(scenario, pop1_value, pop2_value)
            try:
                metrics = solve_one_cell(params)
                status, error = "ok", ""
            except Exception as exc:
                metrics, status, error = {}, "error", str(exc)

            rows.append(
                {
                    "scenario": scenario.key,
                    "scenario_title": scenario.title,
                    "parameter": scenario.parameter,
                    "pop1_value_y": pop1_value,
                    "pop2_value_x": pop2_value,
                    "iy": iy,
                    "ix": ix,
                    "status": status,
                    "error": error,
                    **metrics,
                }
            )

    return pd.DataFrame(rows), y_values, x_values


def matrix_from_results(df: pd.DataFrame, metric: str, y_values, x_values) -> np.ndarray:
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for _, row in df.iterrows():
        if metric in row.index and pd.notna(row[metric]):
            matrix[int(row["iy"]), int(row["ix"])] = float(row[metric])
    return matrix


def fmt_axis_value(x: float) -> str:
    x = float(x)
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 10_000:
        return f"{x:,.0f}".replace(",", " ")
    if ax >= 1_000:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    if ax >= 1e-3:
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return f"{x:.3e}"


def _custom_diverging_cmap():
    colors = [
        (0.00, "#8b0000"),
        (0.18, "#ff2d2d"),
        (0.36, "#b12bb1"),
        (0.50, "#0047ff"),
        (0.68, "#00bcd4"),
        (0.84, "#00c853"),
        (1.00, "#006d2c"),
    ]
    return LinearSegmentedColormap.from_list("custom_strong_diverging", colors, N=256)


def plot_heatmap_delta(matrix, y_values, x_values, scenario, spec, filename, annotate=False, show=True):

    plt.rcParams["svg.fonttype"] = "none"
    matrix = np.asarray(matrix, dtype=float)
    finite = matrix[np.isfinite(matrix)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0

    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    base_cmap = _custom_diverging_cmap()
    cmap = base_cmap if spec.positive_is_better else base_cmap.reversed(
        name="custom_strong_diverging_reversed"
    )

    fig, ax = plt.subplots(figsize=(11, 8), dpi=140)
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(
        f"{scenario.title} — {spec.title}\n"
        "Sous-population 1 : interaction hétérogène − scénario homogène de référence",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel(scenario.x_label, fontsize=11)
    ax.set_ylabel(scenario.y_label, fontsize=11)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels(
        [fmt_axis_value(v) for v in x_values], rotation=45, ha="right", fontsize=7
    )
    ax.set_yticklabels([fmt_axis_value(v) for v in y_values], fontsize=7)

    x0, y0 = np.meshgrid(np.arange(len(x_values)), np.arange(len(y_values)))
    ax.scatter(x0, y0, s=8, c="black", alpha=0.18)
    fig.colorbar(image, ax=ax).set_label(spec.cbar_label)

    if annotate and matrix.shape[0] <= 14 and matrix.shape[1] <= 14:
        for iy in range(matrix.shape[0]):
            for ix in range(matrix.shape[1]):
                if np.isfinite(matrix[iy, ix]):
                    ax.text(ix, iy, f"{matrix[iy, ix]:.3g}", ha="center", va="center", fontsize=6)

    ax.grid(False)
    fig.tight_layout()
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_matrix_csv(matrix, y_values, x_values, filename):
    df_matrix = pd.DataFrame(
        matrix,
        index=[fmt_axis_value(v) for v in y_values],
        columns=[fmt_axis_value(v) for v in x_values],
    )
    df_matrix.index.name = "sous_population_1_axe_y"
    df_matrix.columns.name = "sous_population_2_axe_x"
    df_matrix.to_csv(filename, encoding="utf-8-sig")


def diagonal_validation(df: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    diagonal = df[df["iy"] == df["ix"]]
    columns = (
        "delta_pop1_permanent_impact_unit",
        "delta_pop1_temporary_cost_unit",
        "delta_pop1_risk_normalized",
        "delta_pop1_abs_final_inventory",
    )
    checks = []
    for column in columns:
        values = pd.to_numeric(diagonal[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else np.nan
        checks.append(
            {
                "metric": column,
                "max_abs_diagonal": max_abs,
                "within_tolerance": bool(max_abs <= tolerance) if np.isfinite(max_abs) else False,
            }
        )
    return pd.DataFrame(checks)


def regenerate_figures_from_csv(raw_dir: str | Path, figure_dir: str | Path, *, show: bool = True):
    """Regenerate all heatmap SVGs from existing raw CSVs without model recomputation."""

    raw_dir = Path(raw_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for scenario in HEATMAP_SCENARIOS:
        raw_csv = raw_dir / f"{scenario.key}_interaction_vs_reference_homogene_resultats_complets.csv"
        if not raw_csv.exists():
            raise FileNotFoundError(f"Missing raw heatmap CSV: {raw_csv}")

        df = pd.read_csv(raw_csv)

        # Recalcule explicitement la distance à zéro de l'inventaire terminal.
        for pop in (1, 2):
            interaction_q_col = f"interaction_pop{pop}_final_inventory"
            reference_q_col = f"reference_pop{pop}_final_inventory"

            interaction_abs_col = (
                f"interaction_pop{pop}_abs_final_inventory"
            )
            reference_abs_col = (
                f"reference_pop{pop}_abs_final_inventory"
            )
            delta_abs_col = (
                f"delta_pop{pop}_abs_final_inventory"
            )

            df[interaction_abs_col] = (
                pd.to_numeric(
                    df[interaction_q_col],
                    errors="coerce",
                ).abs()
            )

            df[reference_abs_col] = (
                pd.to_numeric(
                    df[reference_q_col],
                    errors="coerce",
                ).abs()
            )

            df[delta_abs_col] = (
                df[interaction_abs_col]
                - df[reference_abs_col]
            )

        y_values = (
            df[["iy", "pop1_value_y"]]
            .drop_duplicates("iy")
            .sort_values("iy")["pop1_value_y"]
            .astype(float)
            .tolist()
        )
        x_values = (
            df[["ix", "pop2_value_x"]]
            .drop_duplicates("ix")
            .sort_values("ix")["pop2_value_x"]
            .astype(float)
            .tolist()
        )

        for spec in HEATMAP_SPECS:
            matrix = matrix_from_results(df, spec.result_column, y_values, x_values)
            matrix_csv = figure_dir / f"{scenario.key}_heatmap_delta_{spec.key}.csv"
            matrix_svg = figure_dir / f"{scenario.key}_heatmap_delta_{spec.key}.svg"
            save_matrix_csv(matrix, y_values, x_values, matrix_csv)
            plot_heatmap_delta(
                matrix,
                y_values,
                x_values,
                scenario,
                spec,
                matrix_svg,
                annotate=False,
                show=show,
            )
