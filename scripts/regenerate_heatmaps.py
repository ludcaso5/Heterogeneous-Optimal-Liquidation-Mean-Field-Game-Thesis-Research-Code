"""Regenerate heatmap SVGs from existing raw CSVs without recomputing the model."""

from pathlib import Path

from mfg_liquidation.heatmaps import regenerate_figures_from_csv

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    regenerate_figures_from_csv(
        ROOT / "results" / "heatmaps" / "raw",
        ROOT / "results" / "heatmaps" / "figures",
        show=False,
    )
