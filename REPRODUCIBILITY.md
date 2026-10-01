# Reproducibility notes

## Raw data versus presentation

Heatmap raw result CSV files are written only by the full calculation workflow. Figure regeneration reads those CSV files and writes only derived matrix CSV/SVG outputs. It does not convert or overwrite historical raw data.

## Convergence

For ordinary scenario and heatmap calculations, non-convergence raises an error rather than producing a valid-looking result. The robustness notebook is the only exception: it records the convergence status because the `ε = 10^-10` experiment is explicitly intended to document the failure to converge within 100,000 iterations.
