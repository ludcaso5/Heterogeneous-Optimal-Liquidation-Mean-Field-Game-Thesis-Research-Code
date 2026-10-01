# Heterogeneous Mean-Field Optimal Liquidation

Research code accompanying the master's thesis **_Interactions stratégiques et liquidation optimale : un modèle de jeux à champ moyen avec population hétérogène_** (HEC Montréal, 2026).

The repository implements the deterministic mean-field optimal-liquidation model used in the thesis, with discrete heterogeneous sub-populations, permanent market impact, quadratic temporary execution cost, inventory risk and terminal-inventory penalties.

## Repository scope

- `notebooks/thesis_scenarios.ipynb` reproduces Figures **3, 5, 7, 9, 10 and 11** and Tables **2 to 7**.
- `notebooks/heatmaps.ipynb` reproduces the sensitivity heatmaps associated with Figures **4, 6 and 8**.
- `notebooks/robustness.ipynb` reproduces the local numerical-stability analysis in Section **5.9 / Table 8**.

## Model implemented

For population `k`, the mean inventory and mean trading speed follow the discretization of

```text
q̄'_k = v̄_k
v̄'_k = 2 γ_k 𝕍 Σ q̄_k - 2 𝕍 Λ μ
μ = Σ_k ρ_k v̄_k
```

with terminal condition

```text
v̄_k(T) + 4 𝕍 A_k q̄_k(T) = 0,
```

where `𝕍 = diag(𝒱 / (4η))`. The fixed point is solved on the aggregate flow `μ`.

The diagnostics used in the thesis are:

1. terminal mean inventory;
2. total traded volume;
3. unit permanent-impact index;
4. unit temporary execution cost;
5. normalized price-risk exposure.

## Installation

From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the project and notebook dependencies:

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra explicitly installs Notebook, IPython and ipykernel in addition to the scientific dependencies. The notebooks also contain a local `src/` fallback so they can resolve `mfg_liquidation` when launched from inside the repository.

For exact reproduction with the versions used to validate this release, see `requirements-lock.txt`.

## Reproduce the thesis scenarios

Open:

```bash
notebooks/thesis_scenarios.ipynb
```

Run the notebook from top to bottom. The generated SVG files are saved in `results/figures/`; the formatted numerical tables are saved in `results/tables/`.

The numerical notebook uses exactly the Chapter-5 parameters

```text
T = 1
N = 2000
𝒱 = 2,000,000
η = 0.1
σ = 0.3
α = 8e-8
λ = 0.35
ε = 1e-8
```

and the scenario-specific values reported in the thesis.

## Heatmaps

Open:

```bash
notebooks/heatmaps.ipynb
```

The notebook separates the expensive model calculation from figure regeneration. After the raw CSV files have been computed once, figures can be regenerated without recalculating the model:

```bash
python scripts/regenerate_heatmaps.py
```

The regeneration path is read-only with respect to the raw result CSV files. 

## Numerical robustness

Open:

```bash
notebooks/robustness.ipynb
```

Run the notebook from top to bottom. It displays the three stability tables  and saves raw CSV files in `results/tables/`. The `ε = 1e-10` configuration is intentionally allowed to reach the iteration limit because Section 5.9 reports that this case does not converge after 100,000 iterations.


## Code layout

```text
src/mfg_liquidation/
    solver.py       sparse system and fixed-point solver
    metrics.py      execution diagnostics
    references.py   matched homogeneous references
    parameters.py   exact Chapter-5 scenario definitions
    experiments.py  high-level scenario runner
    plotting.py     thesis figure styling and SVG export
    reporting.py    French numerical formatting and thesis tables
    heatmaps.py     heatmap-specific analysis and plotting

notebooks/
    thesis_scenarios.ipynb
    heatmaps.ipynb
    robustness.ipynb

scripts/
    regenerate_heatmaps.py

```

## Reproducibility policy

The numerical core is defined in one place and imported by all experiments. Plotting, Word formatting and notebook presentation are kept outside the solver. Raw heatmap CSV files are treated as immutable inputs when figures are regenerated. A result that fails the requested fixed-point tolerance is not silently accepted as a valid scenario, except in the explicit robustness experiment where non-convergence is itself the reported result.

## Citation

Citation metadata are provided in `CITATION.cff`. 

## License

Copyright © 2026 Ludwig Casaubon. All rights reserved.

The source code is made available under the terms of the included `LICENSE` for academic inspection and non-commercial reproducibility. It is not released under an open-source license. Modification, redistribution, sublicensing and commercial use require prior written permission from the copyright holder.