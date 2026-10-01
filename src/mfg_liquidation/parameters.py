"""Parameters for the numerical experiments reported in Chapter 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

HORIZON = 1.0
N_STEPS = 2000
LIQUIDITY = 2_000_000.0
ETA = 0.1
SIGMA = 0.3
ALPHA = 8e-8
RELAXATION_LAMBDA = 0.35
EPSILON = 1e-8
MAX_ITER = 1000

GAMMA_LOW = 5e-7
GAMMA_BASE = 5e-6
GAMMA_HIGH = 5e-5

A_LOW = 1e-9
A_BASE = 1e-7
A_HIGH = 5e-6


@dataclass(frozen=True)
class ThesisScenario:
    """Definition of one numerical scenario reproduced from the thesis."""

    key: str
    section: str
    title: str
    qbar0: Sequence[float]
    gamma: Sequence[float]
    a_terminal: Sequence[float]
    rho: Sequence[float]
    compare_to_homogeneous_reference: bool
    figure_number: str
    table_number: str


THESIS_SCENARIOS = (
    ThesisScenario(
        key="risk_aversion",
        section="5.3",
        title="Scénario d’aversion au risque",
        qbar0=(100_000.0, 100_000.0),
        gamma=(GAMMA_LOW, GAMMA_HIGH),
        a_terminal=(A_BASE, A_BASE),
        rho=(0.5, 0.5),
        compare_to_homogeneous_reference=True,
        figure_number="3",
        table_number="2",
    ),
    ThesisScenario(
        key="terminal_penalty",
        section="5.4",
        title="Scénario de pénalisation de l’inventaire terminal",
        qbar0=(100_000.0, 100_000.0),
        gamma=(GAMMA_BASE, GAMMA_BASE),
        a_terminal=(A_LOW, A_HIGH),
        rho=(0.5, 0.5),
        compare_to_homogeneous_reference=True,
        figure_number="5",
        table_number="3",
    ),
    ThesisScenario(
        key="initial_inventory",
        section="5.5",
        title="Scénario d’inventaire initial",
        qbar0=(100_000.0, 50_000.0),
        gamma=(GAMMA_BASE, GAMMA_BASE),
        a_terminal=(A_BASE, A_BASE),
        rho=(0.5, 0.5),
        compare_to_homogeneous_reference=True,
        figure_number="7",
        table_number="4",
    ),
    ThesisScenario(
        key="opportunism_equal_masses",
        section="5.6",
        title="Scénario taille des sous-populations et opportunisme — tailles identiques",
        qbar0=(100_000.0, 0.0),
        gamma=(GAMMA_HIGH, 0.0),
        a_terminal=(A_HIGH, A_HIGH),
        rho=(0.5, 0.5),
        compare_to_homogeneous_reference=True,
        figure_number="9",
        table_number="5",
    ),
    ThesisScenario(
        key="opportunism_95_5",
        section="5.6",
        title="Scénario taille des sous-populations et opportunisme — tailles différentes",
        qbar0=(100_000.0, 0.0),
        gamma=(GAMMA_HIGH, 0.0),
        a_terminal=(A_HIGH, A_HIGH),
        rho=(0.95, 0.05),
        compare_to_homogeneous_reference=True,
        figure_number="10",
        table_number="6",
    ),
    ThesisScenario(
        key="four_populations_gamma_A",
        section="5.7",
        title="Scénario de combinaisons d’aversion au risque et de pénalisation terminale",
        qbar0=(100_000.0, 100_000.0, 100_000.0, 100_000.0),
        gamma=(GAMMA_BASE, GAMMA_LOW, GAMMA_LOW, GAMMA_BASE),
        a_terminal=(A_LOW, A_HIGH, A_LOW, A_HIGH),
        rho=(0.25, 0.25, 0.25, 0.25),
        compare_to_homogeneous_reference=True,
        figure_number="11",
        table_number="7",
    ),

)

SCENARIOS_BY_KEY = {scenario.key: scenario for scenario in THESIS_SCENARIOS}
