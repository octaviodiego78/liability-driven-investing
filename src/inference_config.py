"""
Utilities for running inference with real economic and liability inputs.

This module keeps the standard training configuration intact while allowing
`inference.py` to temporarily swap in alternative datasets that live under
`inference/inputs/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config

INFERENCE_ROOT = Path(__file__).parent.parent / "inference"
INFERENCE_INPUT_DIR = INFERENCE_ROOT / "inputs"
# Expected inputs within INFERENCE_INPUT_DIR:
# - real_test_states.csv : Economic scenarios (41 rows x 52 columns per scenario)
# - real_liab.csv        : Liability projections (40 rows x 5 columns per scenario)
DEFAULT_ECONOMIC_PATH = INFERENCE_INPUT_DIR / "real_test_states.csv"
DEFAULT_LIABILITY_PATH = Path(__file__).parent.parent / "input" / "liab.csv"


def apply_inference_inputs(
    *,
    economic_path: Optional[Path] = None,
    liability_path: Optional[Path] = None,
) -> Tuple[Tuple[Path, Path], Tuple[int, int]]:
    """
    Replace the shared config datasets with real inference inputs.

    This function mutates the global configuration module so that subsequent
    imports (including model classes) see the real data instead of the
    simulation datasets bundled with the project.
    """
    INFERENCE_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    econ_path = Path(economic_path) if economic_path else DEFAULT_ECONOMIC_PATH
    liab_path = Path(liability_path) if liability_path else DEFAULT_LIABILITY_PATH

    if not econ_path.exists():
        raise FileNotFoundError(f"Required inference input not found: {econ_path}")
    if not liab_path.exists():
        raise FileNotFoundError(f"Required inference input not found: {liab_path}")

    tmp_arrays = np.array(pd.read_csv(econ_path, header=None))
    liab_all = np.array(pd.read_csv(liab_path, header=None))

    periods = config.n_dyn
    sims_from_macro = tmp_arrays.shape[0] // (periods + 1)
    sims_from_liab = liab_all.shape[0] // periods if periods else 0
    simulations = min(sims_from_macro, sims_from_liab) if periods else 0


    config.tmpArrays = tmp_arrays
    config.liabAll = liab_all
    config.n_sim = simulations

    paths = (econ_path.resolve(), liab_path.resolve())
    dimensions = (simulations, periods)
    return paths, dimensions


def prepare_output_dirs() -> dict[str, Path]:
    """
    Create and return the output directories used during inference runs.
    """
    data_dir = INFERENCE_ROOT / "output" / "data"
    figs_dir = INFERENCE_ROOT / "output" / "figs"
    models_dir = INFERENCE_ROOT / "output" / "models"

    for directory in (data_dir, figs_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "data": data_dir,
        "figs": figs_dir,
        "models": models_dir,
    }