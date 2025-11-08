"""
Run trained models against real-world economic and liability scenarios.

Usage:
    python src/inference.py

Real data should be placed under ``inference/inputs/``. See
``inference_config.py`` for the expected filenames.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch

from inference_config import apply_inference_inputs, prepare_output_dirs, INFERENCE_INPUT_DIR
from models.fcnn_c import FCNNWithConstraint
from models.fcnn_nc import FCNNWithoutConstraint
from models.lstm_c import LSTMWithConstraint
from models.lstm_nc import LSTMWithoutConstraint
from models.ppo_model import PPOModel, PPOContinuousModel, PPOWideDiscreteModel
from models.a2c_model import A2CModel
from utils import save_results, plot_model_comparison

# Suppress noisy warnings from the various numerical libraries.
warnings.filterwarnings("ignore")

OUTPUT_COLUMNS = [
    "AA_rated_bond_investment",
    "public_equity_investment",
    "total_asset_value",
    "funding_ratio",
    "funding_surplus",
]


def _load_torch_checkpoint(model_instance: Any, checkpoint_path: Path) -> None:
    """Load a PyTorch state dictionary into the model wrapper."""
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model_instance.model.load_state_dict(state)
    model_instance.static_model.load_state_dict(model_instance.model.state_dict())
    model_instance.model.eval()
    model_instance.static_model.eval()


def _load_sb_checkpoint(model_instance: Any, checkpoint_path: Path, algorithm: str) -> None:
    """Load a Stable-Baselines checkpoint into the wrapper."""
    if algorithm.lower() == "ppo":
        from stable_baselines3 import PPO

        model_instance.model = PPO.load(str(checkpoint_path), env=model_instance.env, device="cpu")
    elif algorithm.lower() == "a2c":
        from stable_baselines3 import A2C

        model_instance.model = A2C.load(str(checkpoint_path), env=model_instance.env, device="cpu")
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}' for checkpoint loading.")


def _evaluate_model(
    name: str,
    model_factory: Callable[[], Any],
    checkpoint_path: Path,
    loader: Callable[[Any, Path], None],
    simulations: int,
    data_dir: Path,
) -> Tuple[str, Any]:
    """Instantiate, load and evaluate a model."""
    if not checkpoint_path.exists():
        print(f"[WARN] Skipping {name}: checkpoint not found at {checkpoint_path}")
        return name, None

    model = model_factory()
    loader(model, checkpoint_path)

    print(f"[INFO] Evaluating {name} on {simulations} simulations...")
    evaluation = model.evaluate(simulations)
    tliab_array = evaluation[0]
    results_df = save_results(tliab_array, f"{name}_real", data_dir)
    return name, results_df


def main() -> None:
    paths, dims = apply_inference_inputs()
    outputs = prepare_output_dirs()
    economic_path, liability_path = paths
    simulations, periods = dims

    print("============================================================")
    print(" Real Data Inference")
    print("------------------------------------------------------------")
    print(f" Economic input : {economic_path}")
    print(f" Liability input: {liability_path}")
    print(f" Simulations    : {simulations}")
    print(f" Periods        : {periods}")
    print(f" Output (data)  : {outputs['data']}")
    print(f" Output (figs)  : {outputs['figs']}")
    print("============================================================\n")

    model_specs: Dict[str, Dict[str, Any]] = {
        "fcnn_with_constraints": {
            "factory": FCNNWithConstraint,
            "checkpoint": Path("output/models/fcnn_with_constraints_model.pth"),
            "loader": lambda m, p: _load_torch_checkpoint(m, p),
        },
        "fcnn_without_constraints": {
            "factory": FCNNWithoutConstraint,
            "checkpoint": Path("output/models/fcnn_without_constraints_model.pth"),
            "loader": lambda m, p: _load_torch_checkpoint(m, p),
        },
        "lstm_with_constraints": {
            "factory": LSTMWithConstraint,
            "checkpoint": Path("output/models/lstm_with_constraints_model.pth"),
            "loader": lambda m, p: _load_torch_checkpoint(m, p),
        },
        "lstm_without_constraints": {
            "factory": LSTMWithoutConstraint,
            "checkpoint": Path("output/models/lstm_without_constraints_model.pth"),
            "loader": lambda m, p: _load_torch_checkpoint(m, p),
        },
        "ppo": {
            "factory": PPOModel,
            "checkpoint": Path("output/models/ppo_ldi_model.zip"),
            "loader": lambda m, p: _load_sb_checkpoint(m, p, "ppo"),
        },
        "ppo_continuous": {
            "factory": PPOContinuousModel,
            "checkpoint": Path("output/models/ppo_continuous_ldi_model.zip"),
            "loader": lambda m, p: _load_sb_checkpoint(m, p, "ppo"),
        },
        "ppo_wide_discrete": {
            "factory": PPOWideDiscreteModel,
            "checkpoint": Path("output/models/ppo_wide_discrete_ldi_model.zip"),
            "loader": lambda m, p: _load_sb_checkpoint(m, p, "ppo"),
        },
        "a2c": {
            "factory": A2CModel,
            "checkpoint": Path("output/models/a2c_ldi_model.zip"),
            "loader": lambda m, p: _load_sb_checkpoint(m, p, "a2c"),
        },
    }

    results: Dict[str, Any] = {}
    for name, spec in model_specs.items():
        model_name, df = _evaluate_model(
            name=name,
            model_factory=spec["factory"],
            checkpoint_path=spec["checkpoint"],
            loader=spec["loader"],
            simulations=simulations,
            data_dir=outputs["data"],
        )
        if df is not None:
            results[model_name] = df

    if not results:
        print("[WARN] No inference results were generated. Verify checkpoint files.")
        return

    print("\n[INFO] Generating comparison plots...")
    for column in OUTPUT_COLUMNS:
        plot_model_comparison(results, column, outputs["figs"])

    print("\n============================================================")
    print(" Inference complete. Review outputs under inference/output/.")
    print("============================================================")


if __name__ == "__main__":
    if not INFERENCE_INPUT_DIR.exists():
        print(f"[WARN] Expected input directory does not exist: {INFERENCE_INPUT_DIR}")
    main()

