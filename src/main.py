"""
Main script for running LDI experiments.
"""

import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd

# Set random seeds to match original code
random.seed(6)
np.random.seed(6)

from pathlib import Path
from models.fcnn_c import FCNNWithConstraint
from models.fcnn_nc import FCNNWithoutConstraint
from models.lstm_c import LSTMWithConstraint
from models.lstm_nc import LSTMWithoutConstraint
from models.ppo_model import PPOModel, PPOContinuousModel, PPOWideDiscreteModel
from models.a2c_model import A2CModel
from utils import load_data, run_experiment, save_results, plot_model_comparison

def main():
    # Create output directories
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    data_dir = output_dir / 'data'
    figs_dir = output_dir / 'figs'
    models_dir = output_dir / 'models'
    data_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Run experiments
    models = {
        'fcnn_with_constraints': FCNNWithConstraint,
        'fcnn_without_constraints': FCNNWithoutConstraint,
        'lstm_with_constraints': LSTMWithConstraint,
        'lstm_without_constraints': LSTMWithoutConstraint,
        'ppo': PPOModel,
        'ppo_continuous': PPOContinuousModel,
        'ppo_wide_discrete': PPOWideDiscreteModel,
        'a2c': A2CModel
    }
    
    results = {}
    for model_name, model_class in models.items():
        results_path = data_dir / f'{model_name}_results.csv'
        
        #if results_path.exists():
        #    print(f"Skipping {model_name}: existing results found at '{results_path}'.")
        #    results_df = pd.read_csv(results_path)
        #else:
        print(f"Running {model_name}...")
        model_results = run_experiment(model_class, data, num_episodes=5000, num_sims=900)
        results_df = save_results(model_results, model_name, data_dir)
        
        results[model_name] = results_df
    
    # Create comparison plots
    plot_columns = [
        'AA_rated_bond_investment',
        'public_equity_investment',
        'total_asset_value',
        'funding_ratio',
        'funding_surplus'
    ]
    
    for column in plot_columns:
        plot_model_comparison(results, column, figs_dir)
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print("="*60)

if __name__ == '__main__':
    main()