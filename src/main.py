"""
Main script for running LDI experiments.
"""

from pathlib import Path
from models.fcnn_c import FCNNWithConstraint
from models.fcnn_nc import FCNNWithoutConstraint
from models.lstm_c import LSTMWithConstraint
from models.lstm_nc import LSTMWithoutConstraint
from utils import load_data, run_experiment, save_results, plot_model_comparison

def main():
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Run experiments
    models = {
        'fcnn_with_constraints': FCNNWithConstraint,
        'fcnn_without_constraints': FCNNWithoutConstraint,
        'lstm_with_constraints': LSTMWithConstraint,
        'lstm_without_constraints': LSTMWithoutConstraint
    }
    
    results = {}
    for model_name, model_class in models.items():
        print(f"\nRunning {model_name}...")
        model_results = run_experiment(model_class, data, num_episodes=100, num_sims=800)
        results_df = save_results(model_results, model_name, output_dir)
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
        plot_model_comparison(results, column, output_dir)
    
    print("\nAll experiments completed successfully!")

if __name__ == '__main__':
    main()