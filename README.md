# Liability Driven Investing (LDI) with Reinforcement Learning

This project implements various reinforcement learning approaches for Liability Driven Investing (LDI) strategies. It includes implementations of LSTM and FCNN models, both with and without investment constraints.

<img width="1901" height="954" alt="image" src="https://github.com/user-attachments/assets/76a1744b-3226-494e-a3ea-8287ca427a9c" />

## Project Structure

```
liability-driven-investing/
├── src/
│   ├── config.py            # Global variables and paths
│   ├── utils.py             # Helper functions
│   ├── models/
│   │   ├── base_model.py    # Common model components
│   │   ├── fcnn_c.py        # FCNN with constraints
│   │   ├── fcnn_nc.py       # FCNN without constraints  
│   │   ├── lstm_c.py        # LSTM with constraints
│   │   └── lstm_nc.py       # LSTM without constraints
│   └── main.py              # Main execution script
├── input/                   # Input data files
└── output/                  # Output files and results
    ├── figs/                # Models performance comparission
    └── data/                # CSV results

```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib
- Plotly

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script runs all four models automatically without requiring any arguments:

```bash
python src/main.py
```

This will train and evaluate all four model configurations:
- FCNN with Constraints
- FCNN without Constraints  
- LSTM with Constraints
- LSTM without Constraints

### Model Types

1. **FCNN with Constraints**
   - Fixed step size for allocation changes
   - Limited to ±2% changes in allocation
   - 3 possible actions: keep current, increase bonds, decrease bonds

2. **FCNN without Constraints**
   - Flexible allocation changes
   - 51 possible allocation combinations (0% to 100% bonds in 2% increments)

3. **LSTM with Constraints**
   - Temporal dependency modeling with LSTM architecture
   - Fixed step size for allocation changes
   - 3 possible actions: keep current, increase bonds, decrease bonds

4. **LSTM without Constraints**
   - Temporal dependency modeling with LSTM architecture
   - Flexible allocation changes
   - 51 possible allocation combinations (0% to 100% bonds in 2% increments)

## Configuration

Model parameters and experiment settings can be modified in `src/config.py`:

- Batch size
- Learning rates
- Epsilon values for exploration
- Network architectures
- Training episodes
- Simulation parameters

## Input Data

Required input files in the `input/` directory:
- var.csv: VAR model parameters
- mapping.csv: Asset mapping data
- mortality.csv: Mortality tables
- census.csv: Plan participant data
- histMF.csv: Historical macro factors
- histAR.csv: Historical asset returns
- Various other configuration files

## Output

Results are automatically organized in the `output/` directory:

```
output/
├── data/                           # CSV files with model results
│   ├── fcnn_with_constraints_results.csv
│   ├── fcnn_without_constraints_results.csv
│   ├── lstm_with_constraints_results.csv
│   └── lstm_without_constraints_results.csv
└── figs/                           # HTML interactive plots
    ├── AA_rated_bond_investment_comparison.html
    ├── public_equity_investment_comparison.html
    ├── total_asset_value_comparison.html
    ├── funding_ratio_comparison.html
    └── funding_surplus_comparison.html
```

Each CSV file contains:
- Asset allocation histories
- Funding ratio trajectories
- Investment returns
- Liability projections
- Performance metrics over time

The HTML plots provide interactive visualizations comparing all four models across different performance metrics.

## License

[Include your license information here]
