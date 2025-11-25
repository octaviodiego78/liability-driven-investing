import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("inference/output/data")
OUTPUT_DIR = Path("inference/output/final_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

csv_files = {
    "PPO continuous":       DATA_DIR / "ppo_continuous_real_results.csv",
    "FCNN w/ const":        DATA_DIR / "fcnn_with_constraints_real_results.csv",
    "FCNN w/o const":       DATA_DIR / "fcnn_without_constraints_real_results.csv",
    "LSTM w/ const":        DATA_DIR / "lstm_with_constraints_real_results.csv",
    "LSTM w/o const":       DATA_DIR / "lstm_without_constraints_real_results.csv",
    "PPO wide discrete":    DATA_DIR / "ppo_wide_discrete_real_results.csv",
    "A2C":                  DATA_DIR / "a2c_real_results.csv",
}

RISK_FREE_RATE_ANUAL = 0.04

results = []

for name, path in csv_files.items():
    if not path.exists():
        continue

    df = pd.read_csv(path)
    mean_surplus = df.groupby('period')['funding_surplus'].mean() / 1_000_000

    mean_surplus = mean_surplus[mean_surplus > 0]
    if len(mean_surplus) < 2:
        continue

    returns = mean_surplus.pct_change().dropna()

    if len(returns) < 2:
        continue

    n_years = len(returns) / 4.0
    # CAGR
    total_return = (1 + returns).prod()
    cagr = total_return ** (1 / n_years) - 1

    # Max Drawdown
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    max_dd_pct = ((equity - peak) / peak).min() * 100

    # Volatilidad anualizada
    vol_annual = returns.std(ddof=1) * np.sqrt(4)

    # Sharpe Ratio
    RISK_FREE_RATE_ANUAL = 0.04
    rf_quarterly = (1 + RISK_FREE_RATE_ANUAL)**(1/4) - 1
    excess_return_quarterly = returns.mean() - rf_quarterly
    sharpe = (excess_return_quarterly * 4) / vol_annual if vol_annual > 1e-8 else 0.0

    # Sortino Ratio
    downside = np.minimum(returns - rf_quarterly, 0)
    downside_deviation = np.sqrt(np.mean(downside**2))
    downside_dev_annual = downside_deviation * np.sqrt(4)
    sortino = (cagr - RISK_FREE_RATE_ANUAL) / downside_dev_annual if downside_dev_annual > 1e-8 else 0.0

    # Calmar Ratio
    calmar = cagr / abs(max_dd_pct / 100) if abs(max_dd_pct) > 1e-8 else 999.0

    # Funding ratio final
    final_fr = df[df["period"] == df["period"].max()]["funding_ratio"].mean()

    results.append({
        "Model": name,
        "Annual Return (%)": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Max DD (%)": round(max_dd_pct, 1),
        "Final FR": round(final_fr, 4),
    })

# TABLE
df_final = pd.DataFrame(results).sort_values('Final FR', ascending=False).reset_index(drop=True)

print("\n" + "="*100)
print("RISK METRICS FOR LDI".center(100))
print("="*100)
print(df_final.to_string(index=False))
print("="*100)

df_final.to_csv(OUTPUT_DIR / "TABLE_FINAL.csv", index=False)

# GRAPHICS
sns.set_style("whitegrid")
for metric in ['Sharpe', 'Sortino', 'Calmar', 'Final FR']:
    plt.figure(figsize=(11,7))
    bars = plt.bar(df_final['Model'], df_final[metric], color='teal', alpha=0.8, edgecolor='black')
    plt.title(f'{metric} - Backtest Real 2015-2025', fontsize=16, weight='bold')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + h*0.02, f'{h:.3f}', ha='center', va='bottom', weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{metric}.png", dpi=300)
    plt.show()