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

results = []

for name, path in csv_files.items():
    if not path.exists():
        continue

    df = pd.read_csv(path)

    # TRAYECTORIA MEDIA
    mean_surplus = df.groupby('period')['funding_surplus'].mean()

    # MAX DRAWDOWN
    peak = mean_surplus.cummax()
    drawdown = (mean_surplus - peak) / peak
    max_dd = drawdown.min()

    delta = mean_surplus.diff().dropna()
    annual_mean = delta.mean() * 4          # crecimiento medio anual del surplus (M$)
    annual_vol = delta.std() * 2            # volatilidad anualizada (√4 = 2)

    # Downside deviation
    downside = delta[delta < 0]
    downside_vol = downside.std() * 2 if len(downside) > 0 else annual_vol

    # Metrics
    sharpe  = annual_mean / annual_vol if annual_vol > 0 else 0
    sortino = annual_mean / downside_vol if downside_vol > 0 else 0
    calmar  = annual_mean / abs(max_dd * mean_surplus.max()) if abs(max_dd) > 1e-6 else 999

    final_fr = df[df['period'] == df['period'].max()]['funding_ratio'].mean()

    results.append({
        'Model': name,
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'Calmar': round(calmar, 2),
        'Max DD (%)': round(max_dd * 100, 1),
        'Final FR': round(final_fr, 4),
    })

# TABLE
df_final = pd.DataFrame(results).sort_values('Final FR', ascending=False).reset_index(drop=True)

print("\n" + "="*100)
print("MÉTRICAS FINALES CORRECTAS PARA LDI - CAMBIO ABSOLUTO DEL SURPLUS".center(100))
print("="*100)
print(df_final.to_string(index=False))
print("="*100)

df_final.to_csv(OUTPUT_DIR / "TABLA_TESIS_FINAL.csv", index=False)

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
    plt.savefig(OUTPUT_DIR / f"{metric}_TESIS.png", dpi=300)
    plt.show()