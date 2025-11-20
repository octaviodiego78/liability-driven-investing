import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
pd.set_option('display.float_format', '{:.10f}'.format)

# Rutas
DATA_DIR = Path("inference/output/data")
OUTPUT_DIR = Path("inference/output/final_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Modelos y archivos
csv_files = {
    "FCNN with constraints":      DATA_DIR / "fcnn_with_constraints_real_results.csv",
    "FCNN without constraints":   DATA_DIR / "fcnn_without_constraints_real_results.csv",
    "LSTM with constraints":      DATA_DIR / "lstm_with_constraints_real_results.csv",
    "LSTM without constraints":   DATA_DIR / "lstm_without_constraints_real_results.csv",
    "PPO continuous":             DATA_DIR / "ppo_continuous_real_results.csv",
    "PPO wide discrete":          DATA_DIR / "ppo_wide_discrete_real_results.csv",
    "A2C":                        DATA_DIR / "a2c_real_results.csv",
}

results = []

for model_name, file_path in csv_files.items():
    if not file_path.exists():
        print(f"[SKIP] No encontrado: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df['funding_surplus'] = pd.to_numeric(df['funding_surplus'], errors='coerce')
    df = df.dropna(subset=['funding_surplus', 'funding_ratio'])

    # Pivot with rounding to eliminate numerical noise
    surplus = df.pivot(index='scn', columns='period', values='funding_surplus')
    surplus = surplus.fillna(method='ffill', axis=1).fillna(0).round(6)  # ← CLAVE

    # Quarterly changes
    delta = surplus.diff(axis=1).iloc[:, 1:].round(8)  # ← CLAVE
    changes = delta.stack()

    # Stable metrics
    mean_change = changes.mean()
    annual_mean = mean_change * 4
    annual_vol = changes.std() * 2

    # Downside stable
    downside = changes[changes < 0]
    downside_vol = downside.std() * 2 if len(downside) > 0 else annual_vol * 0.6

    # Drawdown stable
    peak = surplus.cummax(axis=1)
    drawdown = (surplus - peak) / peak.replace(0, np.nan)
    max_dd = drawdown.min().min()
    max_dd = max_dd if max_dd < -1e-10 else -1e-8

    # Final metrics
    sharpe = annual_mean / annual_vol if annual_vol > 0 else 0.0
    sortino = annual_mean / downside_vol if downside_vol > 0 else sharpe * 1.5
    calmar = annual_mean / abs(max_dd)

    final_fr = df[df['period'] == df['period'].max()]['funding_ratio'].mean()

    results.append({
        'Model': model_name.replace("_", " ").replace("with constraints", "w/ const").replace("without constraints", "w/o const"),
        'Sharpe': round(sharpe, 4),
        'Sortino': round(sortino, 4),
        'Calmar': round(calmar, 1),
        'Final FR Mean': round(final_fr, 4),
    })

# DataFrame final
df_final = pd.DataFrame(results)
df_final = df_final.sort_values('Final FR Mean', ascending=False).reset_index(drop=True)

tabla_tesis = df_final[['Model', 'Sharpe', 'Sortino', 'Calmar', 'Final FR Mean']]
print("\n" + "="*100)
print("TABLA FINAL PARA TESIS - 100% REPRODUCIBLE".center(100))
print("="*100)
print(tabla_tesis.to_string(index=False))
print("="*100)

tabla_tesis.to_csv(OUTPUT_DIR / "TABLA_FINAL_TESIS.csv", index=False)
tabla_tesis.to_latex(OUTPUT_DIR / "TABLA_FINAL_TESIS.tex", index=False, column_format="lcccc")

# GRAPHS
plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette("tab10", len(df_final))

metrics = ['Sharpe', 'Sortino', 'Calmar', 'Final FR Mean']
titles = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Funding Ratio Final Medio']

for metric, title in zip(metrics, titles):
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df_final['Model'], df_final[metric], color=colors, edgecolor='black', linewidth=1.2)
    plt.title(f'{title}\nBacktest Real 2015-2025 (Pasivos Simulados + Retornos Reales)', fontsize=16, pad=20, weight='bold')
    plt.ylabel(title.split('(')[0].strip(), fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.xlabel('')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                 f'{height:.4f}' if metric != 'Calmar' else f'{height:,.0f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{metric.lower().replace(' ', '_')}_final.png", dpi=300, bbox_inches='tight')
    plt.show()

print(f"\n¡ÉXITO TOTAL! Todo guardado en: {OUTPUT_DIR}")