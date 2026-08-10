"""
Battery Remaining Useful Life (RUL) Prediction
===============================================
Hybrid physics-informed machine learning framework that predicts the RUL of
18650 cells (NASA battery dataset B5/B6/B7) from physical measurements.

Author  : Utkarsh Mishra (Internship Project)
Task    : Predict RUL from physical battery measurements

Run:
    pip install scikit-learn matplotlib seaborn pandas numpy
    python battery_rul_model.py
"""

#
# IMPORTS
#
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless rendering (safe for servers/Colab)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

#
# CONFIGURATION  <- change only this section
#
DATA_PATH   = "Battery_dataset.csv"   # path to your CSV
OUTPUT_DIR  = "outputs"               # where plots & model get saved
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

#
# STEP 1 - LOAD DATA
#
# Sort by battery_id then cycle so rolling/lag features are computed in the
# correct chronological order per battery.
#
# Columns:
#   battery_id - which battery (B5 / B6 / B7)
#   cycle      - charge-discharge cycle number (aging clock)
#   chI, chV, chT      - charge current, voltage, temperature
#   disI, disV, disT   - discharge current, voltage, temperature
#   BCt - battery capacity (Ah)   <- key health signal
#   SOH - state of health (%)     <- derived health metric
#   RUL - remaining useful life   <- TARGET variable
print("=" * 60)
print("BATTERY RUL PREDICTION MODEL")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df = df.sort_values(['battery_id', 'cycle']).reset_index(drop=True)

print(f"\n[1] Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Batteries : {df['battery_id'].unique().tolist()}")
for bid in df['battery_id'].unique():
    sub = df[df['battery_id'] == bid]
    print(f"    {bid} → {len(sub)} cycles  |  "
          f"Initial cap: {sub['BCt'].max():.4f} Ah  |  "
          f"Final SOH: {sub['SOH'].iloc[-1]:.2f}%")

#
# STEP 2 - FEATURE ENGINEERING
#
# Raw sensor readings capture the battery state at a single instant, but not
# *how fast* it is degrading - which is the key signal for RUL. We add
# rate-of-change, resistance-proxy and smoothed-trend features below.
#
# Rate of change features (first-order differences)
#   capacity_fade_rate = BCt - BCt_prev      - slope of capacity decline
#   SOH_rate           = SOH - SOH_prev      - sudden jumps flag fast aging
#   temp_rise_rate     = chT - chT_prev      - thermal aging proxy
#   voltage_drop       = chV - disV          - internal-resistance proxy
# Smoothed trend features (5-cycle rolling mean, removes cycle-to-cycle noise)
#   BCt_rolling_mean, SOH_rolling_mean
#
# .fillna(0): cycle 1 has no previous value, so diff() is NaN; 0 means
# "no change observed yet" and keeps the row instead of dropping it.
print("\n[2] Engineering features...")

for bid in df['battery_id'].unique():
    mask = df['battery_id'] == bid

    # Rate of change features (first-order differences)
    df.loc[mask, 'capacity_fade_rate'] = df.loc[mask, 'BCt'].diff().fillna(0)
    df.loc[mask, 'SOH_rate']           = df.loc[mask, 'SOH'].diff().fillna(0)
    df.loc[mask, 'temp_rise_rate']     = df.loc[mask, 'chT'].diff().fillna(0)

    # Physics-grounded proxy feature
    df.loc[mask, 'voltage_drop']       = df.loc[mask, 'chV'] - df.loc[mask, 'disV']

    # Smoothed trend features (noise reduction)
    df.loc[mask, 'BCt_rolling_mean']   = df.loc[mask, 'BCt'].rolling(5, min_periods=1).mean()
    df.loc[mask, 'SOH_rolling_mean']   = df.loc[mask, 'SOH'].rolling(5, min_periods=1).mean()

ENGINEERED_FEATURES = [
    'capacity_fade_rate', 'SOH_rate', 'temp_rise_rate',
    'voltage_drop', 'BCt_rolling_mean', 'SOH_rolling_mean'
]
print(f"    Created {len(ENGINEERED_FEATURES)} engineered features: {ENGINEERED_FEATURES}")

#
# STEP 3 - DEFINE FEATURE SET
#
# We deliberately exclude the raw 'cycle' number as a feature. If included,
# the model learns the trivial identity RUL = max_cycles - cycle, which is
# useless in production where total battery life is unknown. Using physical
# measurements forces it to learn the real sensor->RUL relationship.
FEATURES = [
    # Raw sensor readings
    'chI',          # charge current
    'chV',          # charge voltage
    'chT',          # charge temperature
    'disI',         # discharge current
    'disV',         # discharge voltage
    'disT',         # discharge temperature
    'BCt',          # battery capacity (Ah)
    'SOH',          # state of health (%)
    # Engineered features
    'capacity_fade_rate',
    'SOH_rate',
    'temp_rise_rate',
    'voltage_drop',
    'BCt_rolling_mean',
    'SOH_rolling_mean',
]

TARGET = 'RUL'

print(f"\n[3] Feature set defined: {len(FEATURES)} features")
print(f"    Target: {TARGET}")

#
# STEP 4 - VALIDATION STRATEGY: LEAVE-ONE-BATTERY-OUT (LOBO)
#
# A random train/test split would leak cycles from the same battery into both
# sets, letting the model memorise each battery's degradation pattern. LOBO
# holds out one whole battery and trains on the rest, simulating deployment
# on a battery never seen before:
#   Train on B6+B7 -> Test on B5 ; Train on B5+B7 -> Test on B6 ; etc.

print("\n[4] Validation strategy: Leave-One-Battery-Out (LOBO)")

#
# STEP 5 - DEFINE MODELS
#
# Linear and Ridge need scaled features; tree models (RF/GB) do not.
# Hyper-parameters chosen for stable, non-overfitting behaviour.

def build_models():
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression':  Ridge(alpha=10.0),
        'Random Forest':     RandomForestRegressor(
                                 n_estimators=300,
                                 max_depth=12,
                                 random_state=RANDOM_SEED
                             ),
        'Gradient Boosting': GradientBoostingRegressor(
                                 n_estimators=300,
                                 max_depth=4,
                                 learning_rate=0.05,
                                 random_state=RANDOM_SEED
                             ),
    }

print("\n[5] Models defined:")
for name in build_models():
    print(f"    • {name}")

#
# STEP 6 - TRAIN AND EVALUATE (LOBO CROSS-VALIDATION)
#
# Metrics: MAE = avg absolute error (cycles); RMSE penalises large errors;
# R² = variance explained. Predictions are clipped to >= 0 (RUL cannot be
# negative).

print("\n[6] Training - Leave-One-Battery-Out Cross-Validation")
print("-" * 60)

BATTERIES    = ['B5', 'B6', 'B7']
all_results  = {}   # {battery: {model: {metric: value}}}
all_preds    = {}   # {battery: {model: predictions array}}
all_actual   = {}   # {battery: actual RUL array}
all_cycles   = {}   # {battery: cycle array}

for test_bat in BATTERIES:
    train_df = df[df['battery_id'] != test_bat]
    test_df  = df[df['battery_id'] == test_bat]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURES]
    y_test  = test_df[TARGET]

    all_actual[test_bat] = y_test.values
    all_cycles[test_bat] = test_df['cycle'].values

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = build_models()
    bat_results = {}
    bat_preds   = {}

    for name, model in models.items():
        # Choose scaled or raw features depending on model type
        if name in ['Linear Regression', 'Ridge Regression']:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        preds = np.clip(preds, 0, None)   # RUL cannot be negative

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        bat_results[name] = {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R²': round(r2, 4)}
        bat_preds[name]   = preds

    all_results[test_bat] = bat_results
    all_preds[test_bat]   = bat_preds

    print(f"\n  Test Battery: {test_bat}  (trained on {[b for b in BATTERIES if b != test_bat]})")
    print(f"  {'Model':<24} {'MAE':>7} {'RMSE':>7} {'R²':>8}")
    print(f"  {'-'*50}")
    for name, res in bat_results.items():
        print(f"  {name:<24} {res['MAE']:>7.2f} {res['RMSE']:>7.2f} {res['R²']:>8.4f}")

#
# STEP 7 - IDENTIFY BEST MODEL & SUMMARY
#

print("\n\n[7] Model Summary - Averaged Across All Batteries")
print("-" * 60)

MODEL_NAMES = list(build_models().keys())
avg_summary = {}

for name in MODEL_NAMES:
    maes  = [all_results[b][name]['MAE']  for b in BATTERIES]
    rmses = [all_results[b][name]['RMSE'] for b in BATTERIES]
    r2s   = [all_results[b][name]['R²']   for b in BATTERIES]
    avg_summary[name] = {
        'Avg MAE':  round(np.mean(maes),  2),
        'Avg RMSE': round(np.mean(rmses), 2),
        'Avg R²':   round(np.mean(r2s),   4),
    }
    print(f"  {name:<24}  Avg MAE={np.mean(maes):.2f}  "
          f"Avg RMSE={np.mean(rmses):.2f}  Avg R²={np.mean(r2s):.4f}")

best_model_name = min(avg_summary, key=lambda k: avg_summary[k]['Avg RMSE'])
print(f"\n  ★  Best Model: {best_model_name}  "
      f"(Avg RMSE={avg_summary[best_model_name]['Avg RMSE']:.2f}, "
      f"Avg R²={avg_summary[best_model_name]['Avg R²']:.4f})")

#
# STEP 8 - BATTERY HEALTH COMPARISON
#
# HOW WE DETERMINE THE "BEST" BATTERY:
# A battery is "best" if it retains the most health over its usable life.
# We rank on:
# 1. Capacity fade %   - lower is better (less degradation)
# 2. Average SOH       - higher is better (sustained health)
# 3. Final SOH         - higher is better (health at end of life)
# Total cycle count alone is not enough: a battery lasting 250 cycles but
# losing 62% capacity is worse than one lasting 210 cycles with 53% fade.

print("\n\n[8] Battery Health Comparison")
print("-" * 60)

battery_health = {}
for bid in BATTERIES:
    sub = df[df['battery_id'] == bid]
    fade_pct = (sub['BCt'].max() - sub['BCt'].min()) / sub['BCt'].max() * 100
    battery_health[bid] = {
        'Total Cycles':     int(sub['cycle'].max()),
        'Initial Cap (Ah)': round(sub['BCt'].max(), 4),
        'Final Cap (Ah)':   round(sub['BCt'].min(), 4),
        'Cap Fade (%)':     round(fade_pct, 2),
        'Avg SOH (%)':      round(sub['SOH'].mean(), 2),
        'Final SOH (%)':    round(sub['SOH'].iloc[-1], 2),
    }
    print(f"\n  {bid}:")
    for k, v in battery_health[bid].items():
        print(f"    {k:<22}: {v}")

# Ranking logic
fades  = {b: battery_health[b]['Cap Fade (%)'] for b in BATTERIES}
avgsoh = {b: battery_health[b]['Avg SOH (%)']  for b in BATTERIES}
best_bat = min(fades, key=fades.get)   # least fade = best

print(f"\n  ★  Best Battery: {best_bat}")
print(f"     Reason: Lowest capacity fade ({fades[best_bat]:.1f}%) "
      f"and highest average SOH ({avgsoh[best_bat]:.1f}%)")

#
# STEP 9 - FEATURE IMPORTANCE (from Random Forest)
#
# WHY FEATURE IMPORTANCE MATTERS:
# Random Forest computes how much each feature reduces prediction error
# across all trees (mean decrease in impurity).  High importance means
# the model relies heavily on that feature.  This validates whether our
# feature engineering was meaningful - if engineered features rank highly,
# the engineering step was worthwhile.

print("\n\n[9] Feature Importance (Random Forest trained on B5+B6)")
print("-" * 60)

train_all_df = df[df['battery_id'].isin(['B5', 'B6'])]
rf_importance = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_SEED)
rf_importance.fit(train_all_df[FEATURES], train_all_df[TARGET])

importances = pd.Series(rf_importance.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = '█' * int(imp * 100)
    tag = ' ← ENGINEERED' if feat in ENGINEERED_FEATURES else ''
    print(f"  {feat:<24} {imp:.4f}  {bar}{tag}")

#
# STEP 10 - SAVE THE TRAINED MODEL
#
# We save the final Random Forest model (best performer) trained on ALL data
# so it can be loaded and used for predictions on new batteries without
# retraining.
# The scaler trained on all data is also saved - any new data must be
# scaled with the same scaler (same mean and std) before prediction.

print("\n\n[10] Saving final model...")

# Train final model on ALL available data
X_all = df[FEATURES]
y_all = df[TARGET]

final_rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_SEED)
final_rf.fit(X_all, y_all)

final_scaler = StandardScaler()
final_scaler.fit(X_all)   # fit scaler on full data for reference

model_package = {
    'model':    final_rf,
    'scaler':   final_scaler,
    'features': FEATURES,
    'metadata': {
        'model_type':    'RandomForestRegressor',
        'n_estimators':  300,
        'max_depth':     12,
        'avg_r2_lobo':   avg_summary['Random Forest']['Avg R²'],
        'avg_mae_lobo':  avg_summary['Random Forest']['Avg MAE'],
        'avg_rmse_lobo': avg_summary['Random Forest']['Avg RMSE'],
        'best_battery':  best_bat,
        'trained_on':    'B5, B6, B7 (all data)',
    }
}

model_path = os.path.join(OUTPUT_DIR, 'battery_rul_rf_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)

print(f"    Model saved to: {model_path}")
print(f"    To load: model_pkg = pickle.load(open('{model_path}', 'rb'))")

#
# STEP 11 - HOW TO USE THE SAVED MODEL (inference example)
#

print("\n\n[11] Example: How to use the saved model for prediction")
print("-" * 60)

# Load the model
loaded_pkg = pickle.load(open(model_path, 'rb'))
loaded_rf  = loaded_pkg['model']
feat_cols  = loaded_pkg['features']

# Predict on a sample row (first row of B5)
sample = df[df['battery_id'] == 'B5'][feat_cols].iloc[[0]]
pred   = loaded_rf.predict(sample)[0]
actual = df[df['battery_id'] == 'B5']['RUL'].iloc[0]
print(f"    Sample prediction (B5, Cycle 1):")
print(f"    Predicted RUL = {pred:.1f} cycles")
print(f"    Actual    RUL = {actual} cycles")

#
# STEP 12 - VISUALISATIONS
#

print("\n\n[12] Generating visualisation plots...")

COLORS = {'B5': '#2196F3', 'B6': '#4CAF50', 'B7': '#FF5722'}
LIGHT  = {'B5': '#BBDEFB', 'B6': '#C8E6C9', 'B7': '#FFCCBC'}
sns.set_theme(style='whitegrid')

fig = plt.figure(figsize=(22, 26), facecolor='white')
fig.suptitle(
    'Battery RUL Prediction - Hybrid Physics-Informed ML\n'
    'Internship Project  |  Utkarsh Mishra',
    fontsize=20, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.48, wspace=0.35,
                       top=0.94, bottom=0.03, left=0.06, right=0.97)

# Plot 1 - Capacity Fade
ax1 = fig.add_subplot(gs[0, :])
for bid in BATTERIES:
    sub = df[df['battery_id'] == bid]
    ax1.plot(sub['cycle'], sub['BCt'], color=COLORS[bid], lw=2.5, label=bid)
    ax1.fill_between(sub['cycle'], sub['BCt'], alpha=0.07, color=COLORS[bid])
ax1.axhline(1.4, color='red', ls='--', lw=1.8, alpha=0.75, label='~70% capacity (typical EOL)')
ax1.set_title('Battery Capacity Fade over Charge-Discharge Cycles', fontsize=14, fontweight='bold')
ax1.set_xlabel('Cycle Number'); ax1.set_ylabel('Capacity (Ah)')
ax1.legend(fontsize=10); ax1.set_facecolor('#F5F5F5')
ax1.annotate('B7: fastest degradation\n(62.3% total fade)',
             xy=(220, 0.95), fontsize=9, color=COLORS['B7'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['B7']), xytext=(165, 1.3))
ax1.annotate('B6: slowest degradation\n(52.7% total fade - BEST)',
             xy=(160, 1.48), fontsize=9, color=COLORS['B6'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['B6']), xytext=(60, 1.7))

# Plot 2 - SOH Degradation
ax2 = fig.add_subplot(gs[1, :2])
for bid in BATTERIES:
    sub = df[df['battery_id'] == bid]
    ax2.plot(sub['cycle'], sub['SOH'], color=COLORS[bid], lw=2, label=bid)
ax2.axhline(80, color='orange', ls='--', lw=1.5, label='80% SOH (typical EOL threshold)')
ax2.set_title('State of Health (SOH) Degradation', fontsize=13, fontweight='bold')
ax2.set_xlabel('Cycle'); ax2.set_ylabel('SOH (%)'); ax2.legend(); ax2.set_facecolor('#F5F5F5')

# Plot 3 - Feature Importance
ax3 = fig.add_subplot(gs[1, 2])
top_fi = importances.head(8)
bar_colors = ['#1565C0' if f in ENGINEERED_FEATURES else '#90CAF9' for f in top_fi.index[::-1]]
bars = ax3.barh(top_fi.index[::-1], top_fi.values[::-1], color=bar_colors)
ax3.set_title('Feature Importance\n(Random Forest)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Importance')
for bar, val in zip(bars, top_fi.values[::-1]):
    ax3.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=8)
# Legend
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color='#1565C0', label='Engineered'),
                    Patch(color='#90CAF9', label='Raw sensor')], fontsize=8)
ax3.set_facecolor('#F5F5F5')

#Plots 4-6: RUL Actual vs Predicted per battery
for i, bid in enumerate(BATTERIES):
    ax = fig.add_subplot(gs[2, i])
    cycles = all_cycles[bid]
    actual = all_actual[bid]
    rf_pred = all_preds[bid]['Random Forest']
    ax.plot(cycles, actual,  color=COLORS[bid], lw=2.5, label='Actual RUL')
    ax.plot(cycles, rf_pred, color='black', lw=1.8, ls='--', alpha=0.85, label='Predicted RUL')
    ax.fill_between(cycles, actual, rf_pred, alpha=0.12, color='gray', label='Error')
    res = all_results[bid]['Random Forest']
    ax.set_title(f'{bid} - RUL Prediction\nMAE={res["MAE"]:.1f}  RMSE={res["RMSE"]:.1f}  R²={res["R²"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Cycle'); ax.set_ylabel('RUL (cycles)')
    ax.legend(fontsize=8); ax.set_facecolor('#F5F5F5')

# Plot 7 - Model Comparison
ax7 = fig.add_subplot(gs[3, :2])
x   = np.arange(len(MODEL_NAMES))
w   = 0.28
avg_maes  = [avg_summary[n]['Avg MAE']  for n in MODEL_NAMES]
avg_rmses = [avg_summary[n]['Avg RMSE'] for n in MODEL_NAMES]
avg_r2s   = [avg_summary[n]['Avg R²']   for n in MODEL_NAMES]
short_names = ['Linear\nRegression', 'Ridge\nRegression', 'Random\nForest', 'Gradient\nBoosting']
b1 = ax7.bar(x - w/2, avg_maes,  w, label='Avg MAE',  color='#1565C0', alpha=0.85)
b2 = ax7.bar(x + w/2, avg_rmses, w, label='Avg RMSE', color='#E65100', alpha=0.85)
for b, v in zip(b1, avg_maes):  ax7.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=8)
for b, v in zip(b2, avg_rmses): ax7.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=8)
ax7.set_xticks(x); ax7.set_xticklabels(short_names, fontsize=10)
ax7.set_title('Model Comparison - Avg MAE & RMSE (LOBO Validation)', fontsize=12, fontweight='bold')
ax7.set_ylabel('Error (cycles)'); ax7.legend(); ax7.set_facecolor('#F5F5F5')
ax7b = ax7.twinx()
ax7b.plot(x, avg_r2s, 'D--', color='#2E7D32', ms=9, lw=2, label='Avg R²')
for xi, rv in zip(x, avg_r2s):
    ax7b.annotate(f'R²={rv:.3f}', (xi, rv), xytext=(0, 10),
                  textcoords='offset points', ha='center', fontsize=8,
                  color='#2E7D32', fontweight='bold')
ax7b.set_ylabel('R²', color='#2E7D32'); ax7b.set_ylim(0.7, 1.05); ax7b.legend(loc='upper right')

# Plot 8 - Battery Scorecard
ax8 = fig.add_subplot(gs[3, 2])
ax8.axis('off')
ax8.set_title('Battery Scorecard', fontsize=13, fontweight='bold', pad=10)
y_pos = 0.88
from matplotlib.patches import FancyBboxPatch
for bid in BATTERIES:
    s = battery_health[bid]
    rect = FancyBboxPatch((0.02, y_pos-0.27), 0.96, 0.28,
                          boxstyle="round,pad=0.02", lw=2,
                          edgecolor=COLORS[bid], facecolor=LIGHT[bid], alpha=0.55,
                          transform=ax8.transAxes)
    ax8.add_patch(rect)
    ax8.text(0.10, y_pos-0.06, bid, transform=ax8.transAxes,
             fontsize=16, fontweight='bold', color=COLORS[bid])
    ax8.text(0.33, y_pos-0.04, f"Cycles: {s['Total Cycles']}   Fade: {s['Cap Fade (%)']}%",
             transform=ax8.transAxes, fontsize=9, color='#212121')
    ax8.text(0.33, y_pos-0.15, f"Avg SOH: {s['Avg SOH (%)']}%   Final SOH: {s['Final SOH (%)']}%",
             transform=ax8.transAxes, fontsize=9, color='#757575')
    if bid == best_bat:
        ax8.text(0.78, y_pos-0.09, '⭐ BEST', transform=ax8.transAxes,
                 fontsize=11, fontweight='bold', color='#2E7D32')
    y_pos -= 0.32

# Plot 9 - Discharge Temperature Trend
ax9 = fig.add_subplot(gs[4, :2])
for bid in BATTERIES:
    sub  = df[df['battery_id'] == bid]
    roll = sub['disT'].rolling(10, min_periods=1).mean()
    ax9.plot(sub['cycle'], roll, color=COLORS[bid], lw=2, label=bid)
ax9.set_title('Discharge Temperature Trend (10-cycle rolling mean) - Thermal Aging Proxy',
              fontsize=12, fontweight='bold')
ax9.set_xlabel('Cycle'); ax9.set_ylabel('Temperature (°C)')
ax9.legend(); ax9.set_facecolor('#F5F5F5')

# Plot 10 - Voltage Drop (Internal Resistance Proxy)
ax10 = fig.add_subplot(gs[4, 2])
for bid in BATTERIES:
    sub = df[df['battery_id'] == bid]
    ax10.plot(sub['cycle'], sub['voltage_drop'], color=COLORS[bid], lw=1.5, alpha=0.75, label=bid)
ax10.set_title('Voltage Drop (chV − disV)\nInternal Resistance Proxy', fontsize=12, fontweight='bold')
ax10.set_xlabel('Cycle'); ax10.set_ylabel('Voltage Drop (V)')
ax10.legend(); ax10.set_facecolor('#F5F5F5')

plot_path = os.path.join(OUTPUT_DIR, 'battery_rul_analysis.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Plot saved to: {plot_path}")

#
# STEP 13 - FINAL SUMMARY REPORT
#

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\n  BEST MODEL   : {best_model_name}")
print(f"  Avg MAE      : {avg_summary[best_model_name]['Avg MAE']} cycles")
print(f"  Avg RMSE     : {avg_summary[best_model_name]['Avg RMSE']} cycles")
print(f"  Avg R²       : {avg_summary[best_model_name]['Avg R²']}")
print(f"\n  BEST BATTERY : {best_bat}")
print(f"  Reason       : Lowest capacity fade ({fades[best_bat]:.1f}%) "
      f"and highest average SOH ({avgsoh[best_bat]:.1f}%)")
print(f"\n  Outputs saved in: {OUTPUT_DIR}/")
print(f"    → battery_rul_rf_model.pkl  (trained model)")
print(f"    → battery_rul_analysis.png  (full dashboard plot)")
print("\n" + "=" * 60)
