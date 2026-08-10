import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH = "ftcs_results.csv"

# One color per battery — used for FTCS lines
BAT_COLORS = {
    'B5': '#e41a1c',   # red
    'B6': '#377eb8',   # blue
    'B7': '#4daf4a',   # green
}

FIGSIZE   = (10, 5)
LINEWIDTH = 1.4
GRID_ALPHA = 0.3

# =============================================================================
# LOAD
# =============================================================================
res = pd.read_csv(CSV_PATH)
batteries = res['battery_id'].unique().tolist()
print(f"Loaded {len(res)} rows — batteries: {batteries}")

# =============================================================================
# PLOT FUNCTION HELPERS
# =============================================================================

def save(fig, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def metrics_box(ax, mse, rmse, mae, loc='upper right'):
    """Add a small MSE/RMSE/MAE text box to an axes."""
    txt = f"MSE  = {mse:.6f}°C²\nRMSE = {rmse:.4f}°C\nMAE  = {mae:.4f}°C"
    # Position based on loc
    if loc == 'upper right':
        x, y, ha, va = 0.98, 0.97, 'right', 'top'
    elif loc == 'upper left':
        x, y, ha, va = 0.02, 0.97, 'left', 'top'
    elif loc == 'lower right':
        x, y, ha, va = 0.98, 0.03, 'right', 'bottom'
    else:  # lower left
        x, y, ha, va = 0.02, 0.03, 'left', 'bottom'
    ax.text(x, y, txt,
            transform=ax.transAxes,
            va=va, ha=ha, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))


# =============================================================================
# PER-BATTERY PLOTS
# =============================================================================
for bat in batteries:
    sub   = res[res['battery_id'] == bat].copy()
    cyc   = sub['cycle']
    color = BAT_COLORS.get(bat, 'tab:red')

    mse_c  = mean_squared_error(sub['T_core_Greens'],    sub['T_core_FTCS'])
    rmse_c = np.sqrt(mse_c)
    mae_c  = mean_absolute_error(       sub['T_core_Greens'],    sub['T_core_FTCS'])
    mse_s  = mean_squared_error(sub['T_surface_Greens'], sub['T_surface_FTCS'])
    rmse_s = np.sqrt(mse_s)
    mae_s  = mean_absolute_error(       sub['T_surface_Greens'], sub['T_surface_FTCS'])
    mse_dt  = mean_squared_error(sub['delta_T_Greens'], sub['delta_T_FTCS'])
    rmse_dt = np.sqrt(mse_dt)
    mae_dt  = mean_absolute_error(sub['delta_T_Greens'], sub['delta_T_FTCS'])

    print(f"\n{bat} — generating plots …")
    print(f"  Core    MSE={mse_c:.8f}°C²  RMSE={rmse_c:.6f}°C  MAE={mae_c:.6f}°C")
    print(f"  Surface MSE={mse_s:.8f}°C²  RMSE={rmse_s:.6f}°C  MAE={mae_s:.6f}°C")
    print(f"  ΔT      MSE={mse_dt:.8f}°C²  RMSE={rmse_dt:.6f}°C  MAE={mae_dt:.6f}°C")

    # ------------------------------------------------------------------
    # PLOT 1 — Core temperature
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(cyc, sub['T_core_Greens'], color='steelblue',
            lw=LINEWIDTH, label="Green's Function (Analytical)", zorder=3)
    ax.plot(cyc, sub['T_core_FTCS'], color=color,
            lw=LINEWIDTH, ls='--', label='FTCS (Numerical)', zorder=2)

    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Core Temperature (°C)', fontsize=12)
    ax.set_title(f'{bat} — Core Temperature: FTCS vs Green\'s Function',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=GRID_ALPHA)
    metrics_box(ax, mse_c, rmse_c, mae_c)

    save(fig, f"plot/{bat}_core_temp.png")

    # ------------------------------------------------------------------
    # PLOT 2 — Surface temperature
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(cyc, sub['T_surface_Greens'], color='steelblue',
            lw=LINEWIDTH, label="Green's Function (Analytical)", zorder=3)
    ax.plot(cyc, sub['T_surface_FTCS'], color=color,
            lw=LINEWIDTH, ls='--', label='FTCS (Numerical)', zorder=2)

    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Surface Temperature (°C)', fontsize=12)
    ax.set_title(f'{bat} — Surface Temperature: FTCS vs Green\'s Function',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=GRID_ALPHA)
    metrics_box(ax, mse_s, rmse_s, mae_s)

    save(fig, f"plot/{bat}_surface_temp.png")

    # ------------------------------------------------------------------
    # PLOT 3 — Error (FTCS − Green's)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(cyc, sub['error_core'],    color=color,    lw=LINEWIDTH,
            label='Core error')
    ax.plot(cyc, sub['error_surface'], color=color,    lw=LINEWIDTH,
            ls=':', alpha=0.75, label='Surface error')
    ax.axhline(0, color='black', ls='--', lw=1.0, alpha=0.5)

    # Trend line for core error
    z = np.polyfit(cyc, sub['error_core'], 1)
    p = np.poly1d(z)
    ax.plot(cyc, p(cyc), color='gray', lw=1.2, ls='-.',
            label=f'Core trend  (slope={z[0]:.2e}°C/cycle)')

    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Error  FTCS − Green\'s (°C)', fontsize=12)
    ax.set_title(f'{bat} — FTCS Error over Cycles',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=GRID_ALPHA)

    # Annotate mean errors
    mean_ec = sub['error_core'].mean()
    mean_es = sub['error_surface'].mean()
    ax.text(0.02, 0.05,
            f"Mean core error   = {mean_ec:+.4f}°C\n"
            f"Mean surface error= {mean_es:+.4f}°C",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    save(fig, f"plot/{bat}_error.png")

    # ------------------------------------------------------------------
    # PLOT 4 — ΔT (core − surface)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(cyc, sub['delta_T_FTCS'],   color=color,     lw=LINEWIDTH,
            label='FTCS ΔT')
    ax.plot(cyc, sub['delta_T_Greens'], color='steelblue', lw=LINEWIDTH,
            ls='--', label="Green's ΔT")

    # Mean lines
    ax.axhline(sub['delta_T_FTCS'].mean(),   color=color,      ls='-.',
               lw=1.2, alpha=0.7,
               label=f"FTCS mean = {sub['delta_T_FTCS'].mean():.4f}°C")
    ax.axhline(sub['delta_T_Greens'].mean(), color='steelblue', ls='-.',
               lw=1.2, alpha=0.7,
               label=f"Green's mean = {sub['delta_T_Greens'].mean():.4f}°C")

    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('ΔT = T_core − T_surface (°C)', fontsize=12)
    ax.set_title(f'{bat} — Temperature Gradient (Core − Surface)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=GRID_ALPHA)
    metrics_box(ax, mse_dt, rmse_dt, mae_dt)

    save(fig, f"plot/{bat}_delta_T.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Battery':<10} {'Core MSE':>14} {'Core RMSE':>12} {'Core MAE':>12} "
      f"{'Surf MSE':>14} {'Surf RMSE':>12} {'Surf MAE':>12} "
      f"{'ΔT MSE':>14} {'ΔT RMSE':>12} {'ΔT MAE':>12}")
print("-" * 120)
for bat in batteries:
    sub    = res[res['battery_id'] == bat]
    mse_c  = mean_squared_error(sub['T_core_Greens'],    sub['T_core_FTCS'])
    rmse_c = np.sqrt(mse_c)
    mae_c  = mean_absolute_error(       sub['T_core_Greens'],    sub['T_core_FTCS'])
    mse_s  = mean_squared_error(sub['T_surface_Greens'], sub['T_surface_FTCS'])
    rmse_s = np.sqrt(mse_s)
    mae_s  = mean_absolute_error(       sub['T_surface_Greens'], sub['T_surface_FTCS'])
    mse_dt  = mean_squared_error(sub['delta_T_Greens'], sub['delta_T_FTCS'])
    rmse_dt = np.sqrt(mse_dt)
    mae_dt  = mean_absolute_error(sub['delta_T_Greens'], sub['delta_T_FTCS'])
    print(f"{bat:<10} {mse_c:>14.8f} {rmse_c:>12.6f} {mae_c:>12.6f} "
          f"{mse_s:>14.8f} {rmse_s:>12.6f} {mae_s:>12.6f} "
          f"{mse_dt:>14.8f} {rmse_dt:>12.6f} {mae_dt:>12.6f}")

print(f"\nOutput files per battery (12 total):")
for bat in batteries:
    for suffix in ['core_temp', 'surface_temp', 'error', 'delta_T']:
        print(f"  plot/{bat}_{suffix}.png")

print("\nplot.py — Done ✓")