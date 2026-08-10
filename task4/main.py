"""
main.py
───────
Entry point — loads CSV, calls compute_ftcs for each battery,
then produces metrics, CSV, and profile plots.

Usage:
    python main.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from compute_ftcs import solve_battery, R0, N, dt, dt_max, F, BC_surf, alpha, dr

# ── 1. SETUP SUMMARY ─────────────────────────────────────────────────────
print("=" * 60)
print("TASK 4 — FTCS Battery Thermal Solver")
print("=" * 60)
print(f"  α          = {alpha:.4e} m²/s")
print(f"  Grid       : N={N}, dr={dr*1000:.3f} mm, {N+1} points")
print(f"  Stability  : dt_max={dt_max:.4f}s, using dt={dt}s  → STABLE ✓")
print(f"  Fourier #  : F={F:.4f}  (must be < 0.5)")
print(f"  Biot term  : BC_surf={BC_surf:.4f}")

# ── 2. LOAD CSV ──────────────────────────────────────────────────────────
CSV_PATH = "HANDOVER_task3_thermal_model.csv"
df = pd.read_csv(CSV_PATH)
batteries = df['battery_id'].unique().tolist()
print(f"\nLoaded {len(df)} rows — batteries: {batteries}")

# ── 3. RUN FTCS FOR EACH BATTERY ─────────────────────────────────────────
all_results = []
profiles    = {}

for bat in batteries:
    sub = df[df['battery_id'] == bat]
    print(f"\n{'─'*60}")
    print(f"  Processing {bat}  ({len(sub)} cycles) …")

    results, profile = solve_battery(sub)
    all_results.extend(results)
    profiles[bat] = profile

# ── 4. RESULTS DATAFRAME + METRICS ───────────────────────────────────────
res = pd.DataFrame(all_results)

print(f"\n{'='*60}")
print("VALIDATION METRICS  (FTCS vs Green's Function)")
print(f"{'='*60}")
for bat in batteries:
    sub    = res[res['battery_id'] == bat]
    rmse_c = np.sqrt(mean_squared_error(sub['T_core_Greens'],    sub['T_core_FTCS']))
    mae_c  = mean_absolute_error(       sub['T_core_Greens'],    sub['T_core_FTCS'])
    rmse_s = np.sqrt(mean_squared_error(sub['T_surface_Greens'], sub['T_surface_FTCS']))
    mae_s  = mean_absolute_error(       sub['T_surface_Greens'], sub['T_surface_FTCS'])
    print(f"\n  {bat}:")
    print(f"    Core    RMSE={rmse_c:.6f}°C   MAE={mae_c:.6f}°C")
    print(f"    Surface RMSE={rmse_s:.6f}°C   MAE={mae_s:.6f}°C")
    print(f"    Mean ΔT  FTCS={sub['delta_T_FTCS'].mean():.4f}°C  "
          f"Green's={sub['delta_T_Greens'].mean():.4f}°C")

rmse_c_all = np.sqrt(mean_squared_error(res['T_core_Greens'],    res['T_core_FTCS']))
mae_c_all  = mean_absolute_error(       res['T_core_Greens'],    res['T_core_FTCS'])
print(f"\n  OVERALL  Core RMSE={rmse_c_all:.6f}°C   MAE={mae_c_all:.6f}°C")

# ── 5. SAVE RESULTS CSV ─────────────────────────────────────────────────
out_csv = "ftcs_results.csv"
res.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ── 6. T(r) PROFILE PLOTS — one per battery ─────────────────────────────
print("\nGenerating T(r) profile plots …")

SNAP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#bcbd22']

for bat in batteries:
    pdata     = profiles[bat]
    snapshots = pdata['snapshots']
    r_mm      = pdata['r'] * 1000
    cyc_num   = pdata['profile_cycle']

    fig, ax = plt.subplots(figsize=(10, 6))

    for k_idx, (t_sec, T_snap) in enumerate(snapshots):
        color = SNAP_COLORS[k_idx % len(SNAP_COLORS)]
        ax.plot(r_mm, T_snap,
                color=color, linewidth=2.0,
                label=f"t = {int(t_sec)} s")
        # Place time label at the right end of each line
        ax.annotate(f"t = {int(t_sec)} s",
                    xy=(r_mm[-1], T_snap[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold',
                    va='center', ha='left')

    ax.set_xlabel('Radial position  r (mm)', fontsize=12)
    ax.set_ylabel('Temperature  T (°C)', fontsize=12)
    ax.set_title(
        f'{bat} — T(r) profile at selected times  (cycle {cyc_num})',
        fontsize=13, fontweight='bold')
    ax.axvline(x=0,       color='gray', ls=':', lw=1, alpha=0.6)
    ax.axvline(x=R0*1000, color='gray', ls='--', lw=1, alpha=0.6)
    ax.text(0.01, 0.97, 'core', transform=ax.transAxes,
            va='top', fontsize=9, color='gray')
    ax.text(0.97, 0.97, 'surface', transform=ax.transAxes,
            va='top', ha='right', fontsize=9, color='gray')
    ax.grid(True, alpha=0.3)

    fname = f"profile_{bat}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

print("\nDone ✓")
