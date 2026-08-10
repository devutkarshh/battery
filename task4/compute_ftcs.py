import numpy as np


R0    = 0.009    # battery radius (m)
k     = 0.5      # thermal conductivity (W/m·K)
rho   = 2600     # density (kg/m³)
cp    = 1000     # specific heat capacity (J/kg·K)
h     = 10       # convective heat transfer coefficient (W/m²·K)
T_amb = 25.0     # ambient temperature (°C)
alpha = k / (rho * cp)   # thermal diffusivity


N  = 18
dr = R0 / N
r  = np.linspace(0, R0, N + 1)

dt_max = (dr ** 2) / (2 * alpha)
dt     = 0.5
assert dt <= dt_max, f"UNSTABLE: dt={dt} > dt_max={dt_max:.4f}"


F       = alpha * dt / dr ** 2      # 2nd order differential equation coefficient
G       = alpha * dt / (2 * dr)     # 1st order differential equation coefficient
S       = dt / (rho * cp)           # Heat generation per unit mass
BC_surf = (h * dr) / k              # Surface boundary condition coefficient

# Snapshot times (seconds) for T(r) profile capture
SNAPSHOT_TIMES_S = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


def solve_battery(sub_df):
    sub = sub_df.reset_index(drop=True)

    # Pick middle cycle for profile snapshots
    profile_cycle_idx = len(sub) // 2
    profile_cycle_num = int(sub.loc[profile_cycle_idx, 'cycle'])
    profile_data      = []

    results = []

    for idx, row in sub.iterrows():
        cycle   = int(row['cycle'])
        Q_dot   = row['Q_dot_W_m3']
        t_cyc   = row['t_cycle_s']
        T_ref_c = row['T_centre_C']
        T_ref_s = row['T_surface_C']

        Nt    = int(t_cyc / dt)
        T     = np.ones(N + 1) * T_amb
        T_new = np.empty(N + 1)
        Q_src = Q_dot * S

        is_profile_cycle = (idx == profile_cycle_idx)

        # Convert target times → nearest step indices
        target_steps = set()
        if is_profile_cycle:
            for t_target in SNAPSHOT_TIMES_S:
                step = int(round(t_target / dt))
                if 0 < step < Nt:
                    target_steps.add(step)
            profile_data.append((0.0, T.copy()))


        # main loop for ftcs
        for n in range(Nt):
            # — Interior points (i = 1 … N-1) —
            for i in range(1, N):
                d2T      = (T[i+1] - 2*T[i] + T[i-1]) * F
                dT_term  = (T[i+1] - T[i-1]) * G / r[i]
                T_new[i] = T[i] + d2T + dT_term + Q_src

            # — Center BC: symmetry  T[0] = T[1]
            T_new[0] = T_new[1]

            # — Surface BC: Convective Heat Transfer
            T_new[N] = (T_new[N-1] + BC_surf * T_amb) / (1 + BC_surf)

            T[:] = T_new

            # Capture snapshot
            if is_profile_cycle and (n + 1) in target_steps:
                profile_data.append(((n + 1) * dt, T.copy()))

        results.append({
            'battery_id'      : row['battery_id'],
            'cycle'           : cycle,
            'Q_dot_W_m3'      : Q_dot,
            't_cycle_s'       : t_cyc,
            'T_core_FTCS'     : T[0],
            'T_surface_FTCS'  : T[N],
            'delta_T_FTCS'    : T[0] - T[N],
            'T_core_Greens'   : T_ref_c,
            'T_surface_Greens': T_ref_s,
            'delta_T_Greens'  : row['delta_T_C'],
            'error_core'      : T[0]  - T_ref_c,
            'error_surface'   : T[N]  - T_ref_s,
        })

        if cycle % 50 == 0 or cycle == 1:
            print(f"    Cycle {cycle:3d}: T_core={T[0]:.4f}°C  "
                  f"(Green's={T_ref_c:.4f}°C)  "
                  f"err={T[0]-T_ref_c:+.4f}°C  "
                  f"ΔT={T[0]-T[N]:.4f}°C")

    profile = {
        'r'            : r,
        'snapshots'    : profile_data,
        'profile_cycle': profile_cycle_num,
    }

    return results, profile
