import numpy as np
import pandas as pd
import os

# Physical parameters for 18650 cell. Collected from internet.
V_ocv = 3.7          # Nominal open-circuit voltage (V)
R_int = 0.07         # Internal resistance (Ohm)
R0 = 0.009           # Outer radius (m) = 9 mm
L_cell = 0.065       # Cell length (m) = 65 mm
Vol = np.pi * (R0 ** 2) * L_cell # Cell volume (m³)
k_r = 0.5            # Radial thermal conductivity (W/(m·K))
E_mod = 10e9         # Young's modulus (Pa)
beta = 10e-6         # Thermal expansion coefficient (1/K)
nu = 0.3             # Poisson's ratio

def compute_heat_generation(ch_I, ch_V, V_ocv=3.7, R_int=0.07):
    """
    Calculate heat generation rate for the charge phase:
    q = I_ch * (V_ch - V_ocv) + I_ch^2 * R_int
    """
    q_ovp = ch_I * (ch_V - V_ocv)
    q_joule = (ch_I ** 2) * R_int
    return q_ovp + q_joule

def compute_temperature_profile(r, T_s, q_vol, R0=0.009, k=0.5):
    """
    Calculate temperature profile T(r) based on analytical heat equation solution:
    T(r) = T_s + q_vol / (4 * k) * (R0^2 - r^2)
    """
    return T_s + (q_vol / (4 * k)) * (R0**2 - r**2)

def compute_delta_T(q_vol, R0=0.009, k=0.5):
    """
    Calculate core-to-surface temperature difference delta_T:
    delta_T = q_vol * R0^2 / (4 * k)
    """
    return (q_vol * (R0 ** 2)) / (4 * k)

def compute_stress_profile(r, q_vol, R0=0.009, k=0.5, E=10e9, beta=10e-6, nu=0.3):
    """
    Calculate thermal stress profile sigma_theta(r):
    sigma = E * beta / (1 - nu) * (A / 4) * (3 * r^2 - R0^2)
    where A = q_vol / (4 * k)
    """
    A = q_vol / (4 * k)
    prefactor = (E * beta) / (1 - nu) * (A / 4)
    return prefactor * (3 * (r ** 2) - R0 ** 2)

def compute_surface_stress(dT, E=10e9, beta=10e-6, nu=0.3):
    """
    Calculate peak tangential thermal stress at the surface:
    sigma = 0.5 * E * beta * dT / (1 - nu)
    """
    return (0.5 * E * beta * dT) / (1 - nu)

def main():
    for name in ["B5", "B6", "B7"]:
        input_path = f"battery_splits/{name}.csv"
        output_path = f"analysed_{name}.csv"
        
        if not os.path.exists(input_path):
            print(f"Warning: Input file '{input_path}' not found, skipping.")
            continue
            
        print(f"\nProcessing {input_path}...")
        df = pd.read_csv(input_path)
        
        # Calculate heat generation for each cycle/row
        print("  Calculating heat generation per cycle (charge phase only)...")
        df['heat_gen'] = df.apply(lambda row: compute_heat_generation(row['chI'], row['chV'], V_ocv, R_int), axis=1)
        
        # Calculate 7-point moving average
        print("  Applying 7-point moving average on heat_gen...")
        df['avg_heat_gen'] = df['heat_gen'].rolling(window=7, min_periods=1).mean()
        
        # Calculate deltaT for each cycle/row
        print("  Calculating core-to-surface temperature difference deltaT...")
        df['deltaT'] = df.apply(lambda row: compute_delta_T(row['heat_gen'] / Vol, R0, k_r), axis=1)
        
        # Calculate thermal stress at the surface for each cycle/row (in Pa)
        print("  Calculating peak tangential thermal stress at the surface...")
        df['stress'] = df.apply(lambda row: compute_surface_stress(row['deltaT'], E_mod, beta, nu), axis=1)
        
        # Calculate cumulative stress (in Pa)
        print("  Calculating cumulative thermal stress...")
        df['cumulative_stress'] = df['stress'].cumsum()
        
        # Save to new CSV file
        df.to_csv(output_path, index=False)
        print(f"  Successfully created '{output_path}'.")

if __name__ == "__main__":
    main()
