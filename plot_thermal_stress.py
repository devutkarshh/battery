import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import parameters and helper profile functions from analytical_model
from analytical_model import (
    Vol, R0, k_r, E_mod, beta, nu,
    compute_temperature_profile,
    compute_stress_profile
)

def main():
    for name in ["B5", "B6", "B7"]:
        input_path = f"analysed_{name}.csv"
        if not os.path.exists(input_path):
            print(f"Warning: Calculated data '{input_path}' not found. Please run analytical_model.py first. Skipping {name}.")
            continue
            
        print(f"\nReading {input_path}...")
        df = pd.read_csv(input_path)
        
        # 1. Plotting raw heat_gen_vs_cycle
        print(f"  Generating heat_gen_vs_cycle plot for {name}...")
        graph_dir_heat = f"graphs/{name}/heat_gen_vs_cycle"
        os.makedirs(graph_dir_heat, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df, 
            x="cycle", 
            y="heat_gen", 
            color="#3A86C8", 
            linewidth=2.0, 
            label="Heat Generation (W)"
        )
        plt.title(f"Heat Generation vs. Cycle (Battery {name}, Charge Phase)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Cycle Number", fontsize=12)
        plt.ylabel("Heat Generation Rate (W)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_heat = os.path.join(graph_dir_heat, "heat_gen_vs_cycle.png")
        plt.savefig(plot_path_heat, dpi=150)
        plt.close()
        print(f"    Saved raw graph to '{plot_path_heat}'")

        # 2. Plotting avg_heat_gen_vs_cycle
        print(f"  Generating avg_heat_gen_vs_cycle plot for {name}...")
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df, 
            x="cycle", 
            y="heat_gen", 
            color="#3A86C8", 
            alpha=0.3, 
            linewidth=1.2, 
            label="Raw Heat Generation (W)"
        )
        sns.lineplot(
            data=df, 
            x="cycle", 
            y="avg_heat_gen", 
            color="#D85A38", 
            linewidth=2.5, 
            label="7-Point Moving Average (W)"
        )
        plt.title(f"7-Point Moving Average Heat Generation vs. Cycle (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Cycle Number", fontsize=12)
        plt.ylabel("Heat Generation Rate (W)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_avg = os.path.join(graph_dir_heat, "avg_heat_gen_vs_cycle.png")
        plt.savefig(plot_path_avg, dpi=150)
        plt.close()
        print(f"    Saved moving average graph to '{plot_path_avg}'")

        # 3. Plotting temperature_profile T(r) for cycle 1
        print(f"  Generating temperature_profile plot for Cycle 1 of {name}...")
        row_1 = df[df['cycle'] == 1].iloc[0]
        T_s_1 = row_1['chT']
        q_1 = row_1['heat_gen']
        q_vol_1 = q_1 / Vol
        
        r_vals = np.linspace(0, R0, 100)
        T_profile_1 = compute_temperature_profile(r_vals, T_s_1, q_vol_1, R0, k_r)
        
        temp_graph_dir = f"graphs/{name}/temperature_profile"
        os.makedirs(temp_graph_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x=r_vals * 1000, 
            y=T_profile_1, 
            color="#D84B20", 
            linewidth=2.5, 
            label="Temperature Profile T(r)"
        )
        plt.title(f"Radial Temperature Profile T(r) at Cycle 1 (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Radial Position r (mm)", fontsize=12)
        plt.ylabel("Temperature T (°C)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_temp = os.path.join(temp_graph_dir, "temp_profile_cycle_1.png")
        plt.savefig(plot_path_temp, dpi=150)
        plt.close()
        print(f"    Saved temperature_profile graph to '{plot_path_temp}'")

        # 4. Plotting deltaT_vs_cycle
        print(f"  Generating deltaT_vs_cycle plot for {name}...")
        delta_graph_dir = f"graphs/{name}/deltaT_vs_cycle"
        os.makedirs(delta_graph_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df, 
            x="cycle", 
            y="deltaT", 
            color="#8C564B", 
            linewidth=2.0, 
            label="Delta T (°C)"
        )
        plt.title(f"Core-to-Surface Temperature Difference (Delta T) vs. Cycle (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Cycle Number", fontsize=12)
        plt.ylabel("Delta T (°C)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_delta = os.path.join(delta_graph_dir, "deltaT_vs_cycle.png")
        plt.savefig(plot_path_delta, dpi=150)
        plt.close()
        print(f"    Saved deltaT_vs_cycle graph to '{plot_path_delta}'")

        # 5. Plotting stress profile sigma_theta(r) for cycle 1
        print(f"  Generating stress profile plot for Cycle 1 of {name}...")
        q_1 = df[df['cycle'] == 1]['heat_gen'].values[0]
        q_vol_1 = q_1 / Vol
        stress_profile_1 = compute_stress_profile(r_vals, q_vol_1, R0, k_r, E_mod, beta, nu)
        
        stress_graph_dir = f"graphs/{name}/stress_vs_r"
        os.makedirs(stress_graph_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x=r_vals * 1000, 
            y=stress_profile_1 / 1e6, 
            color="#E377C2", 
            linewidth=2.5, 
            label="Tangential Stress (MPa)"
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        r_neutral = (R0 / np.sqrt(3)) * 1000
        plt.axvline(r_neutral, color="#2CA02C", linestyle=":", linewidth=1.5, label=f"Neutral Axis (~{r_neutral:.2f} mm)")
        plt.title(f"Radial Tangential Thermal Stress Profile σ_θ(r) at Cycle 1 (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Radial Position r (mm)", fontsize=12)
        plt.ylabel("Tangential Thermal Stress (MPa)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_stress = os.path.join(stress_graph_dir, "stress_vs_r_cycle_1.png")
        plt.savefig(plot_path_stress, dpi=150)
        plt.close()
        print(f"    Saved stress profile graph to '{plot_path_stress}'")

        # 6. Plotting stress_vs_cycle
        print(f"  Generating stress_vs_cycle plot for {name}...")
        stress_cycle_dir = f"graphs/{name}/stress_vs_cycle"
        os.makedirs(stress_cycle_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df, 
            x="cycle", 
            y=df["stress"] / 1e6, 
            color="#E377C2", 
            linewidth=2.0, 
            label="Surface Stress (MPa)"
        )
        plt.title(f"Peak Surface Tangential Thermal Stress vs. Cycle (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Cycle Number", fontsize=12)
        plt.ylabel("Thermal Stress (MPa)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_stress_cycle = os.path.join(stress_cycle_dir, "stress_vs_cycle.png")
        plt.savefig(plot_path_stress_cycle, dpi=150)
        plt.close()
        print(f"    Saved stress_vs_cycle graph to '{plot_path_stress_cycle}'")

        # 7. Plotting stress_vs_deltaT
        print(f"  Generating stress_vs_deltaT plot for {name}...")
        stress_deltaT_dir = f"graphs/{name}/stress_vs_deltaT"
        os.makedirs(stress_deltaT_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x="deltaT", 
            y=df["stress"] / 1e6, 
            color="#2CA02C", 
            s=40,
            label="Cycles"
        )
        plt.title(f"Thermal Stress vs. Core-to-Surface Temperature Difference (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Delta T (°C)", fontsize=12)
        plt.ylabel("Thermal Stress (MPa)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_stress_deltaT = os.path.join(stress_deltaT_dir, "stress_vs_deltaT.png")
        plt.savefig(plot_path_stress_deltaT, dpi=150)
        plt.close()
        print(f"    Saved stress_vs_deltaT graph to '{plot_path_stress_deltaT}'")

        # 8. Plotting cumulative_stress and SOH vs cycle
        print(f"  Generating cumulative_stress and SOH vs cycle plot for {name}...")
        cum_stress_dir = f"graphs/{name}/cumulative_stress_vs_cycle"
        os.makedirs(cum_stress_dir, exist_ok=True)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color_soh = "#1F77B4"
        ax1.set_xlabel("Cycle Number", fontsize=12)
        ax1.set_ylabel("State of Health (SOH %)", color=color_soh, fontsize=12)
        sns.lineplot(
            data=df, 
            x="cycle", 
            y="SOH", 
            color=color_soh, 
            linewidth=2.5, 
            ax=ax1,
            label="Actual SOH (%)"
        )
        ax1.tick_params(axis='y', labelcolor=color_soh)
        ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
        
        ax2 = ax1.twinx()
        color_stress = "#D62728"
        ax2.set_ylabel("Cumulative Thermal Stress (MPa)", color=color_stress, fontsize=12)
        sns.lineplot(
            data=df, 
            x="cycle", 
            y=df["cumulative_stress"] / 1e6, 
            color=color_stress, 
            linewidth=2.5, 
            ax=ax2,
            label="Cumulative Stress (MPa)"
        )
        ax2.tick_params(axis='y', labelcolor=color_stress)
        ax2.yaxis.grid(False)
        plt.title(f"SOH (%) and Cumulative Thermal Stress vs. Cycle (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        fig.tight_layout()
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=11)
        if ax2.get_legend(): ax2.get_legend().remove()
        plot_path_cum = os.path.join(cum_stress_dir, "cumulative_stress_and_soh_vs_cycle.png")
        plt.savefig(plot_path_cum, dpi=150)
        plt.close()
        print(f"    Saved cumulative stress_vs_cycle graph to '{plot_path_cum}'")

        # 9. Plotting cumulative_stress_vs_soh
        print(f"  Generating cumulative_stress_vs_soh plot for {name}...")
        cum_stress_soh_dir = f"graphs/{name}/cumulative_stress_vs_soh"
        os.makedirs(cum_stress_soh_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=df, 
            x=df["cumulative_stress"] / 1e6, 
            y="SOH", 
            color="#2CA02C", 
            scatter_kws={"s": 30, "alpha": 0.7},
            line_kws={"color": "#D62728", "linewidth": 2.0, "label": "Linear Fit"}
        )
        plt.title(f"State of Health (SOH %) vs. Cumulative Thermal Stress (Battery {name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Cumulative Thermal Stress (MPa)", fontsize=12)
        plt.ylabel("State of Health (SOH %)", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path_cum_soh = os.path.join(cum_stress_soh_dir, "cumulative_stress_vs_soh.png")
        plt.savefig(plot_path_cum_soh, dpi=150)
        plt.close()
        print(f"    Saved cumulative stress vs SOH graph to '{plot_path_cum_soh}'")

    print("\nAll graphs successfully created.")

if __name__ == "__main__":
    main()
