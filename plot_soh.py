import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_soh_vs_cycle():
    csv_path = "Battery_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Set up beautiful styling
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 7))
    
    # Plot SOH vs cycle with custom styling
    sns.lineplot(
        data=df,
        x="cycle",
        y="SOH",
        hue="battery_id",
        palette="viridis",
        linewidth=2.5,
        alpha=0.9
    )
    
    # Enhance the labels and title with modern typography
    plt.title("Battery State of Health (SOH) vs. Cycle Life", fontsize=16, fontweight="bold", pad=20, color="#2C3E50")
    plt.xlabel("Cycle Number", fontsize=12, fontweight="medium", color="#34495E")
    plt.ylabel("State of Health (SOH %)", fontsize=12, fontweight="medium", color="#34495E")
    
    # Legend customization
    plt.legend(title="Battery ID", title_fontsize="11", fontsize="10", loc="lower left", frameon=True, facecolor="white", edgecolor="#E2E8F0")
    
    # Adjust layout and grid styling
    plt.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")
    plt.tight_layout()
    
    # Save the plot
    output_image = "soh_vs_cycle.png"
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Plot successfully saved as '{output_image}'.")


def plot_soh_vs_stress():
    # Load analysed_B5.csv and analysed_B6.csv
    b5_path = "analysed_B5.csv"
    b6_path = "analysed_B6.csv"
    b7_path = "analysed_B7.csv"
    
    missing_files = []
    if not os.path.exists(b5_path):
        missing_files.append(b5_path)
    if not os.path.exists(b6_path):
        missing_files.append(b6_path)
    if not os.path.exists(b7_path):
        missing_files.append(b7_path)
        
    if missing_files:
        print(f"Error: Missing files {missing_files}. Please run thermal_stress_model.py first.")
        return
        
    print(f"Loading {b5_path}, {b6_path} and {b7_path}...")
    df_b5 = pd.read_csv(b5_path)
    df_b6 = pd.read_csv(b6_path)
    df_b7 = pd.read_csv(b7_path)
    
    # Combine the dataframes
    df_combined = pd.concat([df_b5, df_b6, df_b7], ignore_index=True)
    
    # Convert cumulative stress to MPa
    df_combined["cumulative_stress_MPa"] = df_combined["cumulative_stress"] / 1e6
    
    # Set up beautiful styling
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 7))
    
    # Plot SOH vs Cumulative Stress with custom styling
    sns.lineplot(
        data=df_combined,
        x="cumulative_stress_MPa",
        y="SOH",
        hue="battery_id",
        palette={"B5": "#D85A38", "B6": "#3A86C8", "B7": "#98C1D9"}, # Specific harmonized/contrasting colors
        linewidth=2.5,
        alpha=0.9
    )
    
    # Enhance the labels and title with modern typography
    plt.title("State of Health (SOH) vs. Cumulative Thermal Stress", fontsize=16, fontweight="bold", pad=20, color="#2C3E50")
    plt.xlabel("Cumulative Thermal Stress (MPa)", fontsize=12, fontweight="medium", color="#34495E")
    plt.ylabel("State of Health (SOH %)", fontsize=12, fontweight="medium", color="#34495E")
    
    # Legend customization
    plt.legend(title="Battery ID", title_fontsize="11", fontsize="10", loc="lower left", frameon=True, facecolor="white", edgecolor="#E2E8F0")
    
    # Adjust layout and grid styling
    plt.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")
    plt.tight_layout()
    
    # Save the plot
    output_image = "soh_vs_cumulative_stress.png"
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Plot successfully saved as '{output_image}'.")


def main():
    plot_soh_vs_cycle()
    plot_soh_vs_stress()

if __name__ == "__main__":
    main()
