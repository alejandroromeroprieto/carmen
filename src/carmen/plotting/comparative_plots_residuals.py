"""
This script creates a multi-panel plot comparing the residuals between CARMEN's emulation
and ESM data.

This code will:
- Load climate model emulation data from .nc files
- Calculate normalized residuals
- Create a multi-panel plot comparing emulated data to the original Earth System Model
  (ESM) data.
"""

import os
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from ..constants import OUTPUT_DIR

# Ignore RuntimeWarning for potential division by zero, which we handle.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*invalid value encountered in divide.*"
)


def generate_residual_plots():
    """
    Loads climate model emulation data from .nc files, calculates normalized
    residuals, and creates a multi-panel plot comparing emulated data to
    the original Earth System Model (ESM) data.

    The plot is organized with scenarios as columns and variables as rows.
    """
    # --- 1. Configuration ---

    folder_path = OUTPUT_DIR

    # Define the structure of the experiment based on your file naming convention
    models = [
        "ACCESS-ESM1-5",
        "BCC-CSM2-MR",
        "CanESM5",
        "CESM2",
        "CMCC-ESM2",
        "CNRM-ESM2-1",
        "GFDL-ESM4",
        "IPSL-CM6A-LR",
        "MIROC-ES2L",
        "MPI-ESM1-2-LR",
        "MRI-ESM2-0",
        "NorESM2-LM",
        "UKESM1-0-LL",
    ]

    # Define the order of scenarios for the plot columns
    scenarios_ordered = [
        "1pctco2",
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
    ]

    # Define the order of variables for the plot rows
    # The string here is the base name, e.g., 'catm' for 'simulated_catm'
    variables_ordered = [
        "catm",
        "cveg",
        "csoil",
        "npp",
        "lit",
        "sres",
    ]

    # Pre-defined friendly names for y-axis labels
    variable_labels = {
        "catm": "Atm. $CO_2$",
        "cveg": "Veg. carbon",
        "csoil": "Soil carbon",
        # 'ocean_carbon_uptake': 'Ocean Carbon Uptake',
        "npp": "NPP",
        "lit": "Litterfall",
        "sres": "Soil respiration",
        # 'oflux': 'Ocean-Air CO2 Flux'
    }

    # --- 2. Data Loading ---

    # Use a dictionary to store loaded datasets for easy access
    all_data = {}
    print(f"Loading data from: {os.path.abspath(folder_path)}")

    if not os.path.isdir(folder_path):
        print(
            f"Error: Folder not found at '{folder_path}'. Please check the 'folder_path' variable."
        )
        return

    for filename in os.listdir(folder_path):
        # Assuming filename format is 'simulation_{model}_{scenario}_npp.nc'
        if filename.startswith("simulation_") and filename.endswith(".nc"):
            try:
                parts = filename.replace(".nc", "").split("_")
                model = parts[1]
                scenario = parts[2]

                # Treat 'esm-ssp534-over' as 'ssp534-over' for consistency
                if scenario == "esm-ssp534-over":
                    scenario = "ssp534-over"

                if model in models and scenario in scenarios_ordered:
                    file_path = os.path.join(folder_path, filename)
                    all_data[(model, scenario)] = xr.open_dataset(file_path)
            except IndexError:
                print(f"Warning: Could not parse filename '{filename}'. Skipping.")

    if not all_data:
        print(
            "Error: No valid .nc files found. Please check the filenames and folder path."
        )
        return

    print(f"Successfully loaded data for {len(all_data)} model-scenario combinations.")

    # --- 3. Plotting Setup ---

    nrows = len(variables_ordered)
    ncols = len(scenarios_ordered)

    # Create a large figure to accommodate all subplots
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3, nrows * 2.5),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0, "wspace": 0},  # No space between plots
    )

    # Create a color map for the models for consistent coloring
    color_map = plt.cm.get_cmap("tab20", len(models))
    model_colors = {model: color_map(i) for i, model in enumerate(models)}

    # --- 4. Main Plotting Loop ---

    for i, var_name in enumerate(variables_ordered):
        for j, scenario_name in enumerate(scenarios_ordered):
            ax = axes[i, j]

            for model_name in models:
                # Check if data for this specific model/scenario combination exists
                if (model_name, scenario_name) in all_data:
                    ds = all_data[(model_name, scenario_name)]

                    sim_var = f"simulated_{var_name}"
                    esm_var = f"esm_{var_name}"

                    # Check if the required variables are in the dataset
                    if sim_var in ds and esm_var in ds:
                        # Calculate normalized residual: (simulated - esm) / esm
                        sim_data = ds[sim_var]
                        esm_data = ds[esm_var]

                        # Calculate raw residual. This may create inf/-inf where esm_data is 0.
                        raw_residual = (sim_data - esm_data) / esm_data

                        # Use xarray's .where() method. This replaces any problematic values
                        # (where the original ESM data was 0) with NaN.
                        residual = raw_residual.where(esm_data != 0)

                        # Plot the time series of the residual
                        if scenario_name == "1pctco2":
                            ax.plot(
                                np.arange(len(residual)),
                                residual,
                                color=model_colors[model_name],
                                label=model_name,
                            )
                        else:
                            ax.plot(
                                ds["time"],
                                residual,
                                color=model_colors[model_name],
                                label=model_name,
                            )

            # --- 5. Subplot Formatting ---

            # Add a horizontal line at y=0 for reference
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, zorder=0)

            # Set column titles on the top row
            if i == 0:
                ax.set_title(scenario_name, fontsize=12, pad=10)

            # Set row labels on the first column
            if j == 0:
                label = variable_labels.get(var_name, var_name)
                ax.set_ylabel(label, fontsize=12, labelpad=10)

            # Turn off tick labels for inner plots to keep it clean
            if i < nrows - 1:
                ax.tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
            if j > 0:
                ax.tick_params(
                    axis="y", which="both", left=False, right=False, labelleft=False
                )

            # Add grid
            ax.grid(True, linestyle=":", alpha=0.6)

    # --- 6. Final Figure Formatting ---

    # Create a single legend for the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),  # Position below the plot
        ncol=min(len(models), 13),  # Adjust number of legend columns
        fontsize=14,
        title_fontsize=16,
        title="Earth System Models",
    )

    # Add a general super-title
    fig.suptitle(
        "Normalized residuals of CARMEN's carbon cycle emulation", fontsize=20, y=0.95
    )

    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust rect to make space for legend and subtitle

    # Save the figure
    output_filename = "emulation_residuals_summary.png"
    plt.savefig(f"{OUTPUT_DIR}/plots/{output_filename}", dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_filename}'")

    plt.show()


if __name__ == "__main__":
    generate_residual_plots()
