"""
Execute the carbon cycle emulator for a series of scenarios and models,
and visualize the results in a grid of subplots for each model.

This code will:
- load the relevant ESM data from the data/scenarios folder.
- load the relevant model parameters from the data/pars folder.
- run the emulator for the same time duration as the ESM data.
- Store results of the emulation in a file for each model and scenario.
- Generate plots showing emulated vs. ESM data for multiple variables
  across different scenarios, organized by model.
"""

import json
import os
from pathlib import Path
import time as systime

import matplotlib.pyplot as plt
import numpy as np

from carmen.utils import load_esm_data, make_all_dirs
from carmen.constants import PARS_DIR, SCEN_DIR, OUTPUT_DIR
from carmen.carmen import CarbonCycle

# Smoothing to perform on input ESM data
# smoothing_algorithm = {"type": "butterworth", "pars": [21, 3]}
smoothing_algorithm = {"type": "butterworth", "pars": [1]}  # No smoothing

NPP_FLAG = True

# Numerical instability can arise if dtoccmax is too large. We recommend to keep it
# to 0.2 or less.
dtpred = 0.05  # main timestep (years) for land carbon cycle
dtoccmax = 0.05  # max timestep (years) for ocean carbon cycle

# Calculate emissions from carbon mass conservation across the different carbon pools
recalc_emis = True

# Define the models and their respective scenarios
runs = {
    'models': {
        # 'ACCESS-ESM1-5': ['1pctco2', 'ssp126', 'ssp245', 'ssp370', 'ssp534-over', 'ssp585'],
        # 'BCC-CSM2-MR': ['1pctco2', 'ssp126', 'ssp245', 'ssp370', 'ssp585'],
        # 'CanESM5': ['1pctco2', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585'],
        # 'CESM2': ['1pctco2', 'ssp126', 'ssp245', 'ssp370', 'ssp585'],
        # 'CMCC-ESM2': ['1pctco2', 'ssp126', 'ssp245', 'ssp370', 'ssp534-over', 'ssp585'],
        # 'CNRM-ESM2-1': ['1pctco2', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585'],
        # 'GFDL-ESM4': ['1pctco2', 'ssp126', 'ssp370'],
        # 'IPSL-CM6A-LR': ['1pctco2', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585'],
        # 'MIROC-ES2L': ['1pctco2', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp534-over', 'ssp585'],
        # 'MPI-ESM1-2-LR': ['1pctco2', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'esm-ssp534-over', 'ssp585'],
        # 'MRI-ESM2-0': ['1pctco2', 'ssp585'],
        # 'NorESM2-LM': ['1pctco2', 'ssp126', 'ssp245', 'ssp370', 'ssp534-over', 'ssp585'],
        'UKESM1-0-LL': ['1pctco2', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp534-over', 'ssp585']
    }
}

# Dictionary to store emulator results for plotting
emulator_results = {model: {} for model in runs['models']}
esm_data_for_plotting = {model: {} for model in runs['models']}

# Run emulation for each model and scenario
for model, scenarios in runs['models'].items():
    for scenario in scenarios:
        # Determine pre-industrial averaging length based on model and scenario type
        if model in ["BCC-CSM2-MR", "CESM2", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC-ES2L", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM", "UKESM1-0-LL"]:
            if "1pctco2" in scenario or "abrupt" in scenario or "flat" in scenario:
                PRE_IND_AVERAGE_LENGTH = 1
            elif "ssp" in scenario or "hist" in scenario:
                PRE_IND_AVERAGE_LENGTH = 20
            else:
                raise ValueError(f"Experiment {scenario} not recognised for model {model}")
            pre_ind_algorithm = {"type": "average", "length": PRE_IND_AVERAGE_LENGTH}
        elif model in ["ACCESS-ESM1-5", "CanESM5", "CMCC-ESM2", "CNRM-ESM2-1"]:
            pre_ind_algorithm = {"type": "butterworth", "length": [10]}
        else:
            raise ValueError(f"Model {model} not recognised")

        # Define paths for ESM data and SCC parameters
        scen_to_use = SCEN_DIR + "/detrended_wrt_decade" if model in ["CNRM-ESM2-1", "IPSL-CM6A-LR"] else SCEN_DIR
        pars_to_use = PARS_DIR

        data_file = Path(__file__).parent.parent / scen_to_use / f"sce_{model}_{scenario}.txt"
        print(f"\nLoading ESM data from: {data_file}")

        esm_data = load_esm_data(
            data_file,
            recalc_emis=recalc_emis,
            smoothing_pars=smoothing_algorithm,
        )

        pars_file = Path(__file__).parent.parent / pars_to_use / f"sccpar_{model}_cross_experiment.txt"
        print(f"Loading SCC parameters from: {pars_file}")

        with open(pars_file, "r", encoding="utf-8") as infile:
            scc_pars = json.load(infile)

        # Record pre-industrial values
        scc_pars["cveg0"] = esm_data.cveg[0]
        scc_pars["csoil0"] = esm_data.csoil[0]
        scc_pars["catm0"] = esm_data.catm[0]

        if pre_ind_algorithm["type"] == "average":
            scc_pars["npp0"] = np.mean(esm_data.npp[0:pre_ind_algorithm["length"]])
            scc_pars["gpp0"] = np.mean(esm_data.gpp[0:pre_ind_algorithm["length"]])
            scc_pars["lit0"] = np.mean(esm_data.lit[0:pre_ind_algorithm["length"]])
            scc_pars["sres0"] = np.mean(esm_data.rh[0:pre_ind_algorithm["length"]])
            scc_pars["vres0"] = np.mean(esm_data.ra[0:pre_ind_algorithm["length"]])
        elif pre_ind_algorithm["type"] == "butterworth":
            esm_input_smoothed = load_esm_data(
                data_file, recalc_emis, smoothing_pars={"type": "butterworth", "pars": [10]}
            )
            scc_pars["npp0"] = esm_input_smoothed.npp[0]
            scc_pars["gpp0"] = esm_input_smoothed.gpp[0]
            scc_pars["lit0"] = esm_input_smoothed.lit[0]
            scc_pars["sres0"] = esm_input_smoothed.rh[0]
            scc_pars["vres0"] = esm_input_smoothed.ra[0]
        else:
            raise ValueError(f"Pre-industrial algorithm {pre_ind_algorithm['type']} not recognised.")

        # Run the carbon cycle emulator
        tbeg = systime.time()
        cc_emulator = CarbonCycle(esm_data, dtpred, dtoccmax, NPP_FLAG, **scc_pars)
        cc_emulator.run_full_simulation()
        cc_emulator.interpolate_results(esm_data.time)
        tend = systime.time()
        print(f"Time to process {model}-{scenario}: {tend - tbeg:.2f} seconds")

        # Store results
        CWD = str(Path.cwd())
        out_file_path = os.path.join(CWD, OUTPUT_DIR)
        out_file_path += f"/simulation_{model}_{scenario}_npp.nc" if NPP_FLAG else f"/simulation_{model}_{scenario}.nc"
        make_all_dirs(out_file_path)
        cc_emulator.store_results(model, scenario, scc_pars, out_file_path)

        emulator_results[model][scenario] = cc_emulator
        esm_data_for_plotting[model][scenario] = esm_data


# --- Plotting Section ---
print("\nGenerating plots for all variables and scenarios...")

# Define the variables to plot, their display names, and units
plot_variables = [
    ("catm", "Atm. concentration", "ppm"),
    ("land.cveg", "Veg. stock", "GtC"),
    ("land.csoil", "Soil stock", "GtC"),
    ("ocean.carbon_increase", "Ocean uptake", "GtC"),
    ("land.npp", "NPP", "GtC/year"),
    ("land.lit", "Litterfall", "GtC/year"),
    ("land.sres", "Soil respiration", "GtC/year"),
    ("ocean.oflux", "Ocean-atm. flux", "GtC/year"),
]


def get_nested_attr(obj, attr_path):
    """
    Get nested attributes from an object (e.g., 'obj.subobj.attr')
    """
    parts = attr_path.split('.')
    current = obj
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None  # Return None if any part of the path is missing
    return current


# Compute global y-limits for each variable across all scenarios
y_limits = {}
emu_data = None
for var_path, _, _ in plot_variables:
    all_vals = []
    for model in runs["models"]:
        for scenario in runs["models"][model]:
            emulator_obj = emulator_results[model].get(scenario)
            esm_obj = esm_data_for_plotting[model].get(scenario)
            if emulator_obj and esm_obj:
                emu_data = get_nested_attr(emulator_obj, var_path)
                if var_path == "ocean.carbon_increase":
                    esm_data = np.cumsum(esm_obj.oflux)
                elif var_path == "land.sres":
                    esm_data = esm_obj.rh
                else:
                    esm_data = get_nested_attr(esm_obj, var_path)

            if emu_data is not None:
                all_vals.extend(emu_data)
            if esm_data is not None:
                all_vals.extend(esm_data)

    if any(x in var_path for x in ["cveg", "csoil", "carbon_increase"]):
        y_limits[var_path] = (np.min(all_vals)-50, np.max(all_vals)+50)
    elif any(x in var_path for x in ["npp", "lit", "sres"]):
        y_limits[var_path] = (np.min(all_vals)-5, np.max(all_vals)+5)
    elif "oflux" in var_path:
        y_limits[var_path] = (np.min(all_vals)-1, np.max(all_vals)+1)
    elif "catm" in var_path:
        y_limits[var_path] = (np.min(all_vals)-10, np.max(all_vals)+50)
    else:
        print("WARNING: unexpcted plotting variable", var_path)

# Create a list to store all the figure objects, one for each model
figs = []

# Loop through each model to create a separate figure for it
for model, scenarios_data in runs['models'].items():
    scenarios_list = scenarios_data  # Get the list of scenarios for the current model
    num_scenarios = len(scenarios_list)
    num_variables = len(plot_variables)

    # Calculate the number of rows (variables) and columns (scenarios)
    num_rows = num_variables
    num_cols = num_scenarios

    # Adjust figsize dynamically based on the number of rows/columns
    # This helps ensure plots are readable, especially with many subplots.
    fig_width = 3 * num_cols  # Further reduced width per column
    fig_height = 2.5 * num_rows  # Further reduced height per row
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(fig_width, fig_height),
                             squeeze=False) # squeeze=False ensures axes is always 2D, even for 1x1

    fig.suptitle(f'Model: {model} - Carbon Cycle Variables Comparison', fontsize=25, y=0.98) # y for suptitle position

    # Adjust the spacing between subplots to zero and manually set overall figure margins
    # This replaces plt.tight_layout() for precise control over zero spacing.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, wspace=0.0, hspace=0.0)

    # Loop through each variable (which forms a row in the subplot grid)
    for i, (var_path, var_title, var_unit) in enumerate(plot_variables):
        # Loop through each scenario (which forms a column in the subplot grid)
        for j, scenario in enumerate(scenarios_list):
            ax = axes[i, j] # Select the specific subplot for the current variable-scenario combination

            # Set y-axis limits based on precomputed global range for this variable
            if var_path in y_limits:
                ax.set_ylim(y_limits[var_path])

            # Get the emulator and ESM data objects for the current model and scenario
            emulator_obj = emulator_results[model].get(scenario)
            esm_obj = esm_data_for_plotting[model].get(scenario)

            if emulator_obj and esm_obj: # Check if data exists for this model-scenario pair
                # Retrieve the specific variable data using the helper function
                emulator_data = get_nested_attr(emulator_obj, var_path)

                # Special handling for ESM data based on var_path
                if var_path == "ocean.carbon_increase":
                    esm_data = np.cumsum(esm_obj.oflux)
                elif var_path == "land.sres":
                    esm_data = esm_obj.rh
                else:
                    # For other variables, try to get the attribute directly from esm_obj or its sub-objects
                    esm_data = getattr(esm_obj, var_path.split('.')[-1], None)
                    if esm_data is None and '.' in var_path: # Try nested access for ESM if direct fails
                        esm_data = get_nested_attr(esm_obj, var_path)

                if emulator_data is not None and esm_data is not None:
                    # Plot emulator data
                    ax.plot(emulator_obj.time, emulator_data, label='Emulated', color='blue', linewidth=1.5)
                    # Plot ESM reference data
                    ax.plot(esm_obj.time, esm_data, label='ESM Reference', color='orange', linewidth=1.5)

                    # Set title for each column (scenario) only on the top row
                    if i == 0:
                        if "ssp" in scenario:
                            scenario_title = "hist+" + scenario
                        else:
                            scenario_title = scenario
                        ax.set_title(f'{scenario_title}', fontsize=18)

                    # Set y-label for each row (variable) only on the leftmost column
                    if j == 0:
                        ax.set_ylabel(f'{var_title}\n({var_unit})', fontsize=15)
                        ax.tick_params(axis='y', which='major', labelsize=7) # Show y-ticks for first column
                    else:
                        # Hide y-axis labels and ticks for all but the leftmost column
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', length=0)

                    # Set x-label only for the bottom row
                    if i == num_rows - 1:
                        ax.set_xlabel('Time (Years)', fontsize=15)
                        ax.tick_params(axis='x', which='major', labelsize=7) # Show x-ticks for bottom row
                    else:
                        # Hide x-axis labels and ticks for all but the bottom row
                        ax.set_xticklabels([])
                        ax.tick_params(axis='x', length=0)

                    ax.grid(True, linestyle=':', alpha=0.6)
                    # Ensure borders are visible
                    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')

                    # Add legend to the bottom-right subplot of the entire figure for clarity
                    if i == num_rows - 1 and j == num_cols - 1:
                        #  ax.legend(loc='best', fontsize=8)
                        handles, labels = ax.get_legend_handles_labels()
                else:
                    # Case: specific variable data is missing for this scenario
                    ax.text(0.5, 0.5, 'Variable Data Missing', horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes, color='red', fontsize=9)
                    # Still show scenario/variable titles even if data is missing for context
                    if i == 0:
                        ax.set_title(f'{scenario}', fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f'{var_title}\n({var_unit})', fontsize=9)
                        ax.tick_params(axis='y', which='major', labelsize=7)
                    else:
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', length=0)
                    if i == num_rows - 1:
                        ax.set_xlabel('Time (Years)', fontsize=9)
                        ax.tick_params(axis='x', which='major', labelsize=7)
                    else:
                        ax.set_xticklabels([])
                        ax.tick_params(axis='x', length=0)
                    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in') # Ensure borders
            else:
                # Case: No emulator or ESM data found for the current model-scenario combination
                ax.text(0.5, 0.5, 'No Data for Scenario', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, color='red', fontsize=9)
                if i == 0:
                    ax.set_title(f'{scenario}', fontsize=10)
                # Keep only outer axis lines for visual separation even without data
                if j == 0:
                    ax.set_ylabel(f'{var_title}\n({var_unit})', fontsize=9)
                    ax.tick_params(axis='y', which='major', labelsize=7)
                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', length=0)
                if i == num_rows - 1:
                    ax.set_xlabel('Time (Years)', fontsize=9)
                    ax.tick_params(axis='x', which='major', labelsize=7)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', length=0)
                ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in') # Ensure borders

    figs.append(fig)  # Store the figure object

    # Place one legend below all subplots
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.01), fontsize=20)

output_filename = f'carmem_emulation_{model}_all_scenarios.png'
plt.savefig(f"{OUTPUT_DIR}/plots/esm_emulation/{output_filename}", dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{output_filename} in {OUTPUT_DIR}/plots/esm_emulation/'")
# Uncomment if you want to print to screen the output figure
# plt.show()
