
"""
Execute the carbon cycle emulator for a series of scenarios and models.

This code will:
- load the relevant ESM data from the data/scenarios folder.
- load the relevant model parameters from the data/pars folder.
- run the emulator for the same time duration as the ESM data.
- Store results of the emulation in a file for each model and scenario.
"""

import os
from pathlib import Path
import time as systime
import math
import json

import matplotlib.pyplot as plt
import numpy as np

from carmen.utils import load_esm_data, make_all_dirs
from carmen.constants import PARS_DIR, SCEN_DIR, OUTPUT_DIR
from carmen.carmen import CarbonCycle

# --- NEW: Define the path to your specific parameter override files ---
SPECIFIC_PARS_DIR = "my_calibration_results/ocean_sigmoid_depth"

# Smoothing to perform on input ESM data
# smoothing_algorithm = {"type": "butterworth", "pars": [21, 3]} # Uncomment to smooth input data
smoothing_algorithm = {"type": "butterworth", "pars": [1]}  # No smoothing

# Set to true if you want to emulate the NPP flux exclusively, set to false if you want
# to emulate GPP/vegetation respiration fluxes instead
NPP_FLAG = True

# Numerical instability can arise if DT_STEP_OCEAN is too large. We recommend to keep it
# to 0.2 or less.
DT_STEP = 0.05  # main timestep (years) for land carbon cycle
DT_STEP_OCEAN = 0.05  # max timestep (years) for ocean carbon cycle

# Calculate emissions from carbon mass conservation across the different carbon pools
RECOMP_EMIS = True

# List of models and scenarios to emulate
runs = {
    'models': {
        'ACCESS-ESM1-5': ['1pctco2', 'ssp126', 'ssp245', 'ssp370', 'ssp534-over', 'ssp585'],
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
        # 'UKESM1-0-LL': ['1pctco2', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp534-over', 'ssp585']
    }
}

# Dictionary to store emulator results for plotting
emulator_results = {model: {} for model in runs['models']}
esm_data_for_plotting = {model: {} for model in runs['models']}

# Run emulation for each model
for model, scenarios in runs['models'].items():
    for scenario in scenarios:
        # If we are dealing with an SSP scenario, average the first few data
        # points to denoise the first value. However, if dealing with a
        # 1pctco2(-cdr) just take the first value, as subsequent values will
        # already have significantly diverged from equilibrium.
        # How to derive pre-industrial values
        if model in ["BCC-CSM2-MR", "CESM2", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC-ES2L", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM", "UKESM1-0-LL"]:
            if "1pctco2" in scenario:
                PRE_IND_AVERAGE_LENGTH = 1
            elif "ssp" in scenario or "hist" in scenario:
                PRE_IND_AVERAGE_LENGTH = 20
            elif "abrupt" in scenario:
                PRE_IND_AVERAGE_LENGTH = 1
            elif "flat" in scenario:
                PRE_IND_AVERAGE_LENGTH = 1
            else:
                raise ValueError(f"Experiment {scenario} not recognised")

            pre_ind_algorithm = {"type": "average", "length": PRE_IND_AVERAGE_LENGTH}

        # Some models produced better emulation if a denoised timeseries was used
        # to calculate the pre-industrial magnitude of fluxes
        elif model in ["ACCESS-ESM1-5", "CanESM5", "CMCC-ESM2", "CNRM-ESM2-1"]:
            pre_ind_algorithm = {"type": "butterworth", "length": [10]}

        else:
            raise ValueError(f"Model {model} not recognised")

        # Load diagnosed ESM data from the data dir
        # A couple of models showed a pre-industrial start point far from equilibrium,
        # so we detrended that data.
        if model == "CNRM-ESM2-1" or model == "IPSL-CM6A-LR":
            scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
            pars_to_use = PARS_DIR
        else:
            scen_to_use = SCEN_DIR
            pars_to_use = PARS_DIR

        data_file = (
            Path(__file__).parent / scen_to_use / f"sce_{model}_{scenario}.txt"
        )

        print("\nLoading ESM data from: ", data_file)

        esm_data = load_esm_data(
            data_file,
            recalc_emis=RECOMP_EMIS,
            smoothing_pars=smoothing_algorithm,
        )

        # Load calibrated SCC parameter from saved dict.
        pars_file = (
            Path(__file__).parent / pars_to_use / f"sccpar_{model}_cross_experiment.txt"
        )
        print("Loading SCC parameters from: ", pars_file)

        with open(pars_file, "r", encoding="utf-8") as infile:
            scc_pars = json.load(infile)

        # Compute pre-industrial values
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
            # # Independetely from how we smooth the data to be used, make sure the initial values
            # # for fluxes are determined from the smoothed data to avoid effects from unusally
            # # high/low flux values from the ESM
            esm_input_smoothed = load_esm_data(
                data_file, RECOMP_EMIS, smoothing_pars={"type": "butterworth", "pars": [10]}
            )

            scc_pars["npp0"] = esm_input_smoothed.npp[0]
            scc_pars["gpp0"] = esm_input_smoothed.gpp[0]
            scc_pars["lit0"] = esm_input_smoothed.lit[0]
            scc_pars["sres0"] = esm_input_smoothed.rh[0]
            scc_pars["vres0"] = esm_input_smoothed.ra[0]
        else:
            raise ValueError(f"Pre-industrial algorithm {pre_ind_algorithm["type"]} not recognised.")

        # Measure the time it takes us to run the model
        tbeg = systime.time()

        # Build the model and run it
        cc_emulator = CarbonCycle(esm_data, DT_STEP, DT_STEP_OCEAN, NPP_FLAG, **scc_pars)
        cc_emulator.run_full_simulation()

        # Interpolate back to ESM time points, so we can compare results
        cc_emulator.interpolate_results(esm_data.time)

        tend = systime.time()
        print("Time to process:", tend - tbeg)

        # Store results
        CWD = str(Path.cwd())
        out_file = os.path.join(CWD, OUTPUT_DIR)
        if NPP_FLAG:
            out_file += f"/simulation_{model}_{scenario}_npp.nc"
        else:
            out_file += f"/simulation_{model}_{scenario}.nc"
        make_all_dirs(out_file)

        cc_emulator.store_results(model, scenario, scc_pars, out_file)

        emulator_results[model][scenario] = cc_emulator
        esm_data_for_plotting[model][scenario] = esm_data



# --- Plotting Section ---
print("\nGenerating plots...")

# Define the maximum number of columns for scenarios within a model's subplot grid
max_cols_per_model = 3 # You can change this to 2 or more as desired

# Create a list to store all the figure objects, one for each model
figs = []

# Loop through each model to create a separate figure for it
for model, scenarios_data in runs['models'].items():
    scenarios_list = scenarios_data # Get the list of scenarios for the current model
    num_scenarios = len(scenarios_list)

    # Calculate the number of rows needed for scenarios based on max_cols_per_model
    num_rows_for_scenarios = math.ceil(num_scenarios / max_cols_per_model)

    # Create a new figure and a set of subplots for the current model
    # Adjust figsize based on the number of rows/columns
    fig, axes = plt.subplots(num_rows_for_scenarios, max_cols_per_model,
                             figsize=(4 * max_cols_per_model, 3 * num_rows_for_scenarios),
                             squeeze=False) # squeeze=False ensures axes is always 2D

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    fig.suptitle(f'Model: {model} - Ocean Carbon Flux (oflux)', fontsize=16)

    # Loop through each scenario for the current model
    for j, scenario in enumerate(scenarios_list):
        ax = axes[j] # Select the j-th subplot in the flattened array

        # Get the emulator and ESM data for the current model and scenario
        emulator_obj = emulator_results[model].get(scenario)
        esm_obj = esm_data_for_plotting[model].get(scenario)

        if emulator_obj and esm_obj: # Ensure both data types exist for this scenario
            # Plot emulator data
            ax.plot(emulator_obj.time, emulator_obj.ocean.oflux, label='Emulated', color='blue')

            # Plot ESM reference data
            ax.plot(esm_obj.time, esm_obj.oflux, label='ESM Reference', color='orange')

            ax.set_title(f'Scenario: {scenario}')
            ax.set_xlabel('Time (Years)')
            ax.set_ylabel('Ocean Carbon Flux')
            ax.legend()
            ax.grid(True)
        else:
            # If data is missing for a scenario, you might want to hide the subplot
            # or add a "No Data" message.
            fig.delaxes(ax) # Deletes the unused subplot if no data is available
            print(f"Warning: No data found for {model} - {scenario}. Skipping plot for this subplot.")

    # Remove any extra empty subplots if the number of scenarios is not a perfect multiple of max_cols_per_model
    for k in range(num_scenarios, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    figs.append(fig) # Store the figure object

# Display all figures
plt.show()
