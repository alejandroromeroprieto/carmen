"""
This script creates a plot with two sets of NRMSE between two datasets: CARMEN's
emulation and ESM reference data.

It will:
- Load all data from CARMEN's emulation and ESM simulations.
- Calculate the RMSE between CARMEN's emulation and the target ESM simulation
  CARMEN is attempting to emulate (for each scenario and variable), normalised by
  the range (max val minus min val) of the ESM data. This is aiming to give an idea
  of the quality of CARMEN's emulation.
- Calculate the RMSE between the target ESM simulation CARMEN is attempting to
  emulate (for each scenario and variable), and all other ESM timeseries for that
  specific scenario and variable, normalised by the range (max val minus min val)
  of the ESM data. This is aming to give an idea of how close simulations across
  ESM are for each scenario and variable.
- Create a bar plot for each scenario and variable, with two bars: one presenting
  the NRMSE of CARMEN's emulation (see second bullet point), and the other one
  presenting the minimum NRMSE across all NRMSE from ESM simulations (see second
  third bullet point). The idea is to demonstrate that CARMEN's emulation is typically
  closer than any other ESM simulation by shoing the corresponding bar is lower,
  hsoing the model's ability to emulate specific ESM carbon cycle behaviour.
"""
import os
import warnings

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..constants import OUTPUT_DIR

# Ignore warnings that may arise during NRMSE calculation
warnings.filterwarnings("ignore", category=RuntimeWarning)


def calculate_nrmse(data1, data2):
    """
    Calculates RMSE normalised by the difference betweent the maximum and minimum values
    of the data in data2 (ESM data) over the overlapping portion of two 1D DataArrays,
    aligned by index position, ignoring time coordinate values.
    """
    len1 = data1.shape[0]
    len2 = data2.shape[0]
    min_len = min(len1, len2)

    if min_len == 0:
        return np.nan

    d1 = data1.isel(time=slice(0, min_len)).values
    d2 = data2.isel(time=slice(0, min_len)).values
    # print(len1, len2, d1, d2)
    valid_indices = ~np.isnan(d1) & ~np.isnan(d2)
    d1_valid, d2_valid = d1[valid_indices], d2[valid_indices]

    if d2_valid.size == 0:
        return np.nan

    squared_error = (d1_valid - d2_valid) ** 2
    rmse = np.sqrt(squared_error.mean())
    mean_ref = d2_valid.mean()

    # return rmse / abs(mean_ref) if abs(mean_ref) > 1e-9 else np.nan
    return rmse / (max(d2_valid) - min(d2_valid)) if abs(mean_ref) > 1e-9 else np.nan


def generate_comparison_plots():
    """
    Generates a comprehensive comparison plot showing both emulator error (NRMSE)
    and the structural uncertainty between ESMs (average inter-model NRMSE).
    """
    # --- 1. Configuration ---
    # Specify folder where data is stored
    folder_path = OUTPUT_DIR
    models = [
        "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CESM2", "CMCC-ESM2",
        "CNRM-ESM2-1", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC-ES2L",
        "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM", "UKESM1-0-LL"
    ]
    scenarios_ordered = [
        "1pctco2", "ssp119", "ssp126", "ssp245", "ssp370",
        "ssp434", "ssp460", "ssp534-over", "ssp585"
    ]
    variables_ordered = [
        'catm', 'cveg', 'csoil', 'ocean_carbon_uptake',
        'npp', 'lit', 'sres', 'oflux'
    ]
    variable_short_labels = {
        'catm': 'Atm. C.', 'cveg': 'Veg. C.', 'csoil': 'Soil C.', 'ocean_carbon_uptake': 'Ocean C.',
        'npp': 'NPP', 'lit': 'Litterfall', 'sres': 'Soil resp.', 'oflux': 'Ocean flux'
    }

    # --- 2. Data Loading & Pre-computation ---
    print("Step 1: Loading all ESM data...")
    all_esm_data = {m: {} for m in models}
    all_sim_data = {m: {} for m in models}

    for model_name in models:
        for scenario_name in scenarios_ordered:
            filename = f"simulation_{model_name}_{scenario_name}_npp.nc"
            if scenario_name == 'ssp534-over':
                alt_filename = f"simulation_{model_name}_esm-ssp534-over_npp.nc"
                if os.path.exists(os.path.join(folder_path, alt_filename)):
                    filename = alt_filename

            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                all_esm_data[model_name][scenario_name] = ds
                all_sim_data[model_name][scenario_name] = ds
    print("Step 1 complete.")

    print("Step 2: Pre-computing all NRMSE values...")
    emulation_nrmse = {m: {} for m in models}
    inter_model_nrmse = {m: {} for m in models}

    for model_name in models:
        for scenario_name in scenarios_ordered:
            if scenario_name not in all_esm_data[model_name]:
                continue

            # A. Calculate Emulation NRMSE (Emulator vs. Reference ESM)
            emulation_nrmse[model_name][scenario_name] = {}
            ds_ref = all_sim_data[model_name][scenario_name]
            for var in variables_ordered:
                sim_var, esm_var = f'simulated_{var}', f'esm_{var}'
                if sim_var in ds_ref and esm_var in ds_ref:
                    emulation_nrmse[model_name][scenario_name][var] = calculate_nrmse(ds_ref[sim_var], ds_ref[esm_var])
                else:
                    emulation_nrmse[model_name][scenario_name][var] = np.nan

            # B. Calculate Inter-Model NRMSE (Reference ESM vs. All Other ESMs)
            inter_model_nrmse[model_name][scenario_name] = {}
            for var in variables_ordered:
                nrmse_list = []
                ref_data = all_esm_data[model_name][scenario_name][f'esm_{var}']
                for other_model in models:
                    if model_name == other_model or scenario_name not in all_esm_data[other_model]:
                        continue

                    comp_data = all_esm_data[other_model][scenario_name][f'esm_{var}']
                    # print(other_model, scenario_name)
                    nrmse_list.append(calculate_nrmse(comp_data, ref_data))

                inter_model_nrmse[model_name][scenario_name][var] = {
                    'mean': np.nanmean(nrmse_list),
                    'std': np.nanstd(nrmse_list),
                    'min': np.nanmin(nrmse_list)
                }
    print("Step 2 complete.")

    # --- 3. Plotting ---
    print("Step 3: Building plot grid...")
    nrows, ncols = len(models) + 1, len(scenarios_ordered) + 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.8, nrows * 2.2),
        sharex=True, sharey=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}
    )

    cmap = plt.cm.get_cmap('Dark2', len(variables_ordered))
    variable_colors = {var: cmap(i) for i, var in enumerate(variables_ordered)}
    bar_positions = np.arange(len(variables_ordered))

    ylim_max = 0.4

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            is_model_avg_col = j == ncols - 1
            is_scenario_avg_row = i == nrows - 1

            # --- Formatting ---
            if i == 0:
                title = scenarios_ordered[j] if not is_model_avg_col else "Model Average"
                ax.set_title(title, fontsize=20, pad=12)
            if j == 0:
                label = models[i] if not is_scenario_avg_row else "Scenario\nAverage"
                ax.set_ylabel(label, fontsize=20, labelpad=10, rotation=0, ha='right', va='center')
            if is_scenario_avg_row:
                labels = [variable_short_labels[v] for v in variables_ordered]
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
            else:
                # Hide x-ticks and labels for all other rows
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if j > 0:
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            ax.set_ylim(0, ylim_max + 0.05)
            ax.tick_params(axis='y', labelsize=15)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

            # --- Data Gathering for the subplot ---
            emu_values_to_plot = []
            inter_model_mins_to_plot = []

            if not is_scenario_avg_row and not is_model_avg_col:
                model_name, scenario_name = models[i], scenarios_ordered[j]
                if scenario_name in emulation_nrmse.get(model_name, {}):
                    emu_values_to_plot = list(emulation_nrmse[model_name][scenario_name].values())
                    inter_model_mins_to_plot = [d['min'] for d in inter_model_nrmse[model_name][scenario_name].values()]

            elif not is_scenario_avg_row and is_model_avg_col:
                model_name = models[i]
                all_emu = [list(emulation_nrmse[model_name][s].values()) for s in scenarios_ordered if s in emulation_nrmse[model_name]]
                all_inter = [[d['min'] for d in inter_model_nrmse[model_name][s].values()] for s in scenarios_ordered if s in inter_model_nrmse[model_name]]
                if all_emu:
                    emu_values_to_plot = np.nanmean(np.array(all_emu), axis=0)
                    inter_model_mins_to_plot = np.nanmean(np.array(all_inter), axis=0)
                ax.set_facecolor('whitesmoke')

            elif is_scenario_avg_row and not is_model_avg_col:
                scenario_name = scenarios_ordered[j]
                all_emu = [list(emulation_nrmse[m][scenario_name].values()) for m in models if scenario_name in emulation_nrmse.get(m, {})]
                all_inter = [[d['min'] for d in inter_model_nrmse[m][scenario_name].values()] for m in models if scenario_name in inter_model_nrmse.get(m, {})]
                if all_emu:
                    emu_values_to_plot = np.nanmean(np.array(all_emu), axis=0)
                    inter_model_mins_to_plot = np.nanmean(np.array(all_inter), axis=0)
                ax.set_facecolor('whitesmoke')

            elif is_scenario_avg_row and is_model_avg_col:
                all_emu = [list(emulation_nrmse[m][s].values()) for m in models for s in scenarios_ordered if s in emulation_nrmse.get(m, {})]
                all_inter = [[d['min'] for d in inter_model_nrmse[m][s].values()] for m in models for s in scenarios_ordered if s in inter_model_nrmse.get(m, {})]
                if all_emu:
                    emu_values_to_plot = np.nanmean(np.array(all_emu), axis=0)
                    inter_model_mins_to_plot = np.nanmean(np.array(all_inter), axis=0)
                ax.set_facecolor('gainsboro')

            # --- Plotting on the axis ---
            if len(emu_values_to_plot) == 0:
                ax.axis('off')
                continue

            # Plot Background Bars (Inter-model spread)
            bars = ax.bar(bar_positions, inter_model_mins_to_plot, width=0.8, color='grey', alpha=0.5, zorder=1)
            for bar in bars:
                bar_height = bar.get_height()
                if not np.isnan(bar_height) and bar_height > ylim_max:
                    bar.set_height(ylim_max)
                    ax.annotate('↑', (bar.get_x() + bar.get_width() / 2, ylim_max * 0.98),
                                ha='center', va='bottom', color='red', fontsize=12, weight='bold')

            # Plot Foreground Bars (Emulator error)
            ax.bar(bar_positions, emu_values_to_plot, width=0.6, color=[variable_colors[v] for v in variables_ordered], zorder=3)

    # --- 4. Final Figure Formatting & Legend ---
    fig.suptitle('Normalized root mean squared error (NRMSE) of CARMEN\'s carbon cycle emulation vs. Inter-Model minimum NRSME', fontsize=30, y=0.99)
    fig.supylabel('NRMSE (RMSE / range)', fontsize=30, x=0.01)

    # Create custom legend
    # Legend handles for each variable
    variable_patches = [
        mpatches.Patch(color=variable_colors[var], label=variable_short_labels[var])
        for var in variables_ordered
    ]

    # Inter-model spread patch
    inter_model_patch = mpatches.Patch(color='grey', alpha=0.5, label='Minimum inter-model NRMSE (Reference vs. Others)')

    # Combine patches
    legend_handles = variable_patches + [inter_model_patch]

    fig.legend(handles=legend_handles, loc='lower center',
            bbox_to_anchor=(0.5, -0.01), ncol=9, fontsize=20)

    # plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.05, wspace=0.0, hspace=0.0)

    output_filename = 'emulation_vs_intermodel_nrmse_min.png'
    plt.savefig(f"{OUTPUT_DIR}/plots/{output_filename}", dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_filename}'")
    # plt.show()


if __name__ == '__main__':
    generate_comparison_plots()
