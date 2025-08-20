"""
This is just a messy script originally employed by Alex Romero Prieto to create some plots comparing different parameter
values. It is not condidere as part of the model code, and as such it is presented "as is" without the same level of 
clarity and comments as the model code. It is left here for archiving purposes and in case it is useful in the future, but
it can be safely ignored.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
file_paths = [
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-all-ssps_default/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-all-ssps_r2i1p1f2/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-all-ssps_r3i1p1f2/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-all-ssps_r4i1p1f2/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-all-ssps_r8i1p1f2/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-all-ssps_all-realisations/model_pars_UKESM1-0-LL.txt",
]

file_paths_others = [
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/NorESM2-LM/model_pars_NorESM2-LM.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/IPSL-CM6A-LR/model_pars_IPSL-CM6A-LR.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/CanESM5/model_pars_CanESM5.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-esm-scenarios/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-ssp534_plus_ssp585/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-1pctco2/model_pars_UKESM1-0-LL.txt",
    "/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-1pctco2-hist-noLU/model_pars_UKESM1-0-LL.txt",
]

# Load JSON data
data = []
for file in file_paths:
    with open(file, "r") as f:
        content = json.load(f)
        data.append(content)

data_others = []
for file in file_paths_others:
    with open(file, "r") as f:
        content = json.load(f)
        data_others.append(content)

# Convert to DataFrame
df = pd.DataFrame(data)
df_others = pd.DataFrame(data_others)

# Create a combined column for easier visualization in plots
df["model_realisation"] = df["model"] + " - " + df["realisation"]

# Set index to model and realisation
df = df.set_index(["model", "realisation"])

# Select only numeric columns
numeric_cols = df.select_dtypes(include=["number"])

# Compute Mean and Std Dev as separate rows
mean_row = numeric_cols.mean().rename("Mean")
std_row = numeric_cols.std().rename("Std Dev")

# Append to the DataFrame
df_numeric = pd.concat([numeric_cols, mean_row.to_frame().T, std_row.to_frame().T])

# Print table
print(df_numeric)
print(df_others)
# Save table to CSV
# df_numeric.to_excel("model_comparison.xlsx")

import seaborn as sns

magnitude_vars = []
# Separate variables into two groups
magnitude_vars = [col for col in df_numeric.columns if "_t_l" in col]
magnitude_vars += [col for col in df_numeric.columns if "_t_e" in col]
# magnitude_vars += [col for col in df_numeric.columns if "_c_l" in col]
# magnitude_vars += [col for col in df_numeric.columns if "_c_half" in col]
# magnitude_vars = ["docn", "docnfac", "ocntemp", "docntemp"]
# magnitude_vars = ["docntemp"]

marker_size = 200

# Create figure
fig, ax1 = plt.subplots(figsize=(14, 6))

# Boxplot for low-magnitude variables
sns.boxplot(
    data=df_numeric[magnitude_vars][0 : len(file_paths) - 1],
    ax=ax1,
    color="lightblue",
    width=1,
    linewidth=1,
)
ax1.set_ylabel("t_l", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# # Compute means correctly
# means_low = df_numeric[low_magnitude_vars].mean()
# means_high = df_numeric[high_magnitude_vars].mean()

# Add means with correct x positions
ax1.scatter(
    range(len(magnitude_vars)),
    df_numeric[magnitude_vars].loc["Mean"],
    color="red",
    label="Mean",
    zorder=3,
    marker="x",
    s=marker_size,
)

# Add all realisations calibration with correct x positions
ax1.scatter(
    range(len(magnitude_vars)),
    df_numeric[magnitude_vars].iloc[5],
    color="purple",
    label="All realisation calibration",
    zorder=3,
    marker="x",
    s=marker_size,
)

# Add other models
ax1.scatter(
    range(len(magnitude_vars)),
    df_others[magnitude_vars].iloc[0],
    color="olive",
    label="NorESM2-LM",
    zorder=3,
    s=marker_size,
)
ax1.scatter(
    range(len(magnitude_vars)),
    df_others[magnitude_vars].iloc[1],
    color="cyan",
    label="IPSL-CM6A-LR",
    zorder=3,
    s=marker_size,
)
ax1.scatter(
    range(len(magnitude_vars)),
    df_others[magnitude_vars].iloc[2],
    color="brown",
    label="CanESM5",
    zorder=3,
    s=marker_size,
)

# Add emissions scenarios calibration
ax1.scatter(
    range(len(magnitude_vars)),
    df_others[magnitude_vars].iloc[3],
    color="gold",
    label="UKESM-emissions",
    zorder=3,
    marker="*",
    s=marker_size,
)
ax1.scatter(
    range(len(magnitude_vars)),
    df_others[magnitude_vars].iloc[4],
    color="red",
    label="UKESM-ssp534-ssp585",
    zorder=3,
    marker="*",
    s=marker_size,
)
ax1.scatter(
    range(len(magnitude_vars)),
    df_others[magnitude_vars].iloc[5],
    color="red",
    label="UKESM-1pctco2",
    zorder=3,
    marker="+",
    s=marker_size,
)
ax1.scatter(
    range(len(magnitude_vars)),
    df_others[magnitude_vars].iloc[6],
    color="red",
    label="UKESM-1pctco2-hist-noLU",
    zorder=3,
    marker="<",
    s=marker_size,
)

# # Second y-axis for high-magnitude variables
# ax2 = ax1.twinx()
# sns.boxplot(data=df_numeric[high_magnitude_vars], ax=ax2, color="lightcoral")
# ax2.set_ylabel("High-Magnitude Variables", color="darkred")
# ax2.tick_params(axis="y", labelcolor="darkred")

# # Align mean values correctly
# ax2.scatter(range(len(low_magnitude_vars), len(low_magnitude_vars) + len(high_magnitude_vars)), means_high, color="red", label="Mean (High-Mag)", zorder=8)

# # Fix x-axis labels
# ax1.set_xticks(range(len(df_numeric.columns)))
# ax1.set_xticklabels(df_numeric.columns, rotation=45, ha="right")

# Title and legend
plt.title("Distribution of Model Parameters")
fig.legend(loc="upper right")

plt.show()
