# NOTE: THIS IS JUST SOME MESSY, AUXILIARY CODE THAT ALEX ROMERO PRIETO WROTE TO CREATE SOME PLOTS TO COMPARE PARAMETER VALUES
# IT IS NOT CONSIDERED PART OF THE MODEL, AND PRESENTED "AS IS" IN CASE IT IS USEFUL IN THE FUTURE, AND FOR ARCHIVE PURPOSES.

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Define file paths
smoothing_type = "no-smoothing"
file_paths_others = [
    f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-1pctco2/model_pars_UKESM1-0-LL.txt",
    f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-1pctco2-hist_noLu/model_pars_UKESM1-0-LL.txt",
    # f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-historical/model_pars_UKESM1-0-LL.txt",
    # f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-esm-historical/model_pars_UKESM1-0-LL.txt",
    f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-hist_noLu/model_pars_UKESM1-0-LL.txt",
    f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-allssps_default/model_pars_UKESM1-0-LL.txt",
    f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-allssps-hist_noLU/model_pars_UKESM1-0-LL.txt",
    f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-allssps-1pctco2/model_pars_UKESM1-0-LL.txt",
    f"/home/eearp/code/carbon-cycle-model/src/carbon_cycle_model/my_calibration_results/UKESM-historical/model_pars_UKESM1-0-LL.txt",
]

# Load JSON data

data_others = []
for file in file_paths_others:
    with open(file.format(smoothing_type=smoothing_type), "r") as f:
        content = json.load(f)
        data_others.append(content)

# Convert to DataFrame
df_others = pd.DataFrame(data_others)

import seaborn as sns

magnitude_vars = []
# Separate variables into two groups
# magnitude_vars = [col for col in df_others.columns if "_t_l" in col]
# magnitude_vars += [col for col in df_others.columns if "_t_e" in col]
# magnitude_vars += [col for col in df_others.columns if "_c_l" in col]
# magnitude_vars += [col for col in df_others.columns if "_c_half" in col]
# magnitude_vars = ["docn", "docnfac", "ocntemp", "docntemp"]
magnitude_vars = ["docntemp"]

# plot: 1pctco2, hist, hist-noLU, is hist-noLU closer to 1pctco2 or hist

marker_size = 200

# Create figure
fig, ax1 = plt.subplots(figsize=(14, 6))

# # Compute means correctly
# means_low = df_numeric[low_magnitude_vars].mean()
# means_high = df_numeric[high_magnitude_vars].mean()

# Add other models
# ax1.set_x(magnitude_vars)
# ax1.set_ylim(-0.2, 0.2)
ax1.set_ylabel("par_value")
ax1.scatter(
    magnitude_vars,
    df_others[magnitude_vars].iloc[0],
    color="red",
    label="1pctco2",
    zorder=3,
    s=marker_size,
    marker="+",
)
ax1.scatter(
    magnitude_vars,
    df_others[magnitude_vars].iloc[1],
    color="red",
    label="1pctco2+hist-noLU",
    zorder=3,
    s=marker_size,
    marker=".",
)
# ax1.scatter(magnitude_vars, df_others[magnitude_vars].iloc[2], color="blue", label="historical", zorder=3, s=marker_size, marker="x")
# ax1.scatter(magnitude_vars, df_others[magnitude_vars].iloc[3], color="orange", label="esm-historical", zorder=3, s=marker_size, marker="1")
ax1.scatter(
    magnitude_vars,
    df_others[magnitude_vars].iloc[2],
    color="red",
    label="hist-noLU",
    zorder=3,
    s=marker_size,
    marker="2",
)
ax1.scatter(
    magnitude_vars,
    df_others[magnitude_vars].iloc[3],
    color="blue",
    label="all-ssps",
    zorder=3,
    s=marker_size,
    marker="3",
)
ax1.scatter(
    magnitude_vars,
    df_others[magnitude_vars].iloc[4],
    color="blue",
    label="all-ssps-hist-nolu",
    zorder=3,
    s=marker_size,
    marker="4",
)
ax1.scatter(
    magnitude_vars,
    df_others[magnitude_vars].iloc[5],
    color="blue",
    label="allssps-1pctco2",
    zorder=3,
    s=marker_size,
    marker="x",
)
ax1.scatter(
    magnitude_vars,
    df_others[magnitude_vars].iloc[5],
    color="green",
    label="historical",
    zorder=3,
    s=marker_size,
    marker="+",
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
plt.title(f"Distribution of Model Parameters - {smoothing_type}")
fig.legend(loc="upper right")

plt.show()
