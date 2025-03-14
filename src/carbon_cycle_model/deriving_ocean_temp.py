from pathlib import Path
import re
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

from carbon_cycle_model.utils import load_esm_data
from carbon_cycle_model.utils import list_files_containing
from carbon_cycle_model.constants import SCEN_DIR

# Model for which we want to load all scenarios for the fitting
model_to_fit = "NorESM2-LM"


def best_linear_fit_scipy(x_values, y_values):
    """Finds the best linear fit using scipy.stats.linregress."""
    # Perform linear regression
    regression = linregress(x_values, y_values)
    slope, intercept = regression.slope, regression.intercept

    # Print the equation and R-squared value
    print(f"Best-fit line: y = {slope:.3f}x + {intercept:.3f}")
    print(f"R-squared: {regression.rvalue**2:.3f}")

    return slope, intercept, regression.rvalue**2


# Load all scenarios for the given model
if model_to_fit == "CNRM-ESM2-1" or model_to_fit == "IPSL-CM6A-LR":
    scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
else:
    scen_to_use = SCEN_DIR

data_dir = (Path(__file__).parent / scen_to_use)

data_files = list_files_containing(data_dir, model_to_fit)

# Load all GMST and SST from each data file.
data_dict = {}
for data_file in data_files:
    # get the scenario from the file name
    scenario = re.match(rf"sce_{re.escape(model_to_fit)}_(.+?)\.txt$", data_file).group(1)

    # load scenario
    esm_data = load_esm_data(
        data_dir / data_file,
        recalc_emis=True,
        ninit=20,  # Because it's an SSP scenario
        smoothing_pars={"type": "butterworth", "pars": [1]},  # no smoothing
    )

    data_dict.update({scenario: {"general_temperature": np.array(esm_data.dtglb), "ocean_temperature": np.array(esm_data.dtocn)}})


general_temperature_list = []
ocean_temperature_list = []
for scenario, vals in data_dict.items():
    # plt.scatter(vals["general_temperature"], vals["ocean_temperature"], label=scenario)
    general_temperature_list.append(vals["general_temperature"])
    ocean_temperature_list.append(vals["ocean_temperature"])

# plt.legend()
# plt.show()

# plt.scatter(general_temperature_list, ocean_temperature_list)
# plt.show()

# Convert to NumPy arrays and flatten them
general_temperature_array = np.concatenate(general_temperature_list).ravel()
ocean_temperature_array = np.concatenate(ocean_temperature_list).ravel()

slope, intercept, r_squared = best_linear_fit_scipy(general_temperature_array, ocean_temperature_array)

# Generate fitted y values
fitted_y = slope * general_temperature_array + intercept

# Plot data points
plt.scatter(general_temperature_array, ocean_temperature_array, label="Data points", alpha=0.5)

# Plot best-fit line
plt.plot(general_temperature_array, fitted_y, color="red", label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")

# Labels and title
plt.xlabel("Entire Surface Temperature")
plt.ylabel("Ocean Temperature")
plt.title(f"Best Linear Fit for {model_to_fit}")

plt.text(max(general_temperature_array)*0.75, max(fitted_y)*0.05, f"R² = {r_squared:.3f}", fontsize=12, color="black",
            verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

plt.legend(loc="upper left")

# Show plot
plt.show()