"""
This script loads data from CARMEN's emulation into a dataframe for potential subsequent
analysis.
"""

import os
from pathlib import Path
import re
import xarray as xr
import pandas as pd

from carbon_cycle_model.constants import OUTPUT_DIR

# Folder with your NetCDF files
CWD = str(Path.cwd())
out_folder = os.path.join(CWD, OUTPUT_DIR)

# Regex to parse filenames
# Matches: simulation_<MODEL>_<SCENARIO>.nc or simulation_<MODEL>_<SCENARIO>_npp.nc
pattern = re.compile(r'simulation_(.+?)_(.+?)(?:_npp)?\.nc')

# Prepare a list to store runs
runs = []

# Loop over files
for filename in os.listdir(out_folder):
    if not filename.endswith('.nc'):
        continue

    match = pattern.match(filename)
    if not match:
        print(f"Skipping unrecognized file: {filename}")
        continue

    model, scenario = match.groups()
    RUN_TYPE = 'npp' if '_npp' in filename else 'default'

    # Load NetCDF file
    ds = xr.open_dataset(os.path.join(out_folder, filename))

    # Example: select a few key variables to extract as time series
    # You can expand this list as needed
    variables_of_interest = [
        # Atmospheric CO2
        'esm_catm',
        'simulated_catm',

        # Land box variables
        'esm_cveg',
        'simulated_cveg',

        'esm_csoil',
        'simulated_csoil',

        'esm_npp',
        'simulated_npp',

        'esm_lit',
        'simulated_lit',

        'esm_sres',
        'simulated_sres',

        'esm_fcva',
        'esm_fcsa',
        'esm_fcvs',

        # Ocean box
        'esm_ocean_carbon_uptake',
        'simulated_ocean_carbon_uptake',

        'esm_oflux',
        'simulated_oflux',

        # General carbon cycle box
        'esm_emissions',
        'simulated_emissions',

        # Conditionally present if not self.npp_flag
        'esm_gpp',
        'simulated_gpp',

        'esm_vres',
        'simulated_vres',
    ]

    # Extract the time array (assuming all datasets have 'time' coordinate)
    time = ds['time'].values

    # Build a row per variable
    for var in variables_of_interest:
        if var in ds:
            ts = ds[var].values
            runs.append({
                'model': model,
                'scenario': scenario,
                'run_type': RUN_TYPE,
                'variable': var,
                'time': time,
                'value': ts
            })

    ds.close()

# Now load everything into a DataFrame
df = pd.DataFrame(runs)

print(df)
