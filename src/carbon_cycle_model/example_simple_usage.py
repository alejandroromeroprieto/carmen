"""
Example script on how to use the carbon cycle emulator.

It does the following:
- Initialise the emulator from one of the saved configurations.
- Load data from the ESM corresponding to the loaded configuration.
- Run the emulator via its single step function, as long as the ESM data extends.
- Interpolate results back to the yearly resolution of the ESM data.
- Create some diagnostic plots about the results.
"""

from pathlib import Path

import numpy as np

from carbon_cycle_model.carbon_cycle_model import CarbonCycle
from carbon_cycle_model.utils import load_esm_data
from carbon_cycle_model.constants import SCEN_DIR


DT_MODEL = 1/8
DT_MODEL_OCEAN = 1/8

MODEL_NAME = "UKESM1-0-LL"
SCENARIO = "ssp119"

NUM_STEPS = int(1 / DT_MODEL)

if 1 % DT_MODEL != 0:
    print(
        "WARNING: model timestep is not an exact divisor of 1 year"
        " Bad things may happen due to timestep mismatch"
    )

cc_emulator = CarbonCycle(
    # if you wanna use ESM scenario data
    # {"model": "UKESM1-0-LL", "scenario": SCENARIO},
    # if you want to run without any pre-loaded scenario
    {"model": "UKESM1-0-LL", "initial_year": 1850, "final_year": 2100},
    DT_MODEL,
    DT_MODEL_OCEAN,
    npp_flag=True,
    **{},
)

# Load diagnosed ESM data from the data dir


# Load diagnosed ESM data from the data dir
if MODEL_NAME == "CNRM-ESM2-1" or MODEL_NAME == "IPSL-CM6A-LR":
    scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
else:
    scen_to_use = SCEN_DIR

data_file = Path(__file__).parent / scen_to_use / f"sce_{MODEL_NAME}_{SCENARIO}.txt"
print("\nLoading ESM data from: ", data_file)

esm_data = load_esm_data(
    data_file,
    recalc_emis=True,
    ninit=20,  # Because it's an SSP scenario
    smoothing_pars={"type": "savgol", "pars": [21, 3]},
)


# TODO: The model right now is broken for timesteps that do not fully divide a year.
# I could have fixed this, but I think the correct fix will depend on how FaIR deals
# with timesteps, particularly when they don't fully divide a year. So I will leave it
# as it is, until I talk to Chris about this.
# A potential idea I had is that I could loop the run_one_step() function self.num_steps
# times, and find the correct year from the ESM data to feed to the model. However, I'm
# guessing that FaIR will want to feed its own data on its own timesteps, so that may not
# be a good idea. Let's wait and see how FaIR does it first.
for i in range(len(esm_data.time)):  # Assuming esm data is yearly
    for j in range(NUM_STEPS):
        new_input = {
            "emis": esm_data.gcmemis[i],  # you are supplying the rate here
            "dtocn": esm_data.dtocn[i],
            "dtglb": esm_data.dtglb[i],
        }

        if isinstance(esm_data.fcvegout, np.ndarray):
            new_input.update({"fcva": esm_data.fcvegout[i]})
        if isinstance(esm_data.fcsoilout, np.ndarray):
            new_input.update({"fcsa": esm_data.fcsoilout[i]})
        if isinstance(esm_data.fcvegoutcsoilin, np.ndarray):
            new_input.update({"fcvs": esm_data.fcvegoutcsoilin[i]})
        print(new_input)
        cc_emulator.run_one_step(new_input)


cc_emulator.interpolate_results(esm_data.time)
cc_emulator.create_plots(MODEL_NAME)
