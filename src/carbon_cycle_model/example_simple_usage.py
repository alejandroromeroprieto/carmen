from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from carbon_cycle_model.carbon_cycle_model import CarbonCycle
from carbon_cycle_model.utils import load_esm_data
from carbon_cycle_model.constants import PPM2GT, PARS_DIR, SCEN_DIR


dt_model = 0.05
dt_model_ocean = 0.05

model = "UKESM1-0-LL"
scenario = "ssp585"

num_steps = int(1/dt_model)

if 1 % dt_model != 0:
    print(
        "WARNING: model timestep is not an exact divisor of 1 year"
        " Bad things may happen due to timestep mismatch"
    )

cc_emulator = CarbonCycle(
    {"model": "UKESM1-0-LL", "scenario": scenario}, dt_model, dt_model_ocean, **{}
)

# Load diagnosed ESM data from the data dir
data_file = Path(__file__).parent / SCEN_DIR / f"sce_{model}_{scenario}.txt"
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
    for j in range(num_steps):
        new_input = {
            "emis": esm_data.gcmemis[i],
            "dtocn": esm_data.dtocn[i],
            "dtglb": esm_data.dtglb[i],
        }

        if isinstance(esm_data.fcvegout, np.ndarray):
            new_input.update({"fcva": esm_data.fcvegout[i]})
        if isinstance(esm_data.fcsoilout, np.ndarray):
            new_input.update({"fcsa": esm_data.fcsoilout[i]})
        if isinstance(esm_data.fcvegoutcsoilin, np.ndarray):
            new_input.update({"fcvs": esm_data.fcvegoutcsoilin[i]})

        output = cc_emulator.run_one_step(new_input, npp_flag=True)

# plt.plot(cc_emulator.time, cc_emulator.land.gpp)
# plt.show()
# print(len(esm_data.time), num_steps_per_year, len(esm_data.time) * num_steps_per_year, len(output.cveg))
# print(len(output.time))
cc_emulator.create_plots(model, output, True)
