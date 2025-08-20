"""
This is just a messy script originally employed by Alex Romero Prieto to create some
plots comparing different realisations. It is not condidere as part of the model code,
and as such it is presented "as is" without the same level of  clarity and comments as
the model code. It is left here for archiving purposes and in case it is useful in the
future, but it can be safely ignored.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from carbon_cycle_model.carbon_cycle_model import CarbonCycle
from carbon_cycle_model.utils import load_esm_data
from carbon_cycle_model.constants import SCEN_DIR, PARS_DIR

realisations = ["default", "r2i1p1f2", "r3i1p1f2", "r4i1p1f2", "r8i1p1f2"]

DT_MODEL = 1 / 8
DT_MODEL_OCEAN = 1 / 8

MODEL_NAME = "UKESM1-0-LL"
SCENARIO = "ssp370"

NUM_STEPS = int(1 / DT_MODEL)

if 1 % DT_MODEL != 0:
    print(
        "WARNING: model timestep is not an exact divisor of 1 year"
        " Bad things may happen due to timestep mismatch"
    )


# Load diagnosed ESM data from the data dir
if MODEL_NAME == "CNRM-ESM2-1" or MODEL_NAME == "IPSL-CM6A-LR":
    scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
    pars_to_use = PARS_DIR + "/detrended_wrt_decade"
else:
    scen_to_use = SCEN_DIR
    pars_to_use = PARS_DIR

esm_data_dict = {}
for realisation in realisations:

    if realisation == "default":
        data_file = (
            Path(__file__).parent.parent / scen_to_use / f"sce_{MODEL_NAME}_{SCENARIO}.txt"
        )
        print("\nLoading ESM data from: ", data_file)
    else:
        data_file = (
            Path(__file__).parent.parent
            / scen_to_use
            / "other_realisations"
            / f"sce_{MODEL_NAME}_{SCENARIO}_{realisation}.txt"
        )
        print("\nLoading ESM data from: ", data_file)

    esm_data = load_esm_data(
        data_file,
        recalc_emis=True,
        smoothing_pars={"type": "savgol", "pars": [21, 3]},
    )
    esm_data_dict.update({realisation: esm_data})


pars_file = (
    Path(__file__).parent.parent / pars_to_use / f"sccpar_{MODEL_NAME}_cross_experiment.txt"
)
print("Loading SCC parameters from: ", pars_file)

with open(pars_file, "r", encoding="utf-8") as infile:
    scc_pars = json.load(infile)

cc_dict = {}
for realisation in realisations:

    if SCENARIO in ["1pctco2", "1pctco2-cdr"]:
        ninit_scenario = 1
    else:
        ninit_scenario = 20

    scc_pars["cveg0"] = esm_data_dict[realisation].cveg[0]
    scc_pars["csoil0"] = esm_data_dict[realisation].csoil[0]
    scc_pars["catm0"] = np.mean(esm_data_dict[realisation].catm[0:ninit_scenario])
    scc_pars["npp0"] = np.mean(esm_data_dict[realisation].npp[0:ninit_scenario])
    scc_pars["gpp0"] = np.mean(esm_data_dict[realisation].gpp[0:ninit_scenario])
    scc_pars["lit0"] = np.mean(esm_data_dict[realisation].lit[0:ninit_scenario])
    scc_pars["sres0"] = np.mean(esm_data_dict[realisation].rh[0:ninit_scenario])
    scc_pars["vres0"] = np.mean(esm_data_dict[realisation].ra[0:ninit_scenario])

    cc_emulator = CarbonCycle(
        # if you wanna use ESM scenario data
        # {"model": "UKESM1-0-LL", "scenario": SCENARIO},
        # if you want to run without any pre-loaded scenario
        esm_data_dict[realisation],
        DT_MODEL,
        DT_MODEL_OCEAN,
        npp_flag=False,
        **scc_pars,
    )
    cc_dict.update({realisation: cc_emulator})

# TODO: The model right now is broken for timesteps that do not fully divide a year.
# I could have fixed this, but I think the correct fix will depend on how FaIR deals
# with timesteps, particularly when they don't fully divide a year. So I will leave it
# as it is, until I talk to Chris about this.
# A potential idea I had is that I could loop the run_one_step() function self.num_steps
# times, and find the correct year from the ESM data to feed to the model. However, I'm
# guessing that FaIR will want to feed its own data on its own timesteps, so that may not
# be a good idea. Let's wait and see how FaIR does it first.
for realisation in realisations:
    for i in range(len(esm_data_dict[realisation].time)):  # Assuming esm data is yearly
        for j in range(NUM_STEPS):
            new_input = {
                "emis": esm_data_dict[realisation].gcmemis[
                    i
                ],  # you are supplying the rate here
                "dtocn": esm_data_dict[realisation].dtocn[i],
                "dtglb": esm_data_dict[realisation].dtglb[i],
            }

            if isinstance(esm_data_dict[realisation].fcvegout, np.ndarray):
                new_input.update({"fcva": esm_data_dict[realisation].fcvegout[i]})
            if isinstance(esm_data_dict[realisation].fcsoilout, np.ndarray):
                new_input.update({"fcsa": esm_data_dict[realisation].fcsoilout[i]})
            if isinstance(esm_data_dict[realisation].fcvegoutcsoilin, np.ndarray):
                new_input.update(
                    {"fcvs": esm_data_dict[realisation].fcvegoutcsoilin[i]}
                )
            cc_dict[realisation].run_one_step(new_input)

    cc_dict[realisation].interpolate_results(esm_data_dict[realisation].time)


# plotting
# for realisation in realisations:
#     plt.scatter(esm_data_dict[realisation].time, esm_data_dict[realisation].cveg, label=f"ESM ({realisation})", s=20, marker="x")
#     plt.plot(esm_data_dict[realisation].time, cc_dict[realisation].land.cveg, label=f"SCM ({realisation})")

fig, axes = plt.subplots(nrows=2, ncols=4)
for realisation in realisations:
    axes[0, 0].scatter(
        esm_data_dict[realisation].time,
        esm_data_dict[realisation].catm,
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[0, 0].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].catm,
        label=f"SCM ({realisation})",
    )

    axes[0, 1].scatter(
        esm_data_dict[realisation].time,
        esm_data_dict[realisation].cveg,
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[0, 1].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].land.cveg,
        label=f"SCM ({realisation})",
    )

    axes[0, 2].scatter(
        esm_data_dict[realisation].time,
        esm_data_dict[realisation].csoil,
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[0, 2].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].land.csoil,
        label=f"SCM ({realisation})",
    )

    axes[0, 3].scatter(
        esm_data_dict[realisation].time,
        np.cumsum(esm_data_dict[realisation].oflux),
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[0, 3].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].ocean.carbon_increase,
        label=f"SCM ({realisation})",
    )

    axes[1, 0].scatter(
        esm_data_dict[realisation].time,
        esm_data_dict[realisation].npp,
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[1, 0].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].land.npp,
        label=f"SCM ({realisation})",
    )

    axes[1, 1].scatter(
        esm_data_dict[realisation].time,
        esm_data_dict[realisation].oflux,
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[1, 1].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].ocean.oflux,
        label=f"SCM ({realisation})",
    )

    axes[1, 2].scatter(
        esm_data_dict[realisation].time,
        esm_data_dict[realisation].lit,
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[1, 2].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].land.lit,
        label=f"SCM ({realisation})",
    )

    axes[1, 3].scatter(
        esm_data_dict[realisation].time,
        esm_data_dict[realisation].rh,
        label=f"ESM ({realisation})",
        s=20,
        marker="x",
    )
    axes[1, 3].plot(
        esm_data_dict[realisation].time,
        cc_dict[realisation].land.sres,
        label=f"SCM ({realisation})",
    )

plt.legend()
plt.show()

# cc_emulator.create_plots(MODEL_NAME)
