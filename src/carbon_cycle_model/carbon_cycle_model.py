"""
Class emulating the carbon cycle, bringing together the land and ocean components. This
file also implements a call to the cli_parser class to enable running the class through
the cli.
"""

from pathlib import Path
import time as systime

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from carbon_cycle_model.cli_parser import cli_parser
from carbon_cycle_model.land_component.land_component import LandCarbonCycle
from carbon_cycle_model.ocean_component.ocean_component import OceanCarbonCycle
from carbon_cycle_model.utils import Data, load_esm_data
from carbon_cycle_model.constants import PPM2GT, PARS_DIR, SCEN_DIR
from carbon_cycle_model import defaults


class CarbonCycle:
    """
    Parent class of the carbon cycle emulator.

    This class implements all the user-facing functionalities of the emulator.
    It also internally initialises the different components of the model.

    Required input to initialise class:
    - esm_data: object of type "Data", containing ESM data necessary to run the emulation.
                The intended provenance of this argument is the output from
                load_esm_data(), loading esm data from files under the data/ folder.
    - dtpred:   timestep size (in years) used to run the emulation. Notice that values
                smaller than a year are recommended for accurate integration, but this
                does not mean the model offers any seasonal prediction.
    - dtoccmax: timestep (in years) used to run the emulation of the ocean carbon cycle.
                The scheme is comparatively slower than the land's, but larger timesteps
                may result in stabilties, so care should be taken to choose the
                appropriate size.
    - npp_flag: whether to run the model with NPP, or alternatively with GPP and
                vegetation respiration.
    - scc_pars: dictionary containing the values for the model parameters, as present
                under the scenarios/ folder.
    """

    def __init__(
        self, esm_data, dtpred=0.03, dtoccmax=0.03, npp_flag=False, **scc_pars
    ):
        # If esm_data is a dictionary, load the esm data and pars for the requested
        # model/scenario
        if isinstance(esm_data, dict):
            model = esm_data.get("model", defaults.MODEL)

            # Prepare data folders to use based on the model
            if model == "CNRM-ESM2-1" or model == "IPSL-CM6A-LR":
                scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
                pars_to_use = PARS_DIR + "/detrended_wrt_decade"
            else:
                scen_to_use = SCEN_DIR
                pars_to_use = PARS_DIR

            # User may want to load data from one of our stored scenarios, for easy
            # comparison between emulation and ESM data
            if "scenario" in esm_data.keys():
                scenario = esm_data.get("scenario", defaults.MODEL)

                # If we are dealing with an SSP scenario, average the first few data
                # points to denoise the first value. However, if dealing with a
                # 1pctco2(-cdr) just take the first value, as subsequent values will
                # already have significantly diverged from equilibrium.
                if scenario in ["1pctco2", "1pctco2-cdr"]:
                    ninit_scenario = 1
                else:
                    ninit_scenario = 20

                # Load diagnosed ESM data from the data dir
                data_file = (
                    Path(__file__).parent / scen_to_use / f"sce_{model}_{scenario}.txt"
                )
                print("\nLoading ESM data from: ", data_file)

                esm_data = load_esm_data(
                    data_file,
                    recalc_emis=True,
                    ninit=ninit_scenario,
                    smoothing_pars={"type": "savgol", "pars": [21, 3]},
                )
            
                initial_year = esm_data.time[0]
                final_year = esm_data.time[-1]

            # Alternatively, the code really just needs the initial and final years
            # to know how many steps it should run for, so make sure that is defined
            elif "initial_year" in esm_data.keys() and "final_year" in esm_data.keys():
                initial_year = esm_data.get("initial_year")
                final_year = esm_data.get("final_year")
            else:
                raise ValueError(
                    "The Carbon Cycle class requires the initial and final "
                    "years to include in the emulation. Make sure these are "
                    "passed to the class initialisation through the esm_data"
                    "dictionary, under the 'initial_year' and 'final years' "
                    "keys."
                )

            # If no sccpars have been loaded, load default values for this model/scenario

            # Right now we only have loaded cross_scenario par calibration, so that
            # is the only one we can load. We could relax this to load calibrations
            # for different scenarios, in which case we could use scenario-specic
            # calirbations here
            scenario_pars = "cross_experiment"
            if not scc_pars:
                # Load calibrated SCC parameter from saved dict.
                pars_file = (
                    Path(__file__).parent
                    / pars_to_use
                    / f"sccpar_{model}_{scenario_pars}.txt"
                )
                print("Loading SCC parameters from: ", pars_file)

                with open(pars_file, "r", encoding="utf-8") as infile:
                    scc_pars = json.load(infile)
        elif isinstance(esm_data, Data):
            # Data has already been loaded to the right variable, so no need to do
            # anything
            initial_year = esm_data.time[0]
            final_year = esm_data.time[-1]
        else:
            raise TypeError(
                f"Unexpected esm_data argument of type: {type(esm_data)}. "
                f" Allowed object types are string or Data."
            )

        # Calculate the number of steps the model will perform, so
        # model arrays can be prepared for that exact number of time points.
        # If we have ESM data from e.g. 1850-2100, we want to have 1/dtpred steps
        # for each of those years (including 2100), as the ESM data is an average
        # of the whole year. Then we can interpolate back to yearly date from
        # 1850-2100 for consistency and comparison purposes with interpolate_results().
        # So, if the year difference (final_year - initial_year) is 150,
        # we actually want to run 151 years, to include the last year.
        # Additionally, since our code is storing the results for the next step
        # (i.e. the last step of year 2100 stores the new carbon stocks for the first
        # step of year 2101), we add an additional step.
        num_steps = round(1 + (1 + final_year - initial_year) / dtpred)
        time_cc = np.linspace(
            initial_year, final_year + 1, num=num_steps, endpoint=True
        )
        timestep = time_cc[1] - time_cc[0]

        # Create land and ocean components
        self.land = LandCarbonCycle(timestep, num_steps, **scc_pars)
        self.ocean = OceanCarbonCycle(
            timestep, dtoccmax, num_steps, initial_year, **scc_pars
        )

        # Store relevant quantities as attributes
        self.catm0 = scc_pars.get("catm0", defaults.CATM0_DEFAULT)
        self.catm = np.ones(num_steps) * scc_pars.get("catm0", defaults.CATM0_DEFAULT)
        self.emis = np.zeros(num_steps)
        self.cemis = 0
        self.num_steps = num_steps
        self.esm_data = esm_data
        self.dtpred = dtpred
        self.dtoccmax = dtoccmax

        # Only for run_one
        # TODO: separate into two cases
        # start = scc_pars.get("time0", defaults.TIME0_DEFAULT)
        # self.time = np.arange(start, start + num_steps * dtpred, dtpred)
        self.time = time_cc
        self.current_step = 0
        self.npp_flag = npp_flag

    def run_full_simulation(self):
        """
        Run a full scenario simulation, making use of the ESM data previously loaded.

        Results of the simulation are available as class attributes.
        """
        # ESM data is in yearly format, but we allow non-yearly timesteps. We need,
        # therefore, to interpolate ESM data to the resolution we are going to use
        # for the emulation, run the emulation in that resolution, and (for convenience)
        # return to the original yearly resolution.
        #  Subscript '_i' is often used below for interploted quantities
        time_i = np.linspace(
            self.esm_data.time[0],
            self.esm_data.time[-1],
            num=self.num_steps,
            endpoint=True,
        )

        emis_i = np.interp(time_i, self.esm_data.time, self.esm_data.emis)
        dtocn_i = np.interp(time_i, self.esm_data.time, self.esm_data.dtocn)
        dtglb_i = np.interp(time_i, self.esm_data.time, self.esm_data.dtglb)

        # Some quantities may or not be present. If they are not present
        # fill with zeros. These quantities are:
        #  - fcva: Flux of Carbon from Vegetation to Atmopshere
        #  - fcsa: Flux of Carbon from Soil to Atmopshere
        #  - fcvs: Flux of Carbon from Vegetation to Soil (unaccounted for in litterfall)
        if self.esm_data.fcvegout is not None and self.esm_data.fcvegout.any():
            fcva_i = np.interp(time_i, self.esm_data.time, self.esm_data.fcvegout)
        else:
            fcva_i = np.zeros(len(emis_i))

        if self.esm_data.fcsoilout is not None and self.esm_data.fcsoilout.any():
            fcsa_i = np.interp(time_i, self.esm_data.time, self.esm_data.fcsoilout)
        else:
            fcsa_i = np.zeros(len(emis_i))

        if (
            self.esm_data.fcvegoutcsoilin is not None
            and self.esm_data.fcvegoutcsoilin.any()
        ):
            fcvs_i = np.interp(
                time_i, self.esm_data.time, self.esm_data.fcvegoutcsoilin
            )
        else:
            fcvs_i = np.zeros(len(emis_i))

        # Cumulative emissions, used to track carbon losses
        cems = np.zeros(self.num_steps)

        # Run the components forward
        for i in range(0, self.num_steps - 1):
            timestep = time_i[i + 1] - time_i[i]
            dt_ems = emis_i[i] * timestep
            dt_fcva = fcva_i[i] * timestep
            dt_fcsa = fcsa_i[i] * timestep
            cems[i + 1] = cems[i] + dt_ems  # cummulative emissions
            co2_atm = self.catm[i] + (dt_ems + dt_fcva + dt_fcsa) / PPM2GT

            # Update land carbon cycle first, then ocean carbon cycle (testing shows
            # order hardly matters).
            delta_cveg, delta_csoil = self.land.update(
                dtglb_i[i],
                co2_atm,
                self.npp_flag,
                fcva=fcva_i[i],
                fcsa=fcsa_i[i],
                fcvs=fcvs_i[i],
            )
            del_lnd = delta_cveg + delta_csoil

            # del_lnd is the difference in carbon in veg and soil
            # so it does include the effects of deforestation and harvesting
            co2_atm = self.catm[i] + (dt_ems - del_lnd) / PPM2GT

            # Here we are using co2_atm as already updated by the land cycle. It could be
            # argued that we should use co2_atm from previous timestep (unupdated) for
            # both land and carbon cycle and then update catm. It probably does not make
            # much of a difference.

            # Update ocean carbon cycle.
            delta_ocn = self.ocean.update(
                co2_atm,
                self.catm[i],
                i,
                dtocn_i[i + 1],
            )

            self.catm[i + 1] = self.catm[i] + (dt_ems - delta_ocn - del_lnd) / PPM2GT

            self.current_step += 1

        return 0

    def run_one_step(self, new_input):
        """
        Run the next step of the simulation, making use of the new input passed as an
        argument.

        - new_input: dictionary with the required input to run a new step:
            + Emissions for the new timestep (GtC/year), under the "emis" key.
            + Sea surface temperature anomaly (Kelvin), under the "dtocn" key.
            + Earth's surface temperature anomaly (Kelvin), under the "dtglb" key.
        The dictionary can also have these optinal variables:
            + Additional carbon flux from vegetation to atmosphere (GtC/year), under
              the "fcva" key.
            + Additional carbon flux from soil to atmosphere (GtC/year), under
              the "fcsa" key.
            + Additional carbon flux from vegetation to soil (GtC/year), under
              the "fcvs" key.

        Results of the simulation are available as class attributes.
        """

        new_emis = new_input.get("emis")
        new_dtocn = new_input.get("dtocn")
        new_dtglb = new_input.get("dtglb")

        if not new_emis or not new_dtocn or not new_dtglb:
            raise ValueError("Incorrect new_input supplied. Missing required values.")

        new_fcva = new_input.get("fcva", 0)
        new_fcsa = new_input.get("fcsa", 0)
        new_fcvs = new_input.get("fcvs", 0)

        # Run the components forward one step
        dt_ems = new_emis * self.dtpred
        dt_fcva = new_fcva * self.dtpred
        dt_fcsa = new_fcsa * self.dtpred
        self.cemis += dt_ems  # cummulative emissions
        co2_atm = self.catm[self.current_step] + (dt_ems + dt_fcva + dt_fcsa) / PPM2GT

        # Update land carbon cycle first, then ocean carbon cycle (testing shows
        # order hardly matters).
        delta_cveg, delta_csoil = self.land.update(
            new_dtglb,
            co2_atm,
            self.npp_flag,
            fcva=new_fcva,
            fcsa=new_fcsa,
            fcvs=new_fcvs,
        )
        del_lnd = delta_cveg + delta_csoil

        # del_lnd is the difference in carbon in veg and soil
        # so it does include the effects of deforestation and harvesting
        co2_atm = self.catm[self.current_step] + (dt_ems - del_lnd) / PPM2GT

        # Here we are using co2_atm as already updated by the land cycle. It could be
        # argued that we should use co2_atm from previous timestep (unupdated) for
        # both land and carbon cycle and then update catm. It probably does not make
        # much of a difference.

        # Update ocean carbon cycle.
        delta_ocn = self.ocean.update(
            co2_atm,
            self.catm[self.current_step],
            self.current_step,
            new_dtocn,
        )

        self.catm[self.current_step + 1] = (
            self.catm[self.current_step] + (dt_ems - delta_ocn - del_lnd) / PPM2GT
        )
        self.emis[self.current_step + 1] = dt_ems

        self.current_step += 1

        return 0

    def create_plots(self, model):
        """
        Create some diagnostic plots about the carbon cycle emulation.

        input:
            -model: name of the model being simulated, so it can be added to plot titles.
        """

        # Set font and size of plots
        matplotlib.rcParams.update({"font.size": 10})
        plt.figure(1, figsize=(24 / 2.54, 9 / 2.54))
        plt.subplots_adjust(
            top=0.90, bottom=0.14, left=0.08, right=0.98, hspace=0.25, wspace=0.25
        )
        # Simple diagnostic plot of emissions, and corresponding SCC catm predictions
        # (with GCM temperature).
        ax = plt.subplot(1, 4, 1)
        if all(field in self.esm_data.keys() for field in ["time", "gcemis"]):
            plt.plot(
                self.esm_data.time, self.esm_data.gcmemis, color="orange", label="Input"
            )
        plt.plot(
            self.time,
            self.emis,
            color="dodgerblue",
            label="Calculated",
        )
        plt.title(model + ": Emis")
        leg = ax.legend(
            loc="upper left",
            handlelength=1.0,
            borderaxespad=0.3,
            handletextpad=0.3,
            labelspacing=0.3,
        )
        leg.draw_frame(False)

        # Compare the calculated atmospheric C02 concentration with the one from the ESM
        # dataset (To see how well the model is performing)
        ax = plt.subplot(1, 4, 2)
        plt.plot(
            self.time,
            self.catm,
            color="orange",
            alpha=0.5,
            label="ESM",
        )
        plt.plot(self.time, self.catm, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": catm")
        leg = ax.legend(
            loc="upper left",
            handlelength=1.0,
            borderaxespad=0.3,
            handletextpad=0.3,
            labelspacing=0.3,
        )
        leg.draw_frame(False)

        # Find the difference between calculated atmospheric C02 concentration with the
        # one from the ESM dataset
        # (To see how well the model is performing)
        ax = plt.subplot(1, 4, 3)
        if all(field in self.esm_data.keys() for field in ["time", "catm"]):
            plt.plot(self.time, self.esm_data.catm - self.catm, lw=1.0, color="blue")
            plt.axhline(0.0, ls=":", color="k")
            plt.title("catm error: ESM-SCC")

        # Plot the carbon contained in vegetation, soil, atmosphere and (difference wrt
        # initial point) ocean.
        ax = plt.subplot(1, 4, 4)
        plt.plot(self.time, self.land.cveg, label="cveg")
        plt.plot(self.time, self.land.csoil, label="csoil")
        plt.plot(self.time, self.catm, label="catm")
        plt.plot(self.time, self.ocean.carbon_increase, label="cocn")
        plt.legend()
        plt.show()

        # === Carbon fluxes subplot ===

        # Atmospheric carbon concentration
        ax = plt.subplot(2, 4, 1)
        # Only attempt to print esm_data if it is present
        if all(field in self.esm_data.keys() for field in ["time", "catm"]):
            plt.plot(
                self.esm_data.time,
                self.esm_data.catm,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
        plt.plot(self.time, self.catm, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": Atm. CO2 concentration")

        # Carbon stock in the vegetation pool
        ax = plt.subplot(2, 4, 2)
        if all(field in self.esm_data.keys() for field in ["time", "cveg"]):
            plt.plot(
                self.esm_data.time,
                self.esm_data.cveg,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
        plt.plot(self.time, self.land.cveg, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": Vegetation carbon")

        # Carbon stock in the soil pool
        ax = plt.subplot(2, 4, 3)
        if all(field in self.esm_data.keys() for field in ["time", "csoil"]):
            plt.plot(
                self.esm_data.time,
                self.esm_data.csoil,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
        plt.plot(self.time, self.land.csoil, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": Soil carbon")

        # TCumulative carbon uptake by the ocean (Notice is not the total stock)
        ax = plt.subplot(2, 4, 4)
        if all(field in self.esm_data.keys() for field in ["time", "oflux"]):
            plt.plot(
                self.esm_data.time,
                np.cumsum(self.esm_data.oflux),
                color="orange",
                alpha=0.5,
                label="ESM",
            )
        plt.plot(
            self.time,
            self.ocean.carbon_increase,
            color="dodgerblue",
            alpha=0.5,
            label="SCC",
        )
        plt.title(model + ": Cum. ocean uptake")

        if not self.npp_flag:
            # GPP flux
            ax = plt.subplot(2, 4, 5)
            if all(field in self.esm_data.keys() for field in ["time", "gpp"]):
                plt.plot(
                    self.esm_data.time,
                    self.esm_data.gpp,
                    color="orange",
                    alpha=0.5,
                    label="ESM",
                )
            plt.plot(
                self.time, self.land.gpp, color="dodgerblue", alpha=0.5, label="SCC"
            )
            plt.title(model + ": GPP")
            plt.legend()

            # Vegetation respiration flux
            ax = plt.subplot(2, 4, 6)
            if all(field in self.esm_data.keys() for field in ["time", "ra"]):
                plt.plot(
                    self.esm_data.time,
                    self.esm_data.ra,
                    color="orange",
                    alpha=0.5,
                    label="ESM",
                )
            plt.plot(
                self.time, self.land.vres, color="dodgerblue", alpha=0.5, label="SCC"
            )
            plt.title(model + ": Veg. respiration")
            plt.legend()
        else:
            # NPP flux
            ax = plt.subplot(2, 4, 5)
            if all(field in self.esm_data.keys() for field in ["time", "npp"]):
                plt.plot(
                    self.esm_data.time,
                    self.esm_data.npp,
                    color="orange",
                    alpha=0.5,
                    label="ESM",
                )
            plt.plot(
                self.time, self.land.npp, color="dodgerblue", alpha=0.5, label="SCC"
            )
            plt.title(model + ": NPP")
            plt.legend()

            # Carbon uptake by the ocean (positive into the ocean)
            ax = plt.subplot(2, 4, 6)
            if all(field in self.esm_data.keys() for field in ["time", "oflux"]):
                plt.plot(
                    self.esm_data.time,
                    self.esm_data.oflux,
                    color="orange",
                    alpha=0.5,
                    label="ESM",
                )
            plt.plot(
                self.time, self.ocean.oflux, color="dodgerblue", alpha=0.5, label="SCC"
            )
            plt.title(model + ": ocean uptake")
            plt.legend()

        # Litterfall flux
        ax = plt.subplot(2, 4, 7)
        if all(field in self.esm_data.keys() for field in ["time", "lit"]):
            plt.plot(
                self.esm_data.time,
                self.esm_data.lit,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
        plt.plot(self.time, self.land.lit, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": Litterfall")
        plt.legend()

        # Soil respiration flux
        ax = plt.subplot(2, 4, 8)
        if all(field in self.esm_data.keys() for field in ["time", "rh"]):
            plt.plot(
                self.esm_data.time,
                self.esm_data.rh,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
        plt.plot(self.time, self.land.sres, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": soil respiration")
        plt.legend()

        plt.show()

    def interpolate_results(self, new_time):
        """
        Interpolate class attributes to a new timeseries.

        input:
        - new_time: new timeseries to interpolate to.

        output: none, but class attributes will be interpolated to the new time series.
        """
        if self.current_step < self.num_steps - 1:
            print(
                "WARNING: you are running data interpolation before the emulator has "
                "finished running all expected timesteps. This is not supported, "
                "and will likely break any future running of the emulator."
            )

        # land box
        self.land.cveg = np.interp(new_time, self.time, self.land.cveg)
        self.land.csoil = np.interp(new_time, self.time, self.land.csoil)
        self.land.npp = np.interp(new_time, self.time, self.land.npp)
        self.land.lit = np.interp(new_time, self.time, self.land.lit)
        self.land.sres = np.interp(new_time, self.time, self.land.sres)
        self.land.fcva = np.interp(new_time, self.time, self.land.fcva)
        self.land.fcsa = np.interp(new_time, self.time, self.land.fcsa)
        self.land.fcvs = np.interp(new_time, self.time, self.land.fcvs)

        if not self.npp_flag:
            self.land.gpp = np.interp(new_time, self.time, self.land.gpp)
            self.land.vres = np.interp(new_time, self.time, self.land.vres)

        # Ocean box
        self.ocean.carbon_increase = np.interp(
            new_time, self.time, self.ocean.carbon_increase
        )
        self.ocean.oflux = np.interp(new_time, self.time, self.ocean.oflux)

        # General carbon cycle box
        self.catm = np.interp(new_time, self.time, self.catm)
        self.emis = np.interp(new_time, self.time, self.emis)
        self.time = new_time


def main():
    """ "
    Enable the execution of the carbon cycle emulator through the use of the CLI.

    This code will:
    - call the cli_parser() function to obtain the necessary details to run the emulator.
    - load the relevant ESM data from the data/scenarios folder.
    - load the relevant model parameters from the data/pars folder.
    - run the emulator for the same time duration as the ESM data.
    - create diagnostic plots.
    """

    # Smoothing to perform on input ESM data
    smoothing_algorithm = {"type": "savgol", "pars": [21, 3]}

    # Numerical instability can arise if dtoccmax is too large. We recommend to keep it
    # to 0.2 or less.
    dtpred = 0.05  # main timestep (years) for land carbon cycle
    dtoccmax = 0.05  # max timestep (years) for ocean carbon cycle

    # Calculate emissions from carbon mass conservation across the different carbon pools
    recalc_emis = True

    # Run the cli parser and retrieve the required information
    models, scenario, scenario_pars, _, npp_flag = cli_parser()

    # Run emulation for each model
    for model in models:
        # If we are dealing with an SSP scenario, average the first few data
        # points to denoise the first value. However, if dealing with a
        # 1pctco2(-cdr) just take the first value, as subsequent values will
        # already have significantly diverged from equilibrium.
        if scenario in ["1pctco2", "1pctco2-cdr"]:
            ninit_scenario = 1
        else:
            ninit_scenario = 20

        # Load diagnosed ESM data from the data dir
        if model == "CNRM-ESM2-1" or model == "IPSL-CM6A-LR":
            scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
            pars_to_use = PARS_DIR + "/detrended_wrt_decade"
        else:
            scen_to_use = SCEN_DIR
            pars_to_use = PARS_DIR

        data_file = Path(__file__).parent / scen_to_use / f"sce_{model}_{scenario}.txt"
        print("\nLoading ESM data from: ", data_file)

        esm_data = load_esm_data(
            data_file,
            recalc_emis=recalc_emis,
            ninit=ninit_scenario,
            smoothing_pars=smoothing_algorithm,
        )

        # Load calibrated SCC parameter from saved dict.
        pars_file = (
            Path(__file__).parent / pars_to_use / f"sccpar_{model}_{scenario_pars}.txt"
        )
        print("Loading SCC parameters from: ", pars_file)

        with open(pars_file, "r", encoding="utf-8") as infile:
            scc_pars = json.load(infile)

        # Measure the time it takes us to run the model
        tbeg = systime.time()

        # Build the model and run it
        cc_emulator = CarbonCycle(esm_data, dtpred, dtoccmax, npp_flag, **scc_pars)
        cc_emulator.run_full_simulation()

        # Interpolate back to ESM time points, so we can compare results
        cc_emulator.interpolate_results(esm_data.time)

        tend = systime.time()
        print("Time to process:", tend - tbeg)

        cc_emulator.create_plots(model)


if __name__ == "__main__":
    main()
