"""
Parent class simulating the carbon cycle, bringing together the land and ocean
components. This file also implements a call to the cli_parser class to enable
running the class through the CLI.
"""

import os
from pathlib import Path
import time as systime

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from carmen.cli_parser import cli_parser
from carmen.land_component.land_component import LandCarbonCycle
from carmen.ocean_component.ocean_component import OceanCarbonCycle
from carmen.utils import Data, load_esm_data, make_all_dirs
from carmen.constants import PPM2GT, PARS_DIR, SCEN_DIR, OUTPUT_DIR
from carmen import defaults

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

                # Load diagnosed ESM data from the data dir
                data_file = (
                    Path(__file__).parent / scen_to_use / f"sce_{model}_{scenario}.txt"
                )
                print("\nLoading ESM data from: ", data_file)

                esm_data = load_esm_data(
                    data_file,
                    recalc_emis=True,
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
            # calibrations here
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

        # emis_i = np.interp(time_i, self.esm_data.time, [10]*len(self.esm_data.time))
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
                dtocn_i[i + 1],
            )

            self.catm[i + 1] = self.catm[i] + (dt_ems - delta_ocn - del_lnd) / PPM2GT
            self.emis[i + 1] = emis_i[i]

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

        if any(key not in new_input for key in ["emis", "dtocn", "dtglb"]):
            raise ValueError(
                "Incorrect new_input supplied. Missing at least one of the required "
                "key values: 'emis', 'dtocn' and 'dtglb'."
            )

        new_emis = new_input.get("emis")
        new_dtocn = new_input.get("dtocn")
        new_dtglb = new_input.get("dtglb")

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
            new_dtocn,
        )

        self.catm[self.current_step + 1] = (
            self.catm[self.current_step] + (dt_ems - delta_ocn - del_lnd) / PPM2GT
        )
        self.emis[self.current_step + 1] = new_emis

        self.current_step += 1

        return 0

    def create_plots(self, model, scenario):
        """
        Create some diagnostic plots about the carbon cycle emulation.

        input:
            - model: name of the model being simulated, so it can be added to plot titles.
            - scenario: name of the scenario being simulated.
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
        if all(field in self.esm_data.keys() for field in ["time", "catm"]):
            plt.plot(
                self.esm_data.time,
                self.esm_data.catm,
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

        # Create the figure and axes
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(21, 10), sharex=True)
        fig.suptitle(f"CARMEN emulation of {model} carbon cycle for {scenario}", fontsize=30)

        # Flatten axes for easier access
        axes = axes.flatten()

        # Titles and Y-axis labels for each subplot
        titles = [
            "Atm. CO2 concentration", "Vegetation carbon", "Soil carbon", "Cum. ocean uptake",
            "GPP" if not self.npp_flag else "NPP",
            "Veg. respiration" if not self.npp_flag else "Ocean uptake",
            "Litterfall", "Soil respiration"
        ]

        ylabels = [
            "ppm", "GtC", "GtC", "GtC",
            "GtC/yr", "GtC/yr",
            "GtC/yr", "GtC/yr"
        ]

        # Plot data in each subplot
        linewidth_to_use = 4
        for i, ax in enumerate(axes):
            if i == 0 and all(hasattr(self.esm_data, k) for k in ["time", "catm"]):
                ax.plot(self.esm_data.time, self.esm_data.catm, color="orange", alpha=1, label="ESM", linewidth=linewidth_to_use)
                ax.plot(self.time, self.catm, color="dodgerblue", alpha=1, label="CARMEN", linewidth=linewidth_to_use)

            elif i == 1 and all(hasattr(self.esm_data, k) for k in ["time", "cveg"]):
                ax.plot(self.esm_data.time, self.esm_data.cveg, color="orange", alpha=1, linewidth=linewidth_to_use)
                ax.plot(self.time, self.land.cveg, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)

            elif i == 2 and all(hasattr(self.esm_data, k) for k in ["time", "csoil"]):
                ax.plot(self.esm_data.time, self.esm_data.csoil, color="orange", alpha=1, linewidth=linewidth_to_use)
                ax.plot(self.time, self.land.csoil, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)

            elif i == 3 and all(hasattr(self.esm_data, k) for k in ["time", "oflux"]):
                ax.plot(self.esm_data.time, np.cumsum(self.esm_data.oflux), color="orange", alpha=1, linewidth=linewidth_to_use)
                ax.plot(self.time, self.ocean.carbon_increase, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)

            elif i == 4:
                field = "gpp" if not self.npp_flag else "npp"
                data = self.land.gpp if not self.npp_flag else self.land.npp
                if all(hasattr(self.esm_data, k) for k in ["time", field]):
                    ax.plot(self.esm_data.time, getattr(self.esm_data, field), color="orange", alpha=1, linewidth=linewidth_to_use)
                    ax.plot(self.time, data, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)

            elif i == 5:
                if not self.npp_flag:
                    if all(hasattr(self.esm_data, k) for k in ["time", "ra"]):
                        ax.plot(self.esm_data.time, self.esm_data.ra, color="orange", alpha=1, linewidth=linewidth_to_use)
                        ax.plot(self.time, self.land.vres, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)
                else:
                    if all(hasattr(self.esm_data, k) for k in ["time", "oflux"]):
                        ax.plot(self.esm_data.time, self.esm_data.oflux, color="orange", alpha=1, linewidth=linewidth_to_use)
                        ax.plot(self.time, self.ocean.oflux, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)

            elif i == 6 and all(hasattr(self.esm_data, k) for k in ["time", "lit"]):
                ax.plot(self.esm_data.time, self.esm_data.lit, color="orange", alpha=1, linewidth=linewidth_to_use)
                ax.plot(self.time, self.land.lit, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)

            elif i == 7 and all(hasattr(self.esm_data, k) for k in ["time", "rh"]):
                ax.plot(self.esm_data.time, self.esm_data.rh, color="orange", alpha=1, linewidth=linewidth_to_use)
                ax.plot(self.time, self.land.sres, color="dodgerblue", alpha=1, linewidth=linewidth_to_use)

            # Set subplot titles and Y-axis labels
            ax.set_title(f"{titles[i]}", fontsize=20)
            ax.set_ylabel(ylabels[i], fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=14)

        # Add a shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()

        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=18)

        # Adjust layout to make room for title and legend
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        output_filename = f'carmem_emulation_{model}_f{scenario}.png'
        plt.savefig(f"{OUTPUT_DIR}/plots/esm_emulation/{output_filename}", dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{output_filename}'")
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


    def store_results(self, model, scenario, scc_pars, out_file):
        """
        Save simulation results to a file.

        Input:
            - model: name of the model we are emulating.
            - scenario: name of the scenario we are emulating.
            - scc_pars: dictionary with the model parameters used for the emulation.
            - out_file: name of the file to store simulation data.
        """
        data_vars = {
            # Atmospheric CO2
            'esm_catm': ('time', self.esm_data.catm),
            'simulated_catm': ('time', self.catm),

            # Land box variables
            'esm_cveg': ('time', self.esm_data.cveg),
            'simulated_cveg': ('time', self.land.cveg),

            'esm_csoil': ('time', self.esm_data.csoil),
            'simulated_csoil': ('time', self.land.csoil),

            'esm_npp': ('time', self.esm_data.npp),
            'simulated_npp': ('time', self.land.npp),

            'esm_lit': ('time', self.esm_data.lit),
            'simulated_lit': ('time', self.land.lit),

            'esm_sres': ('time', self.esm_data.rh),
            'simulated_sres': ('time', self.land.sres),

            'esm_fcva': ('time', self.land.fcva),

            'esm_fcsa': ('time', self.land.fcsa),

            'esm_fcvs': ('time', self.land.fcvs),

            # Ocean box
            'esm_ocean_carbon_uptake': ('time', np.cumsum(self.esm_data.oflux)),
            'simulated_ocean_carbon_uptake': ('time', self.ocean.carbon_increase),

            'esm_oflux': ('time', self.esm_data.oflux),
            'simulated_oflux': ('time', self.ocean.oflux),

            # General carbon cycle boxy
            'esm_emissions': ('time', self.esm_data.gcmemis),
            'simulated_emissions': ('time', self.emis,),
        }

        # Conditionally add gpp and vres
        if not self.npp_flag:
            data_vars['esm_gpp'] = ('time', self.esm_data.gpp)
            data_vars['simulated_gpp'] = ('time', self.land.gpp)

            data_vars['esm_vres'] = ('time', self.esm_data.ra)
            data_vars['simulated_vres'] = ('time', self.land.vres)

        # Create Dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                'time': self.time
            }
        )

        # Add global metadata
        ds.attrs['model'] = model
        ds.attrs['scenario'] = scenario
        ds.attrs['model_pars'] = json.dumps(scc_pars)

        print(f"Saving results to {out_file}")
        ds.to_netcdf(out_file)

        return 0


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
    # smoothing_algorithm = {"type": "butterworth", "pars": [21, 3]}
    smoothing_algorithm = {"type": "butterworth", "pars": [1]}  # No smoothing

    # Numerical instability can arise if dtoccmax is too large. We recommend to keep it
    # to 0.2 or less.
    dtpred = 0.05  # main timestep (years) for land carbon cycle
    dtoccmax = 0.05  # max timestep (years) for ocean carbon cycle

    # Calculate emissions from carbon mass conservation across the different carbon pools
    recalc_emis = True

    # Run the cli parser and retrieve the required information
    models, scenario, scenario_pars, realisation, _, npp_flag, store_flag = cli_parser()

    # Run emulation for each model
    for model in models:
        # If we are dealing with an SSP scenario, average the first few data
        # points to denoise the first value. However, if dealing with a
        # 1pctco2(-cdr) just take the first value, as subsequent values will
        # already have significantly diverged from equilibrium.
        # How to derive pre-industrial values
        if model in ["BCC-CSM2-MR", "CESM2", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC-ES2L", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM", "UKESM1-0-LL"]:
            if "1pctco2" in scenario:
                pre_ind_average_length = 1
            elif "ssp" in scenario or "hist" in scenario:
                pre_ind_average_length = 20
            elif "abrupt" in scenario:
                pre_ind_average_length = 1
            elif "flat" in scenario:
                pre_ind_average_length = 1
            else:
                raise ValueError(f"Experiment {scenario} not recognised")

            pre_ind_algorithm = {"type": "average", "length": pre_ind_average_length}

        # Some models perform better if smoothing is applied before computing the pre-industrial properties
        elif model in ["ACCESS-ESM1-5", "CanESM5", "CMCC-ESM2", "CNRM-ESM2-1"]:
            pre_ind_algorithm = {"type": "butterworth", "length": [10]}

        else:
            raise ValueError(f"Model {model} not recognised")

        # Load diagnosed ESM data from the data dir
        if model == "CNRM-ESM2-1" or model == "IPSL-CM6A-LR":
            scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
            pars_to_use = PARS_DIR
        else:
            scen_to_use = SCEN_DIR
            pars_to_use = PARS_DIR

        if realisation != "default":
            scen_to_use = SCEN_DIR + "/other_realisations"
            data_file = (
                Path(__file__).parent
                / scen_to_use
                / f"sce_{model}_{scenario}_{realisation}.txt"
            )
        else:
            data_file = (
                Path(__file__).parent / scen_to_use / f"sce_{model}_{scenario}.txt"
            )

        print("\nLoading ESM data from: ", data_file)

        esm_data = load_esm_data(
            data_file,
            recalc_emis=recalc_emis,
            smoothing_pars=smoothing_algorithm,
        )

        # Load calibrated SCC parameter from saved dict.
        pars_file = (
            Path(__file__).parent / pars_to_use / f"sccpar_{model}_{scenario_pars}.txt"
        )
        print("Loading SCC parameters from: ", pars_file)

        with open(pars_file, "r", encoding="utf-8") as infile:
            scc_pars = json.load(infile)

        # Record pre-industrial values. Fluxes are an averaged over the
        # 0:ninit_scenario period (which depends on the scenario) and stocks
        # are simply the first value in the array. We do not take the average
        # for stocks because that would lead to larger discrepancies with the
        # ESM data we are trying to emulate, as they would start from a different
        # point.
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
                data_file, recalc_emis, smoothing_pars={"type": "butterworth", "pars": [10]}
            )

            scc_pars["npp0"] = esm_input_smoothed.npp[0]
            scc_pars["gpp0"] = esm_input_smoothed.gpp[0]
            scc_pars["lit0"] = esm_input_smoothed.lit[0]
            scc_pars["sres0"] = esm_input_smoothed.rh[0]
            scc_pars["vres0"] = esm_input_smoothed.ra[0]
        else:
            raise ValueError(
                f"Pre-industrial algorithm {pre_ind_algorithm['type']} not recognised."
            )

        # Measure the time it takes us to run the model
        tbeg = systime.time()

        # Build the model and run it
        cc_emulator = CarbonCycle(esm_data, dtpred, dtoccmax, npp_flag, **scc_pars)
        cc_emulator.run_full_simulation()

        # Interpolate back to ESM time points, so we can compare results
        cc_emulator.interpolate_results(esm_data.time)

        tend = systime.time()
        print("Time to process:", tend - tbeg)

        cc_emulator.create_plots(model, scenario)

        # Store results
        if store_flag:
            cwd = str(Path.cwd())
            out_file = os.path.join(cwd, OUTPUT_DIR)
            out_file += f"/simulation_{model}_{scenario}.nc"
            make_all_dirs(out_file)

            cc_emulator.store_results(model, scenario, scc_pars, out_file)


if __name__ == "__main__":
    main()
