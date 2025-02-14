"""
Class cimulating the carbon cycle, including land and ocean components.
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
from carbon_cycle_model.ocean_component.utils import joos_response
from carbon_cycle_model.utils import Data, load_esm_data
from carbon_cycle_model.constants import PPM2GT
from carbon_cycle_model import defaults


class CarbonCycle:
    """
    Parent class of the carbon cycle emulator.

    This class implements all the user-facing functionalities of the emulator.
    It also internally initialises the different components of the model.

    Required input to initialise class:
    - esm_data: object of type "Data", containing ESM data necessary to run the emulation.
                The intended provenance of this argument is the output from load_esm_data(),
                loading esm data from files under the data/ folder.
    - dtpred:   timestep (in years) used to run the emulation. Notice that values smaller
                than a year are recommended for accurate integration, but this does not
                mean the model offers any seasonal prediction.
    - dtoccmax: timestep (in years) used to run the emulation of the ocean carbon cycle.
                The scheme is comparatively slower than the land's, so users may want to
                use a larger timestep for this component.
    - kwargs:   dictionary containing the values for the model parameters, as presented
                under the scenarios/ folder.
    """

    def __init__(self, esm_data, dtpred=0.03, dtoccmax=0.03, **kwargs):
        # Calculate the number of steps the model will perform, so
        # model arrays can be prepared for that exact number of time points.
        num_steps = round(1 + (esm_data.time[-1] - esm_data.time[0]) / dtpred)
        time_cc = np.linspace(
            esm_data.time[0], esm_data.time[-1], num=num_steps, endpoint=True
        )
        timestep = time_cc[1] - time_cc[0]

        self.land = LandCarbonCycle(timestep, num_steps, **kwargs)
        self.ocean = OceanCarbonCycle(timestep, num_steps, **kwargs)

        self.catm0 = kwargs.get("catm0", defaults.CATM0_DEFAULT)
        self.num_steps = num_steps
        self.esm_data = esm_data
        self.dtpred = dtpred
        self.dtoccmax = dtoccmax

    def run_full_simulation(self, npp_flag=False):
        """
        Given the data from another model, run a full simulation.

        - npp_flag: whether to use NPP, rather than GPP and vegetation respiration, in
                    the emulation.

        return: Data object with results of the emulation.
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

        # Create arrays to store data that is not tracked at subcomponent level
        catm = self.catm0 * np.ones(self.num_steps)
        # Cumulative emissions, used to track carbon losses
        cems = np.zeros(self.num_steps)

        # To avoid instabilities, you may want to use a smaller timestep for
        # the ocean component
        timestep = time_i[1] - time_i[0]
        ocean_steps_per_model_steps = max([1, int(timestep / self.dtoccmax)])
        dt4occ = timestep / ocean_steps_per_model_steps

        timeocc = np.arange(time_i[0], time_i[-1] + 0.000001, dt4occ)
        timeocc[-1] = min(time_i[-1], timeocc[-1])
        ntimeocc = len(timeocc)

        rjoos = joos_response(timeocc)

        ocn_uptake = np.zeros(ntimeocc, "f")

        # Run the components forward
        for i in range(0, self.num_steps - 1):
            timestep = time_i[i + 1] - time_i[i]
            del_ems = emis_i[i] * timestep
            del_fcva = fcva_i[i] * timestep
            del_fcsa = fcsa_i[i] * timestep
            cems[i + 1] = cems[i] + del_ems  # cummulative emissions
            co2_atm = catm[i] + (del_ems + del_fcva + del_fcsa) / PPM2GT

            # Update land carbon cycle first, then ocean carbon cycle (testing shows
            # order hardly matters).
            delta_cveg, delta_csoil = self.land.update(
                dtglb_i[i],
                co2_atm,
                npp_flag,
                fcva=fcva_i[i],
                fcsa=fcsa_i[i],
                fcvs=fcvs_i[i],
            )
            del_lnd = delta_cveg + delta_csoil

            # del_lnd is the difference in carbon in veg and soil
            # so it does include the effects of deforestation and harvesting
            co2_atm = catm[i] + (del_ems - del_lnd) / PPM2GT

            # Here we are using co2_atm as already updated by the land cycle. It could be
            # argued that we should use co2_atm from previous timestep (unupdated) for
            # both land and carbon cycle and then update catm. It probably does not make
            # much of a difference.

            # Update ocean carbon cycle.
            delta_ocn, _ = self.ocean.update(
                co2_atm,
                catm[i],
                i,
                ocn_uptake,
                rjoos,
                dtocn_i[i + 1],
                ocean_steps_per_model_steps,
            )

            catm[i + 1] = catm[i] + (del_ems - delta_ocn - del_lnd) / PPM2GT

        # Undo the change to smaller timesteps by interpolating back to original time
        # points.
        scc_catm = np.interp(self.esm_data.time, time_i, catm)
        scc_cems = np.interp(self.esm_data.time, time_i, cems)
        scc_cveg = np.interp(self.esm_data.time, time_i, self.land.cveg)
        scc_csoil = np.interp(self.esm_data.time, time_i, self.land.csoil)
        scc_cocn = np.interp(self.esm_data.time, time_i, self.ocean.carbon_increase)
        scc_npp = np.interp(self.esm_data.time, time_i, self.land.npp)
        scc_gam = np.interp(self.esm_data.time, time_i, self.land.lit / self.land.cveg)
        scc_sres = np.interp(
            self.esm_data.time, time_i, self.land.sres / self.land.csoil
        )
        scc_lit = np.interp(self.esm_data.time, time_i, self.land.lit)
        scc_rh = np.interp(self.esm_data.time, time_i, self.land.sres)
        scc_fcvo = np.interp(self.esm_data.time, time_i, self.land.fcva)
        scc_fcso = np.interp(self.esm_data.time, time_i, self.land.fcsa)

        scc_clnd = scc_cveg + scc_csoil
        scc_cdif = (scc_catm * PPM2GT + scc_clnd + scc_cocn) - scc_cems
        scc_cdif = scc_cdif - scc_cdif[0]

        if npp_flag is False:
            scc_gpp = np.interp(self.esm_data.time, time_i, self.land.gpp)
            scc_vres = np.interp(self.esm_data.time, time_i, self.land.vres)
            scc_ra = np.interp(self.esm_data.time, time_i, self.land.vres)

            ans = Data(
                time=self.esm_data.time,
                catm=scc_catm,
                cveg=scc_cveg,
                csoil=scc_csoil,
                clnd=scc_clnd,
                cocn=scc_cocn,
                oflux=np.gradient(scc_cocn),
                cdif=scc_cdif,
                gam=scc_gam,
                resp=scc_sres,
                vres=scc_vres,
                gpp=scc_gpp,
                npp=scc_npp,
                lit=scc_lit,
                ra=scc_ra,
                rh=scc_rh,
                fcvo=scc_fcvo,
                fcso=scc_fcso,
            )
        else:
            ans = Data(
                time=self.esm_data.time,
                catm=scc_catm,
                cveg=scc_cveg,
                csoil=scc_csoil,
                clnd=scc_clnd,
                cocn=scc_cocn,
                oflux=np.gradient(scc_cocn),
                cdif=scc_cdif,
                gam=scc_gam,
                resp=scc_sres,
                npp=scc_npp,
                lit=scc_lit,
                rh=scc_rh,
                fcvo=scc_fcvo,
                fcso=scc_fcso,
            )

        return ans

    def create_plots(self, model, output, npp_flag):
        "Create some plots about the carbon cycle emulation."

        # TODO: go over this function and clean it up.

        # Simple diagnostic plot of emissions, and corresponding SCC catm predictions
        # (with GCM temperature).
        matplotlib.rcParams.update({"font.size": 10})
        plt.figure(1, figsize=(24 / 2.54, 9 / 2.54))
        plt.subplots_adjust(
            top=0.90, bottom=0.14, left=0.08, right=0.98, hspace=0.25, wspace=0.25
        )

        ax = plt.subplot(1, 4, 1)
        plt.plot(
            self.esm_data.time, self.esm_data.gcmemis, color="orange", label="Input"
        )
        plt.plot(
            self.esm_data.time,
            self.esm_data.emis,
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

        # ! Compare the calculated atmospheric C02 concentration with the one from the ESM dataset
        # (To see how well the model is performing)
        ax = plt.subplot(1, 4, 2)
        plt.plot(
            self.esm_data.time,
            self.esm_data.catm,
            color="orange",
            alpha=0.5,
            label="ESM",
        )
        plt.plot(output.time, output.catm, color="dodgerblue", alpha=0.5, label="SCC")
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
        plt.plot(output.time, self.esm_data.catm - output.catm, lw=1.0, color="blue")
        plt.axhline(0.0, ls=":", color="k")
        plt.title("catm error: ESM-SCC")

        # ! Plot the carbon contained in vegetation, soil, atmosphere and (difference wrt
        # initial point) ocean.
        ax = plt.subplot(1, 4, 4)
        plt.plot(output.time, self.esm_data.cveg, label="cveg")
        plt.plot(output.time, self.esm_data.csoil, label="csoil")
        plt.plot(output.time, self.esm_data.catm, label="catm")
        plt.plot(output.time, output.cocn, label="cocn")
        plt.legend()
        plt.show()

        # ! Carbon fluxes
        ax = plt.subplot(2, 4, 1)
        plt.plot(
            self.esm_data.time,
            self.esm_data.catm,
            color="orange",
            alpha=0.5,
            label="ESM",
        )
        plt.plot(output.time, output.catm, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": catm")

        ax = plt.subplot(2, 4, 2)
        plt.plot(
            self.esm_data.time,
            self.esm_data.cveg,
            color="orange",
            alpha=0.5,
            label="ESM",
        )
        plt.plot(output.time, output.cveg, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": cveg")

        ax = plt.subplot(2, 4, 3)
        plt.plot(
            self.esm_data.time,
            self.esm_data.csoil,
            color="orange",
            alpha=0.5,
            label="ESM",
        )
        plt.plot(output.time, output.csoil, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": csoil")

        # Comparing fluxes for the ocean is a bit tricky because the data we are
        # obtianing from the scm is at a different time resolution, so the values
        # are also different. We would need to interpolate back to general scm times
        # instead we compare the cumulative carbon in the ocean
        # Instead, just compute the gradient of the total amount of carbon in the ocean
        # accumulated since the beggining
        # ax=plt.subplot(2,4,4)
        # plt.plot(self.esm_data.time, self.esm_data.oflux,color='orange',alpha=0.5,label='GCM')
        # plt.plot(output.time, np.gradient(output.cocn),color='dodgerblue',alpha=0.5,label='SCC')
        # plt.title(model+': fgco2')

        # To plot total amount of carbon stored in the ocean
        ax = plt.subplot(2, 4, 4)
        plt.plot(
            self.esm_data.time,
            np.cumsum(self.esm_data.oflux),
            color="orange",
            alpha=0.5,
            label="ESM",
        )
        plt.plot(output.time, output.cocn, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": fgco2")

        if npp_flag is False:
            ax = plt.subplot(2, 4, 5)
            plt.plot(
                self.esm_data.time,
                self.esm_data.gpp,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
            plt.plot(
                output.time, output.gpp, color="dodgerblue", alpha=0.5, label="SCC"
            )
            plt.title(model + ": GPP")
            plt.legend()

            ax = plt.subplot(2, 4, 6)
            plt.plot(
                self.esm_data.time,
                self.esm_data.ra,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
            plt.plot(output.time, output.ra, color="dodgerblue", alpha=0.5, label="SCC")
            plt.title(model + ": R_veg")
            plt.legend()
        else:
            ax = plt.subplot(2, 4, 5)
            plt.plot(
                self.esm_data.time,
                self.esm_data.npp,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
            plt.plot(
                output.time, output.npp, color="dodgerblue", alpha=0.5, label="SCC"
            )
            plt.title(model + ": NPP")
            plt.legend()

            ax = plt.subplot(2, 4, 6)
            plt.plot(
                self.esm_data.time,
                self.esm_data.oflux,
                color="orange",
                alpha=0.5,
                label="ESM",
            )
            plt.plot(
                output.time, output.oflux, color="dodgerblue", alpha=0.5, label="SCC"
            )
            plt.title(model + ": fgco2")
            plt.legend()

        ax = plt.subplot(2, 4, 7)
        plt.plot(
            self.esm_data.time,
            self.esm_data.lit,
            color="orange",
            alpha=0.5,
            label="ESM",
        )
        plt.plot(output.time, output.lit, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": L")
        plt.legend()

        ax = plt.subplot(2, 4, 8)
        plt.plot(
            self.esm_data.time, self.esm_data.rh, color="orange", alpha=0.5, label="ESM"
        )
        plt.plot(output.time, output.rh, color="dodgerblue", alpha=0.5, label="SCC")
        plt.title(model + ": R_S")
        plt.legend()

        plt.show()


def main():
    """ "
    Main point of entry to build an instance of the carbon cycle emulator, load the
    data and parameters, and run the emulation of the carbon cycle from a CMIP6 ESM.
    """

    pars_dir = "data/pars"
    scen_dir = "data/scenarios"

    smoothing_algorithm = {"type": "savgol", "pars": [21, 3]}

    # Numerical instability can arise if dtoccmax is too large. We recommend to keep it
    # to 0.2 or less.
    dtpred = 0.03  # main timestep (years) for land carbon cycle
    dtoccmax = 0.03  # max timestep (years) for ocean carbon cycle

    # Calculate emissions from carbon mass conservation across the different carbon pools
    recalc_emis = True

    # Run the cli parser and retrieve the required information
    models, scenario, scenario_pars, era, save, npp_flag = cli_parser()

    # Run emulation for each model
    for imodel, model in enumerate(models):
        # If we are dealing with an SSP scenario, average the first few data
        # points to denoise the first value. However, if dealing with a
        # 1pctco2(-cdr) just take the first value, as subsequent values will
        # already have significantly diverged from equilibrium.
        if scenario in ["1pctco2", "1pctco2-cdr"]:
            ninit_scenario = 1
        else:
            ninit_scenario = 20

        # Load diagnosed ESM data from the data dir
        data_file = Path(__file__).parent / scen_dir / f"sce_{model}_{scenario}.txt"
        print("\nLoading ESM data from: ", data_file)

        esm_data = load_esm_data(
            data_file,
            recalc_emis=recalc_emis,
            ninit=ninit_scenario,
            smoothing_pars=smoothing_algorithm,
        )

        # Load calibrated SCC parameter from saved dict.
        pars_file = (
            Path(__file__).parent / pars_dir / f"sccpar_{model}_{scenario_pars}.txt"
        )
        print("Loading SCC parameters from: ", pars_file)

        with open(pars_file, "r") as infile:
            sccpars = json.load(infile)

        # Measure the time it takes us to run the model
        tbeg = systime.time()

        # Build the model and run it
        cc_emulator = CarbonCycle(esm_data, dtpred, dtoccmax, **{})
        cc_output = cc_emulator.run_full_simulation(npp_flag)

        tend = systime.time()
        print("Time to process:", tend - tbeg)

        cc_emulator.create_plots(model, cc_output, npp_flag)


if __name__ == "__main__":
    main()
