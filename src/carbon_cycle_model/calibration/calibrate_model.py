"""
Script to calibrate individual fluxes for the carbon cycle emulator to a specific set of
ESM output files.

In particular, this script does the following for each model specified in model_list:
- Load the data for all specified experiments in experiment_list
- Derive some pre-industrial quantities from this ESM data.
- Run the calibration for all fluxes individually. This involves:
    - Drawing different parameter value sets, starting from initial guesses specified
      in initial_{flux}_guess_dim, and limited to the interval in {flux}_range.
    - Calculating a similarity measure or "cost" between emulated fluxes with thse
      parameter values and the corresponding ESM flux.
    - Minimise this cost function varying the parameter values. This minimisation is done
      in parallel, via the third-party optimparallel package.
    - Plot some diagnostic plots for the calibration.
    - Store best-fist parameter value set in 'OUT_DIR'.

The fluxes to calibrate can be specified in 'fluxes_to_calibrate', which can be a subset
of:
- GPP (gross primary production)
- litterfall
- vegetation respiration
- soil respiration
- ocean carbon uptake
- NPP (Net primary production)
"""

import json
from pathlib import Path
import time as systime
import os

import numpy as np

from carbon_cycle_model.calibration.utils import (
    Normalizer,
    load_and_prepare_esm_data,
    get_general_average,
    run_minimisation,
    calculate_cost_gen_func_cross_experiment,
    calculate_cost_gen_func,
    prepare_contour_data,
    plot_diagnostic,
    calculate_cost_ocean_cross_experiment,
    docn_func,
    costdocn1,
)
from carbon_cycle_model.utils import Data, make_all_dirs
from carbon_cycle_model.land_component.boxes.utils import general_calibration_fun

# Order for optimizations:
#   1. GPP
#   2. LIT
#   3. VRES
#   4. SRES
#   5. NPP
#   6. OFLUX

# Version number for the calibration. A different number will prevent produced data
# overwriting old data.
CC_VERSION = 1

SCENARIO = "CMIP6"

# List of ESMs to perform the calibration on. Possible options are:
#  - ACCESS-ESM1-5
#  - BCC-CSM2-MR
#  - CanESM5
#  - CESM2
#  - CMCC-ESM2
#  - CNRM-ESM2-1
#  - GFDL-ESM4
#  - IPSL-CM6A-LR
#  - MIROC-ES2L
#  - MPI-ESM1-2-LR
#  - MRI-ESM2-0
#  - NorESM2-LM
#  - UKESM1-0-LL
model_list = ["UKESM1-0-LL"]

# List of scenarios to perform the calibration on. Possible options are:
#  - ssp119
#  - ssp126
#  - ssp245
#  - ssp370
#  - ssp434
#  - ssp460
#  - ssp534-over
#  - ssp585
experiment_list = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp534-over",
    "ssp585",
]


# List of fluxes to perform the calibration on. Possible options are:
# - GPP (Gross Primary Production)
# - lit (litterfall)
# - vres (vegetation_respiration)
# - sres (soil_respiration)
# - npp (Net Primary Production)
# - oflux (air_ocean_flux)
# fluxes_to_calibrate = [
#     "GPP",
#     "litter",
#     "vegetation_respiration",
#     "soil_respiration",
#     "npp",
#     "air_ocean_flux",
# ]
fluxes_to_calibrate = ["air_ocean_flux"]

# =============================   Parameter ranges  =============================
# Here, we define the parameter space to explore in search for the optimum solution

#                     gpp_sts  gpp_ste   gpp_lt  gpp_c_half
gpp_range = np.array([[-3, 5], [-3, 5], [-6, 6], [1, 1000]])

#                     lit_sts  lit_ste  lit_lt   sres_c_half
lit_range = np.array([[-3, 5], [-3, 5], [-6, 6], [1, 1000]])

#                      vres_sts  vres_ste  vres_lt  vres_c_half
vres_range = np.array([[-3, 2.0], [-3, 2.5], [-6, 6], [1, 2000]])

#                      sres_sts  sres_ste  sres_lt  sres_c_half
sres_range = np.array([[-3, 2.0], [-3, 2.5], [0, 6], [0, 1000]])

#                      npp_sts  npp_ste  npp_lt  npp_c_half
npp_range = np.array([[-3, 1], [-3, 4], [-6, 6], [0, 1000]])

#                         docn,     docnfac   ocntemp   docntemp
oflux_range = np.array([[30, 90.0], [-0.2, 0.2], [-1, 0.2], [-2, 0]])


#  ========================  Initial guesses for solutions  ========================
#                      gpp_sts  gpp_ste  gpp_lt  gpp_c_half
initial_gpp_guess_dim = np.array([0, 0, 0, 10])
#                      lit_sts  lit_ste  lit_lt  lit_c_half
initial_lit_guess_dim = np.array([1, 0.3406, 2.7698, 10])
#                      vres_sts  vres_ste  vres_lt  vres_c_half
initial_vres_guess_dim = np.array([0, 0, 0, 10])
#                      sres_sts  sres_ste  sres_lt  sres_c_half
initial_sres_guess_dim = np.array([0.54, 0.15, 0, 10])
#                      npp_sts  npp_ste  npp_lt  npp_c_half
initial_npp_guess_dim = np.array([0, 0, 0, 10])
#                             docn  docnfac  ocntemp  docntemp
initial_oflux_guess_dim = np.array([55, 0.33, 0.05, -0.1])


# Tolerance values for our calibration
F_TOL = 1e-5

# Number of decimal places to to for rounding parameter values
N_ROUND = 8

# Get current folder path
CWD = str(Path.cwd())

# Directory to store results
OUT_DIR = CWD + "/src/carbon_cycle_model/calibration/calibration_results"

# Path and prefix to all input data files
PREFIX = CWD + "/src/carbon_cycle_model/data/scenarios/sce_"

# Number of times we cycle through the optimization for each component
# to try to avoid potential local minima:
NUM_REPEAT = 20
NMSAMP_GPP = NUM_REPEAT
NMSAMP_LIT = NUM_REPEAT
NMSAMP_VRES = NUM_REPEAT
NMSAMP_SRES = NUM_REPEAT
NMSAMP_NPP = NUM_REPEAT
NMSAMP_DOCN = (
    1  # ocean function, and by extension its calibration, is relatively slower
)


# Switch that allows either return of the coefficient gamma from lit_func (RETCOEF=True),
# or return of litter flux gamma*V (RETCOEF=False). Determines which diagnostic is
# plotted.
RETCOEF = True

esm_data = {}

for ind, model in enumerate(model_list):
    # ===================================================
    # ====             CALIBRATION SETUP            =====
    # ===================================================
    print("Optimizing for: ", model)

    # model_pars is dict of output calibrated parameters
    model_pars = {}

    for experiment in experiment_list:
        # Number of years to use to determine the pre-industrial values
        if "1pctco2" in experiment:
            PRE_IND_AVERAGE_LENGTH = 1
        elif "ssp" in experiment:
            PRE_IND_AVERAGE_LENGTH = 20
        else:
            raise ValueError("Experiment not recognised")

        # {"type": "butterworth", "pars": [1] means no smoothing
        expriment_data = load_and_prepare_esm_data(
            PREFIX,
            model,
            experiment,
            recalcEmis=True,
            ninit=PRE_IND_AVERAGE_LENGTH,
            smoothing_alg={"type": "butterworth", "pars": [1]},
        )

        if model in esm_data:
            esm_data[model].update(expriment_data[model])
        else:
            esm_data.update(expriment_data)

    # Since we may be calibrating to several experiments, take an average of the initial
    # quantitites across experiments (even though they should be the same)
    model_pars["model"] = model
    model_pars["cveg0"] = get_general_average(
        esm_data,
        model,
        "cveg0",
        N_ROUND,
    )
    model_pars["csoil0"] = get_general_average(
        esm_data,
        model,
        "csoil0",
        N_ROUND,
    )
    model_pars["catm0"] = get_general_average(
        esm_data,
        model,
        "catm0",
        N_ROUND,
    )
    model_pars["npp0"] = get_general_average(
        esm_data,
        model,
        "npp0",
        N_ROUND,
    )
    model_pars["gpp0"] = get_general_average(
        esm_data,
        model,
        "gpp0",
        N_ROUND,
    )
    model_pars["lu0"] = get_general_average(
        esm_data,
        model,
        "lu0",
        N_ROUND,
    )
    model_pars["lit0"] = get_general_average(
        esm_data,
        model,
        "lit0",
        N_ROUND,
    )
    model_pars["rh0"] = get_general_average(
        esm_data,
        model,
        "rh0",
        N_ROUND,
    )
    model_pars["ra0"] = get_general_average(
        esm_data,
        model,
        "ra0",
        N_ROUND,
    )

    # Make a list of individual fits, and use plot_diagnostic to make some plots.
    fit = []

    # =============================================================
    # ====                   GPP CALIBRATION                  =====
    # =============================================================

    if "GPP" in fluxes_to_calibrate:
        print("")
        print("\t GPP")

        # number of parameters to calibrate: (gpp_sts, gpp_ste , gpp_lt, gpp_c_half)
        NPAR = 4

        # Array with maximum and minimum (normalized) values for the parameters
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)

        # Normalizer object normalizes the initial guess.
        # Subscripts '_bar' here refer to normalized quantities. Values will always be in
        # the range [0,1]
        # Subscripts '_dim' here refer to dimensional quantities.
        gpp_normaliser = Normalizer(pranges=gpp_range)

        # If low and high extremes of the parameter range are the same establish maximum
        # parameter value to 0
        for i in range(NPAR):
            if gpp_normaliser.pranges[i, 0] == gpp_normaliser.pranges[i, 1]:
                parhi[i] = 0.0

        initial_gpp_guess_bar = gpp_normaliser.normalise(initial_gpp_guess_dim)

        # This object gathers the required input for the cost function, which will be fed
        # to that function by run_minimisation()
        calibration_args = (gpp_normaliser, esm_data, model, "gpp0", "cveg", "gpp")

        systime_gppbe = systime.time()

        # Use Nelder-Mead downhill simplex method to values that minimise cost function
        # costgpp1
        p_bar, _costout, _ = run_minimisation(
            calculate_cost_gen_func_cross_experiment,
            initial_gpp_guess_bar,
            calibration_args,
            parlo,
            parhi,
            attempts=NMSAMP_GPP,
            ftol=F_TOL,
        )

        systime_gppend = systime.time()

        # p_bar is the normalized solution. Call inv method of gpp_normaliser to get
        # dimensional quantities.
        gpp_sts, gpp_ste, gpp_lt, gpp_c_half = pstar = gpp_normaliser.inv(p_bar)

        print("")
        print("\t ========================== GPP SOLUTIONS ==========================")
        print("\t gpp_sts=", gpp_sts, " gpp_ste=", gpp_ste)
        print("\t gpp_lt=", gpp_lt, " gpp_c_half=", gpp_c_half)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t ========================== GPP SOLUTIONS ==========================")
        print("")

        # Add solution values to the output dictionary
        model_pars["gpp_sts"] = round(gpp_sts, N_ROUND)
        model_pars["gpp_ste"] = round(gpp_ste, N_ROUND)
        model_pars["gpp_lt"] = round(gpp_lt, N_ROUND)
        model_pars["gpp_c_half"] = round(gpp_c_half, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of GPP
            scc_gpp = general_calibration_fun(
                esm_data[model][experiment]["gpp0"],
                gpp_sts,
                gpp_ste,
                gpp_lt,
                gpp_c_half,
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["cveg"][0],
                esm_data[model][experiment]["catm"][0],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            )

            print("Making GPP plot data for:", model)

            conlabel = ["gpp_sts", "gpp_ste", "gpp_lt", "gpp_c_half"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            # (gpp_sts, gpp_ste, gpp_lt, gpp_c_half)
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][experiment]["gpp0"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["gpp"],
            ]

            # Contour par1 and par2 (gpp_sts, gpp_ste)
            jj = [0, 1]

            par1dim, par2dim, costs = prepare_contour_data(
                par1=par1,
                par2=par2,
                cost_func=calculate_cost_gen_func,
                normaliser=gpp_normaliser,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Contour par1 and par3 (gpp_sts, gpp_lt)
            jj = [2, 3]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par4,
                cost_func=calculate_cost_gen_func,
                normaliser=gpp_normaliser,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Save data for later diagnostic plots
            FLUX_NAME = "GPP"
            xlist = [
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][experiment]["gpp"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                scc_gpp = scc_gpp / esm_data[model][experiment]["cveg"]
                esm_flux = esm_flux / esm_data[model][experiment]["cveg"]

            fit_gpp = Data(
                time=esm_data[model][experiment]["time"],
                xlist=xlist,
                gcm=esm_flux,
                scc=scc_gpp,
                par1arr=par1arr,
                par2arr=par2arr,
                costarr=costarr,
                jarr=jarr,
                pstar=pstar,
                conlabel=conlabel,
                xlabel=xlabel,
                title=title,
                name=FLUX_NAME,
            )

            fit.append(fit_gpp)

    # =============================================================
    # ====            LITTERFALL CALIBRATION                  =====
    # =============================================================

    if "litter" in fluxes_to_calibrate:
        print("")
        print("\t Litterfall")

        # number of parameters to calibrate: (lit_sts, lit_ste , lit_lt, lit_c_half)
        NPAR = 4

        # Array with maximum and minimum (normalized) values for the parameters
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)

        # Normalizer object normalizes the initial guess.
        # Subscripts '_bar' here refer to normalized quantities. Values will always be in
        # the range [0,1]
        # Subscripts '_dim' here refer to dimensional quantities.
        lit_normalizer = Normalizer(pranges=lit_range)

        # If low and high extremes of the parameter range are the same establish maximum
        # parameter value to 0
        for i in range(NPAR):
            if lit_normalizer.pranges[i, 0] == lit_normalizer.pranges[i, 1]:
                parhi[i] = 0.0

        initial_lit_guess_bar = lit_normalizer.normalise(initial_lit_guess_dim)

        # This object gathers the required input for the cost function, which will be fed
        # to that function by run_minimisation()
        calibration_args = (lit_normalizer, esm_data, model, "lit0", "cveg", "lit")

        systime_gppbe = systime.time()

        # Use Nelder-Mead downhill simplex method to values that minimise cost function
        # costgpp1
        p_bar, _costout, _ = run_minimisation(
            calculate_cost_gen_func_cross_experiment,
            initial_lit_guess_bar,
            calibration_args,
            parlo,
            parhi,
            attempts=NMSAMP_LIT,
            ftol=F_TOL,
        )

        systime_gppend = systime.time()

        # p_bar is the normalized solution. Call inv method of gpp_normaliser to get
        # dimensional quantities.
        lit_sts, lit_ste, lit_lt, lit_c_half = pstar = lit_normalizer.inv(p_bar)

        print("")
        print("\t ========================== LIT SOLUTIONS ==========================")
        print("\t lit_sts=", lit_sts, " lit_ste=", lit_ste)
        print("\t lit_lt=", lit_lt, " lit_c_half=", lit_c_half)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t ========================== LIT SOLUTIONS ==========================")
        print("")

        # Add solution values to the output dictionary
        model_pars["lit_sts"] = round(lit_sts, N_ROUND)
        model_pars["lit_ste"] = round(lit_ste, N_ROUND)
        model_pars["lit_lt"] = round(lit_lt, N_ROUND)
        model_pars["lit_c_half"] = round(lit_c_half, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of GPP
            scc_lit = general_calibration_fun(
                esm_data[model][experiment]["lit0"],
                lit_sts,
                lit_ste,
                lit_lt,
                lit_c_half,
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["cveg"][0],
                esm_data[model][experiment]["catm"][0],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            )

            print("Making litterfall plot data for:", model)

            conlabel = ["lit_sts", "lit_ste", "lit_lt", "lit_c_half"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            # (lit_sts, lit_ste, lit_lt, lit_c_half)
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][experiment]["lit0"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["lit"],
            ]

            # Contour par1 and par2 (lit_sts, lit_ste)
            jj = [0, 1]

            par1dim, par2dim, costs = prepare_contour_data(
                par1=par1,
                par2=par2,
                cost_func=calculate_cost_gen_func,
                normaliser=lit_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Contour par1 and par3 (lit_sts, lit_lt)
            jj = [2, 3]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par4,
                cost_func=calculate_cost_gen_func,
                normaliser=lit_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Save data for later diagnostic plots
            FLUX_NAME = "Litter"
            xlist = [
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][experiment]["lit"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                scc_lit = scc_lit / esm_data[model][experiment]["cveg"]
                esm_flux = esm_flux / esm_data[model][experiment]["cveg"]

            fit_litter = Data(
                time=esm_data[model][experiment]["time"],
                xlist=xlist,
                gcm=esm_flux,
                scc=scc_lit,
                par1arr=par1arr,
                par2arr=par2arr,
                costarr=costarr,
                jarr=jarr,
                pstar=pstar,
                conlabel=conlabel,
                xlabel=xlabel,
                title=title,
                name=FLUX_NAME,
            )

            fit.append(fit_litter)

    # =============================================================
    # ====          VEGETATION RESPIRATION CALIBRATION        =====
    # =============================================================

    if "vegetation_respiration" in fluxes_to_calibrate:
        print("")
        print("\t Vegetation respiration")

        # number of parameters to calibrate: (vres_sts, vres_ste , vres_lt, vres_c_half)
        NPAR = 4

        # Array with maximum and minimum (normalized) values for the parameters
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)

        # Normalizer object normalizes the initial guess.
        # Subscripts '_bar' here refer to normalized quantities. Values will always be in
        # the range [0,1]
        # Subscripts '_dim' here refer to dimensional quantities.
        vres_normalizer = Normalizer(pranges=vres_range)

        # If low and high extremes of the parameter range are the same establish maximum
        # parameter value to 0
        for i in range(NPAR):
            if vres_normalizer.pranges[i, 0] == vres_normalizer.pranges[i, 1]:
                parhi[i] = 0.0

        initial_vres_guess_bar = vres_normalizer.normalise(initial_vres_guess_dim)

        # This object gathers the required input for the cost function, which will be fed
        # to that function by run_minimisation()
        calibration_args = (vres_normalizer, esm_data, model, "ra0", "cveg", "ra")

        systime_gppbe = systime.time()

        # Use Nelder-Mead downhill simplex method to values that minimise cost function
        # costgpp1
        p_bar, _costout, _ = run_minimisation(
            calculate_cost_gen_func_cross_experiment,
            initial_vres_guess_bar,
            calibration_args,
            parlo,
            parhi,
            attempts=NMSAMP_VRES,
            ftol=F_TOL,
        )

        systime_gppend = systime.time()

        # p_bar is the normalized solution. Call inv method of gpp_normaliser to get
        # dimensional quantities.
        vres_sts, vres_ste, vres_lt, vres_c_half = pstar = vres_normalizer.inv(p_bar)

        print("")
        print("\t ================= VEGETATION RESPIRATION SOLUTIONS ================")
        print("\t vres_sts=", vres_sts, " vres_ste=", vres_ste)
        print("\t vres_lt=", vres_lt, " vres_c_half=", vres_c_half)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t ================= VEGETATION RESPIRATION SOLUTIONS ================")
        print("")

        # Add solution values to the output dictionary
        model_pars["vres_sts"] = round(vres_sts, N_ROUND)
        model_pars["vres_ste"] = round(vres_ste, N_ROUND)
        model_pars["vres_lt"] = round(vres_lt, N_ROUND)
        model_pars["vres_c_half"] = round(vres_c_half, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of GPP
            scc_vres = general_calibration_fun(
                esm_data[model][experiment]["ra0"],
                vres_sts,
                vres_ste,
                vres_lt,
                vres_c_half,
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["cveg"][0],
                esm_data[model][experiment]["catm"][0],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            )

            print("Making vegetation respiration plot data for:", model)

            conlabel = ["vres_sts", "vres_ste", "vres_lt", "vres_c_half"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            # (vres_sts, vres_ste, vres_lt, vres_c_half)
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][experiment]["ra0"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["ra"],
            ]

            # Contour par1 and par2 (vres_sts, vres_ste)
            jj = [0, 1]

            par1dim, par2dim, costs = prepare_contour_data(
                par1=par1,
                par2=par2,
                cost_func=calculate_cost_gen_func,
                normaliser=vres_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Contour par1 and par3 (vres_sts, vres_lt)
            jj = [2, 3]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par4,
                cost_func=calculate_cost_gen_func,
                normaliser=vres_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Save data for later diagnostic plots
            FLUX_NAME = "sres"
            xlist = [
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][experiment]["ra"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                scc_vres = scc_vres / esm_data[model][experiment]["cveg"]
                esm_flux = esm_flux / esm_data[model][experiment]["cveg"]

            fit_vres = Data(
                time=esm_data[model][experiment]["time"],
                xlist=xlist,
                gcm=esm_flux,
                scc=scc_vres,
                par1arr=par1arr,
                par2arr=par2arr,
                costarr=costarr,
                jarr=jarr,
                pstar=pstar,
                conlabel=conlabel,
                xlabel=xlabel,
                title=title,
                name=FLUX_NAME,
            )

            fit.append(fit_vres)

    # =============================================================
    # ====             SOIL RESPIRATION CALIBRATION           =====
    # =============================================================

    if "soil_respiration" in fluxes_to_calibrate:
        print("")
        print("\t Soil respiration")

        # number of parameters to calibrate: (sres_sts, sres_ste , sres_lt, sres_c_half)
        NPAR = 4

        # Array with maximum and minimum (normalized) values for the parameters
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)

        # Normalizer object normalizes the initial guess.
        # Subscripts '_bar' here refer to normalized quantities. Values will always be in
        # the range [0,1]
        # Subscripts '_dim' here refer to dimensional quantities.
        sres_normalizer = Normalizer(pranges=sres_range)

        # If low and high extremes of the parameter range are the same establish maximum
        # parameter value to 0
        for i in range(NPAR):
            if sres_normalizer.pranges[i, 0] == sres_normalizer.pranges[i, 1]:
                parhi[i] = 0.0

        initial_sres_guess_bar = sres_normalizer.normalise(initial_sres_guess_dim)

        # This object gathers the required input for the cost function, which will be fed
        # to that function by run_minimisation()
        calibration_args = (sres_normalizer, esm_data, model, "rh0", "csoil", "rh")

        systime_gppbe = systime.time()

        # Use Nelder-Mead downhill simplex method to values that minimise cost function
        # costgpp1
        p_bar, _costout, _ = run_minimisation(
            calculate_cost_gen_func_cross_experiment,
            initial_sres_guess_bar,
            calibration_args,
            parlo,
            parhi,
            attempts=NMSAMP_SRES,
            ftol=F_TOL,
        )

        systime_gppend = systime.time()

        # p_bar is the normalized solution. Call inv method of gpp_normaliser to get
        # dimensional quantities.
        sres_sts, sres_ste, sres_lt, sres_c_half = pstar = sres_normalizer.inv(p_bar)

        print("")
        print("\t =================    SOIL RESPIRATION SOLUTIONS    ================")
        print("\t sres_sts=", sres_sts, " sres_ste=", sres_ste)
        print("\t sres_lt=", sres_lt, " sres_c_half=", sres_c_half)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t =================    SOIL RESPIRATION SOLUTIONS    ================")
        print("")

        # Add solution values to the output dictionary
        model_pars["sres_sts"] = round(sres_sts, N_ROUND)
        model_pars["sres_ste"] = round(sres_ste, N_ROUND)
        model_pars["sres_lt"] = round(sres_lt, N_ROUND)
        model_pars["sres_c_half"] = round(sres_c_half, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of GPP
            scc_sres = general_calibration_fun(
                esm_data[model][experiment]["rh0"],
                sres_sts,
                sres_ste,
                sres_lt,
                sres_c_half,
                esm_data[model][experiment]["csoil"],
                esm_data[model][experiment]["csoil"][0],
                esm_data[model][experiment]["catm"][0],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            )

            print("Making soil respiration plot data for:", model)

            conlabel = ["sres_sts", "sres_ste", "sres_lt", "sres_c_half"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            # (sres_sts, sres_ste, sres_lt, sres_c_half)
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][experiment]["rh0"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["csoil"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["rh"],
            ]

            # Contour par1 and par2 (sres_sts, sres_ste)
            jj = [0, 1]

            par1dim, par2dim, costs = prepare_contour_data(
                par1=par1,
                par2=par2,
                cost_func=calculate_cost_gen_func,
                normaliser=sres_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Contour par1 and par3 (sres_sts, sres_lt)
            jj = [2, 3]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par4,
                cost_func=calculate_cost_gen_func,
                normaliser=sres_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Save data for later diagnostic plots
            FLUX_NAME = "sres"
            xlist = [
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][experiment]["rh"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                scc_sres = scc_sres / esm_data[model][experiment]["csoil"]
                esm_flux = esm_flux / esm_data[model][experiment]["csoil"]

            fit_sres = Data(
                time=esm_data[model][experiment]["time"],
                xlist=xlist,
                gcm=esm_flux,
                scc=scc_sres,
                par1arr=par1arr,
                par2arr=par2arr,
                costarr=costarr,
                jarr=jarr,
                pstar=pstar,
                conlabel=conlabel,
                xlabel=xlabel,
                title=title,
                name=FLUX_NAME,
            )

            fit.append(fit_sres)

    # =============================================================
    # ====              NPP RESPIRATION CALIBRATION           =====
    # =============================================================

    if "npp" in fluxes_to_calibrate:
        print("")
        print("\t NPP")

        # number of parameters to calibrate: (npp_sts, npp_ste , npp_lt, npp_c_half)
        NPAR = 4

        # Array with maximum and minimum (normalized) values for the parameters
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)

        # Normalizer object normalizes the initial guess.
        # Subscripts '_bar' here refer to normalized quantities. Values will always be in
        # the range [0,1]
        # Subscripts '_dim' here refer to dimensional quantities.
        npp_normalizer = Normalizer(pranges=npp_range)

        # If low and high extremes of the parameter range are the same establish maximum
        # parameter value to 0
        for i in range(NPAR):
            if npp_normalizer.pranges[i, 0] == npp_normalizer.pranges[i, 1]:
                parhi[i] = 0.0

        initial_npp_guess_bar = npp_normalizer.normalise(initial_npp_guess_dim)

        # This object gathers the required input for the cost function, which will be fed
        # to that function by run_minimisation()
        calibration_args = (npp_normalizer, esm_data, model, "npp0", "cveg", "npp")

        systime_gppbe = systime.time()

        # Use Nelder-Mead downhill simplex method to values that minimise cost function
        # costgpp1
        p_bar, _costout, _ = run_minimisation(
            calculate_cost_gen_func_cross_experiment,
            initial_npp_guess_bar,
            calibration_args,
            parlo,
            parhi,
            attempts=NMSAMP_NPP,
            ftol=F_TOL,
        )

        systime_gppend = systime.time()

        # p_bar is the normalized solution. Call inv method of gpp_normaliser to get
        # dimensional quantities.
        npp_sts, npp_ste, npp_lt, npp_c_half = pstar = npp_normalizer.inv(p_bar)

        print("")
        print("\t =================        NPP SOLUTIONS          ================")
        print("\t npp_sts=", npp_sts, " npp_ste=", npp_ste)
        print("\t npp_lt=", npp_lt, " npp_c_half=", npp_c_half)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t =================        NPP SOLUTIONS          ================")
        print("")

        # Add solution values to the output dictionary
        model_pars["npp_sts"] = round(npp_sts, N_ROUND)
        model_pars["npp_ste"] = round(npp_ste, N_ROUND)
        model_pars["npp_lt"] = round(npp_lt, N_ROUND)
        model_pars["npp_c_half"] = round(npp_c_half, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of GPP
            scc_npp = general_calibration_fun(
                esm_data[model][experiment]["npp0"],
                npp_sts,
                npp_ste,
                npp_lt,
                npp_c_half,
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["cveg"][0],
                esm_data[model][experiment]["catm"][0],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            )

            print("Making npp plot data for:", model)

            conlabel = ["npp_sts", "npp_ste", "npp_lt", "npp_c_half"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            # (npp_sts, npp_ste, npp_lt, npp_c_half)
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][experiment]["npp0"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["cveg"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["npp"],
            ]

            # Contour par1 and par2 (npp_sts, npp_ste)
            jj = [0, 1]

            par1dim, par2dim, costs = prepare_contour_data(
                par1=par1,
                par2=par2,
                cost_func=calculate_cost_gen_func,
                normaliser=npp_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Contour par1 and par3 (npp_sts, npp_lt)
            jj = [2, 3]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par4,
                cost_func=calculate_cost_gen_func,
                normaliser=npp_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=esm_vals,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Save data for later diagnostic plots
            FLUX_NAME = "npp"
            xlist = [
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][experiment]["npp"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                scc_npp = scc_npp / esm_data[model][experiment]["cveg"]
                esm_flux = esm_flux / esm_data[model][experiment]["cveg"]

            fit_npp = Data(
                time=esm_data[model][experiment]["time"],
                xlist=xlist,
                gcm=esm_flux,
                scc=scc_npp,
                par1arr=par1arr,
                par2arr=par2arr,
                costarr=costarr,
                jarr=jarr,
                pstar=pstar,
                conlabel=conlabel,
                xlabel=xlabel,
                title=title,
                name=FLUX_NAME,
            )

            fit.append(fit_npp)

    ####################
    # 6. OFLUX
    ####################

    if "air_ocean_flux" in fluxes_to_calibrate:
        # Four parameters: docn, docnfac, ocntemp, docntemp

        # Warning - for large timesteps (eg 1 year), small docn, and higher forcing
        # scenarios, we can get numerical instability in the model predictions for
        # ocean carbon flux. This is a feature of IMOGEN and the implementation of this
        # in scmpy. To resolve this, follow IMOGEN and interpolate to finer time
        # resolution in the SCC, with a specified timestep DTIME_0. A value of 0.1 years
        # seems to be ok. This makes optimizing for docn rather slow.

        print("")
        print("\t OFLUX")

        NPAR = 4
        DTIME_0 = 0.03  # Same as the one used to run the model prognostically
        # NOTE - for docn, fminBoundedSample is slow, and since resampling the optimized
        # solution invariably gives the same solution for these variables, do not
        # resample, so set NMSAMP_DOCN=0.

        docn_normalizer = Normalizer(pranges=oflux_range)

        initial_docn_guess_bar = docn_normalizer.normalise(initial_oflux_guess_dim)

        p0_dim = oflux_range.mean(axis=1)
        p0_bar = docn_normalizer.normalise(initial_oflux_guess_dim)
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)
        for i in range(NPAR):
            if docn_normalizer.pranges[i, 0] == docn_normalizer.pranges[i, 1]:
                parhi[i] = 0.0

        calibration_args = (docn_normalizer, esm_data, model, DTIME_0)

        print("Please be patient, this step takes a while...")
        # raise AssertionError('Stop for debugging...')
        systime_docnbeg = systime.time()

        p_bar, _costout, _ = run_minimisation(
            calculate_cost_ocean_cross_experiment,
            initial_docn_guess_bar,
            calibration_args,
            parlo,
            parhi,
            attempts=NMSAMP_DOCN,
            ftol=F_TOL,
        )

        systime_docnend = systime.time()

        docn, docnfac, ocntemp, docntemp = pstar = docn_normalizer.inv(p_bar)

        model_pars["docn"] = round(docn, N_ROUND)
        model_pars["docnfac"] = round(docnfac, N_ROUND)
        model_pars["ocntemp"] = round(ocntemp, N_ROUND)
        model_pars["docntemp"] = round(docntemp, N_ROUND)

        print("")
        print("\t =================  OCEAN FLUX SOLUTIONS  ================")
        print("\t docn=", docn, " docnfac=", docnfac)
        print("ocntemp=", ocntemp, "docntemp=", docntemp)
        print("Time to optimize costdocn: ", systime_docnend - systime_docnbeg)
        print("\t =================  OCEAN FLUX SOLUTIONS  ================")
        print("")

        num_steps = round(
            1
            + (
                esm_data[model][experiment]["time"][-1]
                - esm_data[model][experiment]["time"][0]
            )
            / DTIME_0
        )
        for experiment in experiment_list:
            scc_oflux = docn_func(
                esm_data[model][experiment]["catm0"],
                DTIME_0,
                num_steps,
                docn,
                docnfac,
                ocntemp,
                docntemp,
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["dtocn"],
            )

            print("Making OFLUX plot data for:", model)
            conlabel = ["docn", "docnfac", "ocntemp", "docntemp"]

            N4CON = 15  # slow, so don't do too many

            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Contour par1 and par2
            jj = [0, 1]
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)

            gcm_values = [
                esm_data[model][experiment]["catm0"],
                DTIME_0,
                num_steps,
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["oflux"],
                esm_data[model][experiment]["dtocn"],
            ]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par1,
                par2=par2,
                cost_func=costdocn1,
                normaliser=docn_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=gcm_values,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            # Contour par1 and par2
            jj = [2, 3]
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)

            gcm_values = [
                esm_data[model][experiment]["catm0"],
                DTIME_0,
                num_steps,
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["catm"],
                esm_data[model][experiment]["oflux"],
                esm_data[model][experiment]["dtocn"],
            ]

            par1dim, par2dim, costs = prepare_contour_data(
                par1=par1,
                par2=par2,
                cost_func=costdocn1,
                normaliser=docn_normalizer,
                normaliser_par_ind=jj,
                nor_sol=p_bar,
                esm_vals=gcm_values,
            )

            par1arr.append(par1dim)
            par2arr.append(par2dim)
            costarr.append(costs)
            jarr.append(jj)

            #####
            FLUX_NAME = r"$f_o$"
            xlist = [
                esm_data[model][experiment]["time"],
                esm_data[model][experiment]["dtglb"],
                esm_data[model][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]

            gcm = esm_data[model][experiment]["oflux"]

            fit_oflux = Data(
                time=esm_data[model][experiment]["time"],
                xlist=xlist,
                gcm=gcm,
                scc=scc_oflux,
                par1arr=par1arr,
                par2arr=par2arr,
                costarr=costarr,
                jarr=jarr,
                pstar=pstar,
                conlabel=conlabel,
                xlabel=xlabel,
                title=title,
                name=FLUX_NAME,
            )

            fit.append(fit_oflux)

    # ===================================================
    # ====  OUTPUT DATA: PARAMETER VALUES AND PLOTS  =====
    # ===================================================

    # Make and save diagnostic plots
    for i, experiment in enumerate(experiment_list):
        outname = (
            f"ALL/optscc_v{CC_VERSION}"
            + "_cmip6_"
            + SCENARIO.lower()
            + "_"
            + model
            + "_general_"
            + experiment
            + ".pdf"
        )
        outplot = os.path.join(OUT_DIR, outname)

        # Make all sub-directories for outplot file if they don't already exist
        o = make_all_dirs(outplot)
        print("outplot: ", outplot)

        plot_diagnostic(
            OUT_DIR,
            esm_data[model][experiment],
            fit[i :: len(experiment_list)],
            outplot=outplot,
            fontsize=10,
            xsize=16.0,
            ysize=22.0,
        )

    # Save dict of calibrated parameters
    outname = SCENARIO.lower() + "/model_pars_" + model + ".txt"
    txtfile = os.path.join(OUT_DIR, outname)
    o = make_all_dirs(txtfile)

    with open(txtfile, "w", encoding="utf-8") as ofile:
        json.dump(model_pars, ofile)

    print("Written dictionary to json txt file:", txtfile)
