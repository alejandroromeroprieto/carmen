"""
DEPRECATED.
This script was used initially to run a whole-model calibration, but results did not
improve with respect to the flux-specific calibration. Not currently maintained but
leaving as reference in case we want to do something similar in the future.
"""

import json
import numpy as np
import os
import time as systime

from pathlib import Path

from carbon_cycle_model.calibration.utils import (
    Normalizer,
    calculate_cost_whole_scc_cross_experiment,
    create_range_from_guess,
    load_and_prepare_esm_data,
    run_minimisation,
    cost_whole_scc,
)
from carbon_cycle_model.constants import *
from carbon_cycle_model.utils import make_all_dirs


##################################################
####           START MAIN PROCESSING          ####
##################################################

# Version number for the calibration. A different number will prevent produced data overwriting old data.
ccversion = 1

# Scenario family to be used for the calibration
SCENARIO    = 'CMIP6'

# List of ESMs to perform the calibration on. The possible options are:
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
model_list = ['CanESM5']

experiment_list = [
    # "abrupt-4xCO2",
    "1pctco2",
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
    # "hist-noLu",
    # "esm-ssp585",
    # "esm-ssp534-over",
    # "historical",
    # "esm-historical",
]

# =============================   Parameter ranges  =============================
# Here, we define the parameter space to explore in search for the optimum solution

fast_inertia = [-1, 0.2]

#                     gpp_t_l  gpp_t_e  gpp_c_l  gpp_c_half gpp_c_e  gpp_hyst   gpp_c_tan      gpp_fast
gpp_range = np.array([[-3, 5], [-3, 5], [-5, 0], [0, 2000], [-5, 0], [-10, 10], fast_inertia, [0.8, 1]])

#                     lit_t_l  lit_t_e  lit_c_l   sres_c_half  lit_c_e  lit_hyst  lit_c_tan    lit_fast
lit_range = np.array([[-3, 5], [-3, 5], [-0, 5], [0, 2000], [-0, 5], [-10, 10], fast_inertia, [0.8, 1]])

#                      vres_t_l  vres_t_e  vres_c_l  vres_c_half vres_c_e  vres_hyst  vres_c_tan     vres_fast
vres_range = np.array([[-3, 2.0], [-3, 2.5], [-0, 5], [0, 2000], [-0, 5], [-10, 10], fast_inertia, [0.8, 1]])

#                      sres_t_l  sres_t_e  sres_c_l  sres_c_half sres_c_e  sres_hyst  sres_c_tan     sres_fast
sres_range = np.array([[-3, 2.0], [-3, 2.5], [-0, 5], [0, 2000], [-0, 5], [-10, 10], fast_inertia, [0.8, 1]])

#                      npp_t_l  npp_t_e  npp_c_l  npp_c_half npp_c_e  npp_hyst  npp_c_tan     npp_fast
npp_range = np.array([[-3, 5], [-3, 5], [-5, 0], [0, 2000], [-5, 0], [-20, 10], fast_inertia, [0.8, 1]])

#                         docn,     docnfac   ocntemp   docntemp
oflux_range = np.array([[35, 70.0], [-10, 10], [0.00, 0.3], [0, 10]])

all_array = np.concatenate([gpp_range, lit_range, vres_range, sres_range, npp_range, oflux_range])
# scc_range = np.array(all_array)


# Tolerance values for our calibration
f_tol = 1e-5

# Number of decimal places to to for rounding parameter values
nround = 8 

# Get current folder path
CWD = str(Path.cwd())
CWD = os.path.join(CWD, 'src/carbon_cycle_model')

# Directory to store results
OUT_DIR = CWD + '/calibration/calibration_results'

# Path and prefix to all input data files
PREFIX = CWD + "/data/scenarios/"
PREFIX_DETRENDED = CWD + "/data/scenarios/detrended_wrt_decade/"

# Number of times we cycle through the optimization for each component
# to try to avoid potential local minima:
nsamp_scc= 1

# Note we haven't implemented logic to train the model for different realisations yet
realisations = ["default"]

esm_data = {}

for ind, model in enumerate(model_list):
    #===================================================
    #====             CALIBRATION SETUP            =====
    #===================================================
    print('Optimizing for: ', model)
    
    # load parameters
    txtfile = os.path.join(CWD, PARS_DIR, 'sccpar_' + model + '_' + 'cross_experiment' + '.txt')
    with open(txtfile, 'r') as infile:
        sccparam = json.load(infile)

    initial_scc_guess_dim = np.array([
        sccparam['gpp_t_l'],
        sccparam['gpp_t_e'],
        sccparam['gpp_c_l'],
        sccparam['gpp_c_half'],
        sccparam['gpp_c_e'],
        sccparam['gpp_hyst'],
        sccparam['gpp_c_tan'],
        sccparam['gpp_fast'],
        sccparam['lit_t_l'],
        sccparam['lit_t_e'],
        sccparam['lit_c_l'],
        sccparam['lit_c_half'],
        sccparam['lit_c_e'],
        sccparam['lit_hyst'],
        sccparam['lit_c_tan'],
        sccparam['lit_fast'],
        sccparam['vres_t_l'],
        sccparam['vres_t_e'],
        sccparam['vres_c_l'],
        sccparam['vres_c_half'],
        sccparam['vres_c_e'],
        sccparam['vres_hyst'],
        sccparam['vres_c_tan'],
        sccparam['vres_fast'],
        sccparam['sres_t_l'],
        sccparam['sres_t_e'],
        sccparam['sres_c_l'],
        sccparam['sres_c_half'],
        sccparam['sres_c_e'],
        sccparam['sres_hyst'],
        sccparam['sres_c_tan'],
        sccparam['sres_fast'],
        sccparam['npp_t_l'],
        sccparam['npp_t_e'],
        sccparam['npp_c_l'],
        sccparam['npp_c_half'],
        sccparam['npp_c_e'],
        sccparam['npp_hyst'],
        sccparam['npp_c_tan'],
        sccparam['npp_fast'],
        sccparam['docn'],
        sccparam['docnfac'],
        sccparam['ocntemp'],
        sccparam['docntemp']
    ])
    scc_range = create_range_from_guess(initial_scc_guess_dim, 1.5)
    # model_pars is dict of output calibrated parameters
    model_pars={}

    for experiment in experiment_list:
        # Number of years to use to determine the pre-industrial values
        if '1pctco2' in experiment:
            pre_ind_average_length = 1
        elif 'ssp' in experiment:
            pre_ind_average_length = 20
        else:
            raise ValueError('Experiment not recognised')

        if model == "CNRM-ESM2-1" or model == "IPSL-CM6A-LR":
            prefix_to_use = PREFIX_DETRENDED
        else:
            prefix_to_use = PREFIX

        for realisation in realisations:
            if realisation == "default":
                prefix_to_use_real = prefix_to_use + "sce_"
            else:
                prefix_to_use_real = prefix_to_use + "other_realisations/sce_"

        experiment_data = load_and_prepare_esm_data(prefix_to_use_real, model, experiment, recalc_emis=True, smoothing_alg = {"type": "butterworth", "pars": [1]}, ninit=pre_ind_average_length, realisation=realisation, data_output=True)

        if model in esm_data:
            if realisation in esm_data[model]:
                esm_data[model][realisation].update(
                    experiment_data[model][realisation]
                )
            else:
                esm_data[model].update(experiment_data[model])
        else:
            esm_data.update(experiment_data)


    ####################
    # SCC CALIBRATION
    ####################    

    print('')
    print('\t SCC CALIBRATION')


    npar  = 44
    parlo = np.zeros(npar)
    parhi = np.ones(npar)  

    sccNormalizer = Normalizer(pranges=scc_range)
    for i in range(npar):
        if sccNormalizer.pranges[i,0] == sccNormalizer.pranges[i,1]:  parhi[i] = 0.0 

    initial_scc_guess_bar = sccNormalizer.normalise(initial_scc_guess_dim)

    args   = (sccNormalizer, esm_data, model)


    # Temporary: calculate initial cost for initial guess:
    initial_cost = calculate_cost_whole_scc_cross_experiment(
        initial_scc_guess_bar,
        sccNormalizer,
        esm_data,
        model,
    )
    print("Initial cost was: ", initial_cost)

    systime_litbe = systime.time()
    p_bar, costout, _ = run_minimisation(
        calculate_cost_whole_scc_cross_experiment,
        initial_scc_guess_bar,
        args,
        parlo,
        parhi,
        attempts=nsamp_scc,
        ftol=f_tol,
    )    
    systime_litend = systime.time()
    
    pstar = sccNormalizer.inv(p_bar)    
    gpp_t_l, gpp_t_e, gpp_c_l, gpp_c_half, gpp_c_e, gpp_hyst, gpp_c_tan, gpp_fast, lit_t_l, lit_t_e, lit_c_l, lit_c_half, lit_c_e, lit_hyst, lit_c_tan, lit_fast, vres_t_l, vres_t_e, vres_c_l, vres_c_half, vres_c_e, vres_hyst, vres_c_tan, vres_fast, sres_t_l, sres_t_e, sres_c_l, sres_c_half, sres_c_e, sres_hyst, sres_c_tan, sres_fast, npp_t_l, npp_t_e, npp_c_l, npp_c_half, npp_c_e, npp_hyst, npp_c_tan, npp_fast, docn, docnfac, ocntemp, docntemp = pstar


    print('')
    print('\t ============================== SCC SOLUTIONS ==============================')
    print('\t gpp_t_l=', gpp_t_l,' gpp_t_e=', gpp_t_e,' gpp_c_l=', gpp_c_l)
    print('\t gpp_c_half=', gpp_c_half, ' gpp_c_tan=', gpp_c_tan, ' gpp_fast=', gpp_fast)
    print('\t lit_t_l=', lit_t_l,' lit_t_e=', lit_t_e,' lit_c_l=', lit_c_l,' lit_fast=', lit_fast)
    print('\t lit_c_half=', lit_c_half, ' lit_c_tan=', lit_c_tan)
    print('\t vres_t_l=', vres_t_l,' vres_t_e=', vres_t_e,' vres_c_l=', vres_c_l)
    print('\t vres_c_half=', vres_c_half, ' vres_c_tan=', vres_c_tan,' vres_fast=', vres_fast)
    print('\t sres_t_l=', sres_t_l,' sres_t_e=', sres_t_e,' sres_c_l=', sres_c_l)
    print('\t sres_c_half=', sres_c_half, ' sres_c_tan=', sres_c_tan,' sres_fast=', sres_fast)
    print('\t npp_t_l=', npp_t_l,' npp_t_e=', npp_t_e,' npp_c_l=', npp_c_l,' npp_fast=', npp_fast)
    print('\t npp_c_half=', npp_c_half, ' npp_c_tan=', npp_c_tan)
    print('\t docn=', docn,' docnfac=', docnfac)
    print('\t ocntemp=', ocntemp,' docntemp=', docntemp)
    print('\t Time to calculate optimisation: ', systime_litend - systime_litbe)
    print('\t Cost vector: ', costout)
    print('\t ============================== SCC SOLUTIONS ==============================')
    print('')
    
    model_pars['gpp_t_l'] = round(gpp_t_l,  nround)
    model_pars['gpp_t_e'] = round(gpp_t_e,  nround)
    model_pars['gpp_c_l'] = round(gpp_c_l,  nround)
    model_pars['gpp_c_half'] = round(gpp_c_half,  nround)
    model_pars['gpp_c_e'] = round(gpp_c_e,  nround)
    model_pars['gpp_hyst'] = round(gpp_hyst,  nround)
    model_pars['gpp_c_tan'] = round(gpp_c_tan,  nround)
    model_pars['gpp_fast'] = round(gpp_fast,  nround)

    model_pars['lit_t_l'] = round(lit_t_l,  nround)
    model_pars['lit_t_e'] = round(lit_t_e,  nround)
    model_pars['lit_c_l'] = round(lit_c_l,  nround)
    model_pars['lit_c_half'] = round(lit_c_half,  nround)
    model_pars['lit_c_e'] = round(lit_c_e,  nround)
    model_pars['lit_hyst'] = round(lit_hyst,  nround)
    model_pars['lit_c_tan'] = round(lit_c_tan,  nround)
    model_pars['lit_fast'] = round(lit_fast,  nround)

    model_pars['vres_t_l'] = round(vres_t_l,  nround)
    model_pars['vres_t_e'] = round(vres_t_e,  nround)
    model_pars['vres_c_l'] = round(vres_c_l,  nround)
    model_pars['vres_c_half'] = round(vres_c_half,  nround)
    model_pars['vres_c_e'] = round(vres_c_e,  nround)
    model_pars['vres_hyst'] = round(vres_hyst,  nround)
    model_pars['vres_c_tan'] = round(vres_c_tan,  nround)
    model_pars['vres_fast'] = round(vres_fast,  nround)
    
    model_pars['sres_t_l'] = round(sres_t_l,  nround)
    model_pars['sres_t_e'] = round(sres_t_e,  nround)
    model_pars['sres_c_l'] = round(sres_c_l,  nround)
    model_pars['sres_c_half'] = round(sres_c_half,  nround)
    model_pars['sres_c_e'] = round(sres_c_e,  nround)
    model_pars['sres_hyst'] = round(sres_hyst,  nround)
    model_pars['sres_c_tan'] = round(sres_c_tan,  nround)
    model_pars['sres_fast'] = round(sres_fast,  nround)

    model_pars['npp_t_l'] = round(npp_t_l,  nround)
    model_pars['npp_t_e'] = round(npp_t_e,  nround)
    model_pars['npp_c_l'] = round(npp_c_l,  nround)
    model_pars['npp_c_half'] = round(npp_c_half,  nround)
    model_pars['npp_c_e'] = round(npp_c_e,  nround)
    model_pars['npp_hyst'] = round(npp_hyst,  nround)
    model_pars['npp_c_tan'] = round(npp_c_tan,  nround)
    model_pars['npp_fast'] = round(npp_fast,  nround)

    model_pars['docn'] = round(docn,  nround)
    model_pars['docnfac'] = round(docnfac,  nround)
    model_pars['ocntemp'] = round(ocntemp,  nround)
    model_pars['docntemp'] = round(docntemp,  nround)
    
    #===================================================
    #====  OUTPU DATA: PARAMETER VALUES AND PLOTS  =====
    #===================================================

    # Save dict of calibrated parameters    
    outname = SCENARIO.lower() + '/model_pars_'+ model + '_whole_scc.txt'
    txtfile = os.path.join(OUT_DIR, outname)
    o = make_all_dirs(txtfile)     
    with open(txtfile, 'w') as ofile:
        json.dump(model_pars, ofile)  
    print('Written dictionary to json txt file:', txtfile)


