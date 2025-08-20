"""
Bayesian calibration modification for the carbon cycle model.
This example shows how to convert the GPP calibration to a Bayesian approach using emcee.
"""

import json
import os
import time as systime
from pathlib import Path

import numpy as np

import emcee
import corner

from carbon_cycle_model.calibration.utils import (
    Normalizer,
    calculate_cost_gen_func,
    calculate_cost_gen_func_cross_experiment,
    calculate_cost_ocean_cross_experiment,
    costdocn1,
    docn_func,
    load_and_prepare_esm_data,
    plot_diagnostic,
    prepare_contour_data,
    run_minimisation,
)
from carbon_cycle_model.land_component.boxes.utils import general_calibration_fun
from carbon_cycle_model.utils import Data, make_all_dirs

import matplotlib.pyplot as plt
from multiprocessing import Pool


import os

def log_prior(params, param_ranges):
    """
    Log prior probability function.
    Using uniform priors within the specified parameter ranges.
    
    Args:
        params: Parameter vector (normalized, 0-1)
        param_ranges: Parameter ranges from your existing code
    
    Returns:
        Log prior probability
    """
    # Check if all parameters are within bounds [0, 1]
    if np.any(params < 0) or np.any(params > 1):
        return -np.inf
    
    # For uniform priors, log prior is constant (0) within bounds
    return 0.0

def log_likelihood(params, normalizer, esm_data, model, flux0, stock, esm_flux, noise_std=None):
    """
    Log likelihood function for Bayesian calibration.
    
    Args:
        params: Parameter vector (normalized, 0-1)
        normalizer: Your existing normalizer object
        esm_data: ESM data dictionary
        model: Model name
        flux0, stock, esm_flux: Flux identifiers as in your original code
        noise_std: Standard deviation of observational noise (estimated if None)
    
    Returns:
        Log likelihood
    """
    # Convert normalized parameters to dimensional
    par_t_l, par_t_e, par_c_l, par_c_half, par_c_e, par_hyst, par_c_tan, par_fast = normalizer.inv(params)
    
    total_log_likelihood = 0.0
    
    # Loop over all realizations and experiments
    for realisation in esm_data[model].keys():
        for experiment in esm_data[model][realisation].keys():
            # Get model prediction
            scc_val = general_calibration_fun(
                esm_data[model][realisation][experiment][flux0] / esm_data[model][realisation][experiment][stock][0],
                par_t_l, par_t_e, par_c_l, par_c_half, par_c_e, par_hyst, par_c_tan, par_fast,
                esm_data[model][realisation][experiment][stock] - esm_data[model][realisation][experiment][stock][0],
                esm_data[model][realisation][experiment][stock][0],
                esm_data[model][realisation][experiment]["catm"][0],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            )
            
            # Convert to absolute flux if needed (same logic as your original code)
            if esm_flux == "gpp" or esm_flux == "npp":
                scc_val = scc_val * esm_data[model][realisation][experiment][stock][0]
            else:
                scc_val = scc_val * esm_data[model][realisation][experiment][stock]
            
            # Observed data
            obs_flux = esm_data[model][realisation][experiment][esm_flux]
            
            # Estimate noise standard deviation if not provided
            if noise_std is None:
                residuals = scc_val - obs_flux
                noise_std_local = np.std(residuals)
            else:
                noise_std_local = noise_std
            
            # Avoid division by zero
            if noise_std_local <= 0:
                noise_std_local = 1e-6
            
            # Calculate log likelihood (assuming Gaussian errors)
            log_like = -0.5 * np.sum((scc_val - obs_flux)**2 / noise_std_local**2)
            log_like -= len(obs_flux) * np.log(noise_std_local * np.sqrt(2 * np.pi))
            
            total_log_likelihood += log_like
    
    return total_log_likelihood

def log_posterior(params, normalizer, esm_data, model, flux0, stock, esm_flux, param_ranges):
    """
    Log posterior probability function.
    
    Returns:
        Log posterior probability
    """
    # Calculate log prior
    lp = log_prior(params, param_ranges)
    if not np.isfinite(lp):
        return -np.inf
    
    # Calculate log likelihood
    try:
        ll = log_likelihood(params, normalizer, esm_data, model, flux0, stock, esm_flux)
        if not np.isfinite(ll):
            return -np.inf
    except:
        return -np.inf
    
    return lp + ll

def run_bayesian_calibration_gpp(esm_data, model, gpp_range, initial_gpp_guess_dim, 
                                nwalkers=32, nsteps=5000, burn_in=1000):
    """
    Run Bayesian calibration for GPP using MCMC.
    
    Args:
        esm_data: Your ESM data dictionary
        model: Model name
        gpp_range: Parameter ranges for GPP
        initial_gpp_guess_dim: Initial guess in dimensional units
        nwalkers: Number of MCMC walkers
        nsteps: Number of MCMC steps
        burn_in: Number of burn-in steps to discard
    
    Returns:
        samples: MCMC samples from posterior distribution
        sampler: emcee sampler object
        gpp_normalizer: Parameter normalizer
    """
    
    print(f"\nRunning Bayesian calibration for GPP, model: {model}")
    
    # Set up normalizer (same as your original code)
    NPAR = 8
    gpp_normalizer = Normalizer(pranges=gpp_range)
    
    # Convert initial guess to normalized space
    initial_gpp_guess_bar = gpp_normalizer.normalise(initial_gpp_guess_dim)
    
    # Set up calibration arguments
    calibration_args = (gpp_normalizer, esm_data, model, "gpp0", "cveg", "gpp", gpp_range)
    
    # Initialize walkers around the initial guess
    ndim = len(initial_gpp_guess_bar)
    pos = initial_gpp_guess_bar + 0.1 * np.random.randn(nwalkers, ndim)
    
    # Ensure all walker positions are within bounds [0, 1]
    pos = np.clip(pos, 0.01, 0.99)
    
    # Set up the sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, 
            args=calibration_args,
            pool=pool
        )
        
        print("Running MCMC...")
        sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Extract samples after burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    # Convert samples back to dimensional space
    samples_dim = np.array([gpp_normalizer.inv(sample) for sample in samples])
    
    print(f"MCMC completed. Shape of samples: {samples_dim.shape}")
    
    return samples_dim, sampler, gpp_normalizer

def run_bayesian_calibration_litter(esm_data, model, lit_range, initial_lit_guess_dim, 
                                nwalkers=32, nsteps=5000, burn_in=1000):
    """
    Run Bayesian calibration for litter using MCMC.
    
    Args:
        esm_data: Your ESM data dictionary
        model: Model name
        lit_range: Parameter ranges for litter
        initial_lit_guess_dim: Initial guess in dimensional units
        nwalkers: Number of MCMC walkers
        nsteps: Number of MCMC steps
        burn_in: Number of burn-in steps to discard
    
    Returns:
        samples: MCMC samples from posterior distribution
        sampler: emcee sampler object
        lit_normalizer: Parameter normalizer
    """
    
    print(f"\nRunning Bayesian calibration for litter, model: {model}")
    
    # Set up normalizer (same as your original code)
    NPAR = 8
    lit_normalizer = Normalizer(pranges=lit_range)
    
    # Convert initial guess to normalized space
    initial_lit_guess_bar = lit_normalizer.normalise(initial_lit_guess_dim)
    ndim = len(initial_lit_guess_bar)
    
    # Set up calibration arguments
    calibration_args = (lit_normalizer, esm_data, model, "lit0", "cveg", "lit", lit_range)
    
    # # Initialize walkers around the initial guess
    # pos = initial_lit_guess_bar + 0.1 * np.random.randn(nwalkers, ndim)

    # Start walkers at random
    pos = np.array([
        np.random.uniform(low, high, size=nwalkers)
        for (low, high) in lit_normalizer.normalise(lit_range)
    ]).T

    # Ensure all walker positions are within bounds [0, 1]
    pos = np.clip(pos, 0.01, 0.99)
    
    # Set up the sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, 
            args=calibration_args,
            pool=pool
        )
        
        print("Running MCMC...")
        sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Extract samples after burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    # Convert samples back to dimensional space
    samples_dim = np.array([lit_normalizer.inv(sample) for sample in samples])
    
    print(f"MCMC completed. Shape of samples: {samples_dim.shape}")
    
    return samples_dim, sampler, lit_normalizer

def analyze_bayesian_results(samples_dim, param_names, output_dir, model_name, flux_name="GPP"):
    """
    Analyze and plot results from Bayesian calibration.
    
    Args:
        samples_dim: MCMC samples in dimensional space
        param_names: Names of parameters
        output_dir: Output directory for plots
        model_name: Name of the model
        flux_name: Name of the flux being calibrated
    """
    
    # Calculate summary statistics
    print(f"\n=== Bayesian Calibration Results for {flux_name} ===")
    
    percentiles = [16, 50, 84]  # For median and 1-sigma bounds
    
    for i, param_name in enumerate(param_names):
        values = np.percentile(samples_dim[:, i], percentiles)
        median, lower, upper = values[1], values[0], values[2]
        print(f"{param_name}: {median:.6f} +{upper-median:.6f} -{median-lower:.6f}")
    
    # Create corner plot
    plt.figure(figsize=(15, 15))
    fig = corner.corner(
        samples_dim, 
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    
    plt.suptitle(f'Posterior Distributions - {model_name} {flux_name}', fontsize=16)
    plt.savefig(f'{output_dir}/{flux_name}_corner_plot_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create trace plots
    fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 2*len(param_names)))
    
    for i, (ax, param_name) in enumerate(zip(axes, param_names)):
        ax.plot(samples_dim[:, i], alpha=0.7)
        ax.set_ylabel(param_name)
        ax.set_xlabel('Step')
    
    plt.suptitle(f'MCMC Traces - {model_name} {flux_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{flux_name}_traces_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {param: {"median": np.percentile(samples_dim[:, i], 50),
                    "std": np.std(samples_dim[:, i]),
                    "ci_lower": np.percentile(samples_dim[:, i], 16),
                    "ci_upper": np.percentile(samples_dim[:, i], 84)}
            for i, param in enumerate(param_names)}

def save_bayesian_parameters(results_dict, model_name, output_dir):
    """
    Save Bayesian calibration results to JSON file.
    """
    import json
    
    # Add model name to results
    results_dict["model"] = model_name
    results_dict["calibration_type"] = "bayesian"
    
    outname = f"bayesian_model_pars_{model_name}.json"
    txtfile = os.path.join(output_dir, outname)
    
    with open(txtfile, "w", encoding="utf-8") as ofile:
        json.dump(results_dict, ofile, indent=2)
    
    print(f"Bayesian parameters saved to: {txtfile}")

# Example of how to modify your main calibration loop:
def bayesian_calibration_example():
    """
    Example showing how to replace the GPP calibration section with Bayesian approach.
    """
    
    # ... (your existing setup code) ...
    
    # =============================================================
    # ====              BAYESIAN GPP CALIBRATION              ====
    # =============================================================
    
    if "GPP" in fluxes_to_calibrate:
        print("\n\t Bayesian GPP Calibration")
        
        # Run Bayesian calibration
        samples_dim, sampler, gpp_normalizer = run_bayesian_calibration_gpp(
            esm_data, model, gpp_range, initial_gpp_guess_dim,
            nwalkers=32, nsteps=5000, burn_in=1000
        )
        
        # Analyze results
        param_names = ["gpp_t_l", "gpp_t_e", "gpp_c_l", "gpp_c_half", 
                      "gpp_c_e", "gpp_hyst", "gpp_c_tan", "gpp_fast"]
        
        results = analyze_bayesian_results(
            samples_dim, param_names, OUT_DIR, model, "GPP"
        )
        
        # Save results
        save_bayesian_parameters(results, model, OUT_DIR)
        
        # For compatibility with your existing plotting code, you can use the median values
        median_params = [results[param]["median"] for param in param_names]
        gpp_t_l, gpp_t_e, gpp_c_l, gpp_c_half, gpp_c_e, gpp_hyst, gpp_c_tan, gpp_fast = median_params
        
        print(f"\nMedian parameter values:")
        for param, value in zip(param_names, median_params):
            print(f"{param} = {value:.6f}")
        
        # You can continue with your existing plotting code using median values
        # ... (rest of your plotting code) ...

# Additional utility functions for prediction with uncertainty

def predict_with_uncertainty(samples_dim, normalizer, esm_data_experiment, flux0, stock, n_samples=1000):
    """
    Make predictions with uncertainty quantification using posterior samples.
    
    Returns:
        predictions: Array of predictions for each posterior sample
        percentiles: Dictionary with prediction percentiles
    """
    # Randomly sample from posterior
    if len(samples_dim) > n_samples:
        idx = np.random.choice(len(samples_dim), n_samples, replace=False)
        selected_samples = samples_dim[idx]
    else:
        selected_samples = samples_dim
    
    predictions = []
    
    for sample in selected_samples:
        par_t_l, par_t_e, par_c_l, par_c_half, par_c_e, par_hyst, par_c_tan, par_fast = sample
        
        pred = general_calibration_fun(
            esm_data_experiment[flux0] / esm_data_experiment[stock][0],
            par_t_l, par_t_e, par_c_l, par_c_half, par_c_e, par_hyst, par_c_tan, par_fast,
            esm_data_experiment[stock] - esm_data_experiment[stock][0],
            esm_data_experiment[stock][0],
            esm_data_experiment["catm"][0],
            esm_data_experiment["dtglb"],
            esm_data_experiment["catm"],
        )
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate percentiles
    percentiles = {
        'median': np.percentile(predictions, 50, axis=0),
        'ci_lower': np.percentile(predictions, 16, axis=0),
        'ci_upper': np.percentile(predictions, 84, axis=0),
        'ci_95_lower': np.percentile(predictions, 2.5, axis=0),
        'ci_95_upper': np.percentile(predictions, 97.5, axis=0),
    }
    
    return predictions, percentiles


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


model_experiment_dict = {
    # "ACCESS-ESM1-5":    ["1pctco2", "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
    # "BCC-CSM2-MR":      ["1pctco2", "ssp126", "ssp245", "ssp370", "ssp585"],
    # "CanESM5":          ["1pctco2", "ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"],
    # "CESM2":            ["1pctco2", "ssp126", "ssp245", "ssp370", "ssp585"],
    # "CMCC-ESM2":        ["1pctco2", "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
    # "CNRM-ESM2-1":      ["1pctco2", "ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"],
    # "GFDL-ESM4":        ["1pctco2", "ssp126", "ssp370"],
    # "IPSL-CM6A-LR":     ["1pctco2", "ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"],
    # "MIROC-ES2L":       ["1pctco2", "ssp119", "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
    # "MPI-ESM1-2-LR":    ["1pctco2", "ssp119", "ssp126", "ssp245", "ssp370", "esm-ssp534-over", "ssp585"],#, "flat10"], # "abrupt-4xCO2"],
    # "MRI-ESM2-0":       ["1pctco2", "ssp585"],
    # "NorESM2-LM":       ["1pctco2", "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
    "UKESM1-0-LL":      ["1pctco2", "ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp534-over", "ssp585"], # "abrupt-4xCO2"],
}


# List of fluxes to perform the calibration on. Possible options are:
# - GPP (Gross Primary Production)
# - lit (litterfall)
# - vres (vegetation_respiration)
# - sres (soil_respiration)
# - npp (Net Primary Production)
# - oflux (air_ocean_flux)

fluxes_to_calibrate = [
    # "GPP",
    "litter",
    # "vegetation_respiration",
    # "soil_respiration",
    # "npp",
    # "air_ocean_flux",
]

# realisations = ["default", "r2i1p1f2", "r3i1p1f2", "r4i1p1f2", "r8i1p1f2"]
realisations = ["default"]

# =============================   Parameter ranges  =============================
# Here, we define the parameter space to explore in search for the optimum solution

# -l_t values are constraint to -1, 0 to ensure a stable solution. I found a solution
# for CMCC with sres = -5 and it was unstable

fast_inertia = [-1, 0.2]

# Each flux has the following associated parameters:
# x_t_l: linear dependence on temperature
# x_t_e: exponential dependence on temperature
# x_c_l: linear dependence on carbon anomaly
# x_c_e: exponential dependence on carbon anomaly
# x_c_half: dependence on atmospheric carbon
# x_hyst: linear scaling of hysteresis factor
# x_c_tan (fast inertia): linear scaling of inertia factor
# x_fast: inertia memory parameter for the exponential moving average

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


#  ========================  Initial guesses for solutions  ========================
initial_gpp_guess_dim = np.array([0, 0, 0, 10, 0, 0, 0, 0.8])
initial_lit_guess_dim = np.array([0, 0, 0, 10, 0, 0, 0, 0.8])
initial_vres_guess_dim = np.array([0, 0, 0, 10, 0, 0, 0, 0.8])
initial_sres_guess_dim = np.array([0, 0, 0, 10, 0, 0, 0, 0.8])
initial_npp_guess_dim = np.array([0, 0, 0, 10, 0, 0, 0, 0.8])
initial_oflux_guess_dim = np.array([50.9, 0, 0.0423, -0.1])


# Tolerance values for our calibration
F_TOL = 1e-5

# Number of decimal places to to for rounding parameter values
N_ROUND = 8

# Get current folder path
CWD = str(Path.cwd())

# Directory to store results
OUT_DIR = CWD + "/src/carbon_cycle_model/calibration/calibration_results"

# Path and prefix to all input data files
PREFIX = CWD + "/src/carbon_cycle_model/data/scenarios/"
PREFIX_DETRENDED = CWD + "/src/carbon_cycle_model/data/scenarios/detrended_wrt_decade/"

# Number of times we cycle through the optimization for each component
# to try to avoid potential local minima:
NUM_REPEAT = 20
NMSAMP_GPP = NUM_REPEAT
NMSAMP_LIT = NUM_REPEAT
NMSAMP_VRES = NUM_REPEAT
NMSAMP_SRES = NUM_REPEAT
NMSAMP_NPP = NUM_REPEAT
# ocean function, and by extension its calibration, is relatively slower
# So you may want to choose a smaller value here
NMSAMP_DOCN = 10

# Switch that allows either return of the flux coefficient (i.e. flux/stock)
# (RETCOEF=True), or return of absolute flux (RETCOEF=False). It also
# determines which diagnostic is plotted.
RETCOEF = False

esm_data = {}

for model, experiment_list in model_experiment_dict.items():
    # ===================================================
    # ====             CALIBRATION SETUP            =====
    # ===================================================
    print("Optimizing for: ", model)

    # model_pars is dict of output calibrated parameters
    model_pars = {}

    for experiment in experiment_list:
        # How to derive pre-industrial values
        if model in ["BCC-CSM2-MR", "CESM2", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC-ES2L", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM", "UKESM1-0-LL"]:
            if "1pctco2" in experiment:
                PRE_IND_AVERAGE_LENGTH = 1
            elif "ssp" in experiment or "hist" in experiment:
                PRE_IND_AVERAGE_LENGTH = 20
            elif "abrupt" in experiment:
                PRE_IND_AVERAGE_LENGTH = 1
            elif "flat10" in experiment:
                PRE_IND_AVERAGE_LENGTH = 1
            else:
                raise ValueError(f"Experiment {experiment} not recognised")

            pre_ind_algorithm = {"type": "average", "length": PRE_IND_AVERAGE_LENGTH}

        elif model in ["ACCESS-ESM1-5", "CanESM5", "CMCC-ESM2", "CNRM-ESM2-1"]:
            pre_ind_algorithm = {"type": "butterworth", "length": [10]}

        else:
            raise ValueError(f"Model {model} not recognised")

        if model == "CNRM-ESM2-1" or model == "IPSL-CM6A-LR":
            prefix_to_use = PREFIX_DETRENDED
        else:
            prefix_to_use = PREFIX

        for realisation in realisations:
            if realisation == "default":
                prefix_to_use_real = prefix_to_use + "sce_"
            else:
                prefix_to_use_real = prefix_to_use + "other_realisations/sce_"

            # {"type": "butterworth", "pars": [1] means no smoothing
            expriment_data = load_and_prepare_esm_data(
                prefix_to_use_real,
                model,
                experiment,
                recalc_emis=True,
                ninit=pre_ind_algorithm,
                # smoothing_alg={"type": "butterworth", "pars": [1]},
                # Different available smoothing options
                # smoothing_alg={"type": "savgol", "pars": [21, 3]},
                smoothing_alg={"type": "butterworth", "pars": [1]},
                realisation=realisation,
            )

            if model in esm_data:
                if realisation in esm_data[model]:
                    esm_data[model][realisation].update(
                        expriment_data[model][realisation]
                    )
                else:
                    esm_data[model].update(expriment_data[model])
            else:
                esm_data.update(expriment_data)

    model_pars["model"] = model

    # Make a list of individual fits, and use plot_diagnostic to make some plots.
    fit = []

    # =============================================================
    # ====                   GPP CALIBRATION                  =====
    # =============================================================

    if "GPP" in fluxes_to_calibrate:
        print("\n\t Bayesian GPP Calibration")
        
        # Run Bayesian calibration
        samples_dim, sampler, gpp_normalizer = run_bayesian_calibration_gpp(
            esm_data, model, gpp_range, initial_gpp_guess_dim,
            nwalkers=32, nsteps=5000, burn_in=1000
        )
        
        # Analyze results
        param_names = ["gpp_t_l", "gpp_t_e", "gpp_c_l", "gpp_c_half", 
                      "gpp_c_e", "gpp_hyst", "gpp_c_tan", "gpp_fast"]
        
        results = analyze_bayesian_results(
            samples_dim, param_names, OUT_DIR, model, "GPP"
        )
        
        # Save results
        save_bayesian_parameters(results, model, OUT_DIR)
        
        # For compatibility with your existing plotting code, you can use the median values
        median_params = [results[param]["median"] for param in param_names]
        gpp_t_l, gpp_t_e, gpp_c_l, gpp_c_half, gpp_c_e, gpp_hyst, gpp_c_tan, gpp_fast = median_params
        
        print(f"\nMedian parameter values:")
        for param, value in zip(param_names, median_params):
            print(f"{param} = {value:.6f}")

        print("")
        print("\t ========================== GPP SOLUTIONS ==========================")
        print("\t gpp_t_l=", gpp_t_l, " gpp_t_e=", gpp_t_e)
        print("\t gpp_c_l=", gpp_c_l, " gpp_c_half=", gpp_c_half, " gpp_c_e=", gpp_c_e)
        print("\t gpp_hyst=", gpp_hyst, " gpp_c_tan=", gpp_c_tan, " gpp_fast=", gpp_fast)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t ========================== GPP SOLUTIONS ==========================")
        print("")

        # Add solution values to the output dictionary
        model_pars["gpp_t_l"] = round(gpp_t_l, N_ROUND)
        model_pars["gpp_t_e"] = round(gpp_t_e, N_ROUND)
        model_pars["gpp_c_l"] = round(gpp_c_l, N_ROUND)
        model_pars["gpp_c_half"] = round(gpp_c_half, N_ROUND)
        model_pars["gpp_c_e"] = round(gpp_c_e, N_ROUND)
        model_pars["gpp_hyst"] = round(gpp_hyst, N_ROUND)
        model_pars["gpp_c_tan"] = round(gpp_c_tan, N_ROUND)
        model_pars["gpp_fast"] = round(gpp_fast, N_ROUND)

        NPAR = 8
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of GPP
            scc_gpp = general_calibration_fun(
                esm_data[model][realisation][experiment]["gpp0"],
                gpp_t_l,
                gpp_t_e,
                gpp_c_l,
                gpp_c_half,
                gpp_c_e,
                gpp_hyst,
                gpp_c_tan,
                gpp_fast,
                esm_data[model][realisation][experiment]["cveg"]
                - esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["catm"][0],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            )

            print("Making GPP plot data for:", model)

            conlabel = ["gpp_t_l", "gpp_t_e", "gpp_c_l", "gpp_c_half", "gpp_c_e", "gpp_hyst", "gpp_c_tan", "gpp_fast"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)
            par5 = np.linspace(parlo[4], parhi[4], num=N4CON, endpoint=True)
            par6 = np.linspace(parlo[5], parhi[5], num=N4CON, endpoint=True)
            par7 = np.linspace(parlo[6], parhi[6], num=N4CON, endpoint=True)
            par8 = np.linspace(parlo[7], parhi[7], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][realisation][experiment]["gpp0"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["cveg"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["gpp"],
                "gpp",
            ]

            # Contour par1 and par2 (vres_t_l, vres_t_e)
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

            # Contour par1 and par3 (vres_t_l, vres_c_l)
            jj = [2, 4]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par5,
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

            # Contour par4 and par5 (vres_c_e, vres_hyst)
            jj = [3, 5]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par4,
                par2=par6,
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

            jj = [6, 7]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par7,
                par2=par8,
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
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][realisation][experiment]["gpp"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                # esm_flux = esm_flux / esm_data[model][realisation][experiment]["cveg"]
                esm_flux = esm_flux / esm_data[model][realisation][experiment]["cveg"]
                scc_gpp = scc_gpp / esm_data[model][realisation][experiment]["cveg"]
            else:
                # scc_gpp = scc_gpp * esm_data[model][realisation][experiment]["cveg"]
                pass

            fit_gpp = Data(
                time=esm_data[model][realisation][experiment]["time"],
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
        print("\n\t Bayesian Litter Calibration")
        
        # Run Bayesian calibration
        systime_litbe = systime.time()
        samples_dim, sampler, lit_normalizer = run_bayesian_calibration_litter(
            esm_data, model, lit_range, initial_lit_guess_dim,
            nwalkers=32, nsteps=50000, burn_in=20000
        )
        systime_litend = systime.time()
        
        # Analyze results
        param_names = ["lit_t_l", "lit_t_e", "lit_c_l", "lit_c_half", 
                      "lit_c_e", "lit_hyst", "lit_c_tan", "lit_fast"]
        
        results = analyze_bayesian_results(
            samples_dim, param_names, OUT_DIR, model, "litter"
        )

        tau = sampler.get_autocorr_time(tol=0)
        print("tau is:", tau) # you want chain length > 50xtau. If nsteps is less, results will look noisy

        print(np.mean(sampler.acceptance_fraction)) # ideally 
        
        # Save results
        save_bayesian_parameters(results, model, OUT_DIR)
        
        # For compatibility with your existing plotting code, you can use the median values
        median_params = [results[param]["median"] for param in param_names]
        lit_t_l, lit_t_e, lit_c_l, lit_c_half, lit_c_e, lit_hyst, lit_c_tan, lit_fast = pstar = median_params
        p_bar = lit_normalizer.normalise(pstar)
        
        print(f"\nMedian parameter values:")
        for param, value in zip(param_names, median_params):
            print(f"{param} = {value:.6f}")


        print("")
        print("\t ========================== LIT SOLUTIONS ==========================")
        print("\t lit_t_l=", lit_t_l, " lit_t_e=", lit_t_e)
        print("\t lit_c_l=", lit_c_l, " lit_c_half=", lit_c_half, " lit_c_e=", lit_c_e)
        print("\t lit_hyst=", lit_hyst, " lit_c_tan=", lit_c_tan, "lit_fast=", lit_fast)
        print("\t Time to calculate optimisation: ", systime_litend - systime_litbe)
        print("\t ========================== LIT SOLUTIONS ==========================")
        print("")

        # Add solution values to the output dictionary
        model_pars["lit_t_l"] = round(lit_t_l, N_ROUND)
        model_pars["lit_t_e"] = round(lit_t_e, N_ROUND)
        model_pars["lit_c_l"] = round(lit_c_l, N_ROUND)
        model_pars["lit_c_half"] = round(lit_c_half, N_ROUND)
        model_pars["lit_c_e"] = round(lit_c_e, N_ROUND)
        model_pars["lit_hyst"] = round(lit_hyst, N_ROUND)
        model_pars["lit_c_tan"] = round(lit_c_tan, N_ROUND)
        model_pars["lit_fast"] = round(lit_fast, N_ROUND)

        NPAR = 8
        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of litterfall
            scc_lit = general_calibration_fun(
                esm_data[model][realisation][experiment]["lit0"]
                / esm_data[model][realisation][experiment]["cveg"][0],
                lit_t_l,
                lit_t_e,
                lit_c_l,
                lit_c_half,
                lit_c_e,
                lit_hyst,
                lit_c_tan,
                lit_fast,
                esm_data[model][realisation][experiment]["cveg"]
                - esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["catm"][0],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            )

            print(scc_lit[0] * esm_data[model][realisation][experiment]["cveg"][0], esm_data[model][realisation][experiment]["cveg"][0])

            conlabel = ["lit_t_l", "lit_t_e", "lit_c_l", "lit_c_half", "lit_c_e", "lit_hyst", "lit_c_tan", "lit_fast"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)
            par5 = np.linspace(parlo[4], parhi[4], num=N4CON, endpoint=True)
            par6 = np.linspace(parlo[5], parhi[5], num=N4CON, endpoint=True)
            par7 = np.linspace(parlo[6], parhi[6], num=N4CON, endpoint=True)
            par8 = np.linspace(parlo[7], parhi[7], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][realisation][experiment]["lit0"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["cveg"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["lit"],
                "lit",
            ]

            # Contour par1 and par2 (vres_t_l, vres_t_e)
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

            # Contour par1 and par3 (vres_t_l, vres_c_l)
            jj = [2, 4]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par5,
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

            # Contour par4 and par5 (vres_c_e, vres_hyst)
            jj = [3, 5]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par4,
                par2=par6,
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

            jj = [6, 7]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par7,
                par2=par8,
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
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][realisation][experiment]["lit"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                esm_flux = esm_flux / esm_data[model][realisation][experiment]["cveg"]
            else:
                scc_lit = scc_lit * esm_data[model][realisation][experiment]["cveg"]

            fit_litter = Data(
                time=esm_data[model][realisation][experiment]["time"],
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

        # number of parameters to calibrate: (vres_t_l, vres_t_e , vres_c_l, vres_c_half)
        NPAR = 8

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
        vres_t_l, vres_t_e, vres_c_l, vres_c_half, vres_c_e, vres_hyst, vres_c_tan, vres_fast = pstar = vres_normalizer.inv(p_bar)

        print("")
        print("\t ================= VEGETATION RESPIRATION SOLUTIONS ================")
        print("\t vres_t_l=", vres_t_l, " vres_t_e=", vres_t_e)
        print("\t vres_c_l=", vres_c_l, " vres_c_half=", vres_c_half, "vres_c_e=", vres_c_e)
        print("\t vres_hyst=", vres_hyst, " vres_c_tan=", vres_c_tan, "vres_fast=", vres_fast)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t ================= VEGETATION RESPIRATION SOLUTIONS ================")
        print("")

        # Add solution values to the output dictionary
        model_pars["vres_t_l"] = round(vres_t_l, N_ROUND)
        model_pars["vres_t_e"] = round(vres_t_e, N_ROUND)
        model_pars["vres_c_l"] = round(vres_c_l, N_ROUND)
        model_pars["vres_c_half"] = round(vres_c_half, N_ROUND)
        model_pars["vres_c_e"] = round(vres_c_e, N_ROUND)
        model_pars["vres_hyst"] = round(vres_hyst, N_ROUND)
        model_pars["vres_c_tan"] = round(vres_c_tan, N_ROUND)
        model_pars["vres_fast"] = round(vres_fast, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of vegetation resp.
            scc_vres = general_calibration_fun(
                esm_data[model][realisation][experiment]["ra0"]
                / esm_data[model][realisation][experiment]["cveg"][0],
                vres_t_l,
                vres_t_e,
                vres_c_l,
                vres_c_half,
                vres_c_e,
                vres_hyst,
                vres_c_tan,
                vres_fast,
                esm_data[model][realisation][experiment]["cveg"]
                - esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["catm"][0],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            )

            print("Making vegetation respiration plot data for:", model)

            conlabel = ["vres_t_l", "vres_t_e", "vres_c_l", "vres_c_half", "vres_c_e", "vres_hyst", "vres_c_tan", "vres_fast"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)
            par5 = np.linspace(parlo[4], parhi[4], num=N4CON, endpoint=True)
            par6 = np.linspace(parlo[5], parhi[5], num=N4CON, endpoint=True)
            par7 = np.linspace(parlo[6], parhi[6], num=N4CON, endpoint=True)
            par8 = np.linspace(parlo[7], parhi[7], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][realisation][experiment]["ra0"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["cveg"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["ra"],
                "vres",
            ]

            # Contour par1 and par2 (vres_t_l, vres_t_e)
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

            # Contour par1 and par3 (vres_t_l, vres_c_l)
            jj = [2, 4]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par5,
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

            # Contour par4 and par5 (vres_c_e, vres_hyst)
            jj = [3, 5]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par4,
                par2=par6,
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

            jj = [6, 7]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par7,
                par2=par8,
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
            FLUX_NAME = "vres"
            xlist = [
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][realisation][experiment]["ra"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                esm_flux = esm_flux / esm_data[model][realisation][experiment]["cveg"]
            else:
                scc_vres = scc_vres * esm_data[model][realisation][experiment]["cveg"]

            fit_vres = Data(
                time=esm_data[model][realisation][experiment]["time"],
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

        # number of parameters to calibrate: (sres_t_l, sres_t_e , sres_c_l, sres_c_half)
        NPAR = 8

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
        sres_t_l, sres_t_e, sres_c_l, sres_c_half, sres_c_e, sres_hyst, sres_c_tan, sres_fast = pstar = sres_normalizer.inv(p_bar)

        print("")
        print("\t =================    SOIL RESPIRATION SOLUTIONS    ================")
        print("\t sres_t_l=", sres_t_l, " sres_t_e=", sres_t_e)
        print("\t sres_c_l=", sres_c_l, " sres_c_half=", sres_c_half, "sres_c_e =", sres_c_e)
        print("\t sres_hyst=", sres_hyst, " sres_c_tan=", sres_c_tan, " sres_fast=", sres_fast)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t =================    SOIL RESPIRATION SOLUTIONS    ================")
        print("")

        # Add solution values to the output dictionary
        model_pars["sres_t_l"] = round(sres_t_l, N_ROUND)
        model_pars["sres_t_e"] = round(sres_t_e, N_ROUND)
        model_pars["sres_c_l"] = round(sres_c_l, N_ROUND)
        model_pars["sres_c_half"] = round(sres_c_half, N_ROUND)
        model_pars["sres_c_e"] = round(sres_c_e, N_ROUND)
        model_pars["sres_hyst"] = round(sres_hyst, N_ROUND)
        model_pars["sres_c_tan"] = round(sres_c_tan, N_ROUND)
        model_pars["sres_fast"] = round(sres_fast, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of soil resp.
            scc_sres = general_calibration_fun(
                esm_data[model][realisation][experiment]["rh0"]
                / esm_data[model][realisation][experiment]["csoil"][0],
                sres_t_l,
                sres_t_e,
                sres_c_l,
                sres_c_half,
                sres_c_e,
                sres_hyst,
                sres_c_tan,
                sres_fast,
                esm_data[model][realisation][experiment]["csoil"]
                - esm_data[model][realisation][experiment]["csoil"][0],
                esm_data[model][realisation][experiment]["csoil"][0],
                esm_data[model][realisation][experiment]["catm"][0],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            )

            print("Making soil respiration plot data for:", model)

            conlabel = ["sres_t_l", "sres_t_e", "sres_c_l", "sres_c_half", "sres_c_e", "sres_hyst", "sres_c_tan", "sres_fast"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            # (sres_t_l, sres_t_e, sres_c_l, sres_c_half)
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)
            par5 = np.linspace(parlo[4], parhi[4], num=N4CON, endpoint=True)
            par6 = np.linspace(parlo[5], parhi[5], num=N4CON, endpoint=True)
            par7 = np.linspace(parlo[6], parhi[6], num=N4CON, endpoint=True)
            par8 = np.linspace(parlo[7], parhi[7], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][realisation][experiment]["rh0"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["csoil"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["rh"],
                "sres",
            ]

            # Contour par1 and par2 (vres_t_l, vres_t_e)
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

            # Contour par1 and par3 (vres_t_l, vres_c_l)
            jj = [2, 4]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par5,
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

            # Contour par4 and par5 (vres_c_e, vres_hyst)
            jj = [3, 5]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par4,
                par2=par6,
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

            jj = [6, 7]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par7,
                par2=par8,
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
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][realisation][experiment]["rh"]

            # If the RETCOEF flag is True, divide by the relevant stock to obtain
            # the efficiency parameter exclusively
            if RETCOEF:
                esm_flux = esm_flux / esm_data[model][realisation][experiment]["csoil"]
            else:
                scc_sres = scc_sres * esm_data[model][realisation][experiment]["csoil"]

            fit_sres = Data(
                time=esm_data[model][realisation][experiment]["time"],
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

        # number of parameters to calibrate: (npp_t_l, npp_t_e , npp_c_l, npp_c_half)
        NPAR = 8

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
        npp_t_l, npp_t_e, npp_c_l, npp_c_half, npp_c_e, npp_hyst, npp_c_tan, npp_fast = pstar = npp_normalizer.inv(p_bar)

        print("")
        print("\t =================        NPP SOLUTIONS          ================")
        print("\t npp_t_l=", npp_t_l, " npp_t_e=", npp_t_e)
        print("\t npp_c_l=", npp_c_l, " npp_c_half=", npp_c_half, "npp_c_e=", npp_c_e)
        print("\t npp_hyst=", npp_hyst, " npp_c_tan=", npp_c_tan, "npp_fast=", npp_fast)
        print("\t Time to calculate optimisation: ", systime_gppend - systime_gppbe)
        print("\t =================        NPP SOLUTIONS          ================")
        print("")

        # Add solution values to the output dictionary
        model_pars["npp_t_l"] = round(npp_t_l, N_ROUND)
        model_pars["npp_t_e"] = round(npp_t_e, N_ROUND)
        model_pars["npp_c_l"] = round(npp_c_l, N_ROUND)
        model_pars["npp_c_half"] = round(npp_c_half, N_ROUND)
        model_pars["npp_c_e"] = round(npp_c_e, N_ROUND)
        model_pars["npp_hyst"] = round(npp_hyst, N_ROUND)
        model_pars["npp_c_tan"] = round(npp_c_tan, N_ROUND)
        model_pars["npp_fast"] = round(npp_fast, N_ROUND)

        for experiment in experiment_list:
            # Call same function that is minimized to get SCC prediction of NPP
            scc_npp = general_calibration_fun(
                esm_data[model][realisation][experiment]["npp0"],
                npp_t_l,
                npp_t_e,
                npp_c_l,
                npp_c_half,
                npp_c_e,
                npp_hyst,
                npp_c_tan,
                npp_fast,
                esm_data[model][realisation][experiment]["cveg"]
                - esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["cveg"][0],
                esm_data[model][realisation][experiment]["catm"][0],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            )

            print("Making npp plot data for:", model)

            conlabel = ["npp_t_l", "npp_t_e", "npp_c_l", "npp_c_half", "npp_c_e", "npp_hyst", "npp_c_tan", "npp_fast"]
            N4CON = 50  # number of elements for contour plots in each axis

            # Make contour plots for all four variables
            # (npp_t_l, npp_t_e, npp_c_l, npp_c_half, npp_c_e, npp_hyst)
            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)
            par5 = np.linspace(parlo[4], parhi[4], num=N4CON, endpoint=True)
            par6 = np.linspace(parlo[5], parhi[5], num=N4CON, endpoint=True)
            par7 = np.linspace(parlo[6], parhi[6], num=N4CON, endpoint=True)
            par8 = np.linspace(parlo[7], parhi[7], num=N4CON, endpoint=True)

            # Lists to store data for plotting
            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            # Constants we are going to be using for evaluating costs
            esm_vals = [
                esm_data[model][realisation][experiment]["npp0"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["cveg"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["npp"],
                "npp",
            ]

            # Contour par1 and par2 (vres_t_l, vres_t_e)
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

            # Contour par1 and par3 (vres_t_l, vres_c_l)
            jj = [2, 4]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par5,
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

            # Contour par4 and par5 (vres_c_e, vres_hyst)
            jj = [3, 5]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par4,
                par2=par6,
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

            jj = [6, 7]
            par1dim, par2dim, costs = prepare_contour_data(
                par1=par7,
                par2=par8,
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
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]
            esm_flux = esm_data[model][realisation][experiment]["npp"]

            if RETCOEF:
                # esm_flux = esm_flux / esm_data[model][realisation][experiment]["cveg"]
                esm_flux = esm_flux / esm_data[model][realisation][experiment]["cveg"]
                scc_npp = scc_npp / esm_data[model][realisation][experiment]["cveg"]
            else:
                # scc_npp = scc_npp * esm_data[model][realisation][experiment]["cveg"]
                pass

            fit_npp = Data(
                time=esm_data[model][realisation][experiment]["time"],
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

        parlo = np.zeros(NPAR)
        parhi = np.ones(NPAR)
        for i in range(NPAR):
            if docn_normalizer.pranges[i, 0] == docn_normalizer.pranges[i, 1]:
                parhi[i] = 0.0

        calibration_args = (docn_normalizer, esm_data, model, DTIME_0, realisation)

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

        num_t_eps = round(
            1
            + (
                esm_data[model][realisation][experiment]["time"][-1]
                - esm_data[model][realisation][experiment]["time"][0]
            )
            / DTIME_0
        )
        for experiment in experiment_list:
            scc_oflux = docn_func(
                esm_data[model][realisation][experiment]["catm0"],
                DTIME_0,
                num_t_eps,
                esm_data[model][realisation][experiment]["time"][0],
                docn,
                docnfac,
                ocntemp,
                docntemp,
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["dtocn"],
            )

            print("Making OFLUX plot data for:", model)
            conlabel = ["docn", "docnfac", "ocntemp", "docntemp"]

            N4CON = 15  # slow, so don't do too many

            par1arr = []
            par2arr = []
            costarr = []
            jarr = []

            par1 = np.linspace(parlo[0], parhi[0], num=N4CON, endpoint=True)
            par2 = np.linspace(parlo[1], parhi[1], num=N4CON, endpoint=True)
            par3 = np.linspace(parlo[2], parhi[2], num=N4CON, endpoint=True)
            par4 = np.linspace(parlo[3], parhi[3], num=N4CON, endpoint=True)

            # Contour par1 and par2
            jj = [0, 1]

            gcm_values = [
                esm_data[model][realisation][experiment]["catm0"],
                DTIME_0,
                num_t_eps,
                esm_data[model][realisation][experiment]["time"][0],
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["oflux"],
                esm_data[model][realisation][experiment]["dtocn"],
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

            gcm_values = [
                esm_data[model][realisation][experiment]["catm0"],
                DTIME_0,
                num_t_eps,
                esm_data[model][realisation][experiment]["time"][0],
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["catm"],
                esm_data[model][realisation][experiment]["oflux"],
                esm_data[model][realisation][experiment]["dtocn"],
            ]

            par1dim, par2dim, costs = prepare_contour_data(
                par1=par3,
                par2=par4,
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
                esm_data[model][realisation][experiment]["time"],
                esm_data[model][realisation][experiment]["dtglb"],
                esm_data[model][realisation][experiment]["catm"],
            ]
            xlabel = ["Year", r"$\Delta T$", "c_atm"]
            title = [model + ", " + experiment, FLUX_NAME]

            gcm = esm_data[model][realisation][experiment]["oflux"]

            fit_oflux = Data(
                time=esm_data[model][realisation][experiment]["time"],
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
            esm_data[model][realisation][experiment],
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
