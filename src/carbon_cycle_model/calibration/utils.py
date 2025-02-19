"""
File to store any auxiliary functions for the calibration code.
"""

import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optimparallel import minimize_parallel

from carbon_cycle_model.utils import load_esm_data
from carbon_cycle_model.land_component.boxes.utils import general_calibration_fun
from carbon_cycle_model.utils import butterworth
from carbon_cycle_model.ocean_component.ocean_component import OceanCarbonCycle
from carbon_cycle_model.ocean_component.utils import joos_response


class Normalizer:
    """
    A class for normalization of parameters so they lie in the range [0,1].

    Initialisation just requires array with the desired ranges for all the parameters for
    a given variable. E.g.:
        gpp_range     = np.array([[-3, 5], [-3, 5], [-6, 6], [1, 1000]])
        gppNormalizer = utils.Normalizer(prange=gpp_range)

    The "normalise" method converts input dimensional x into normalized values.
    The invert method returns dimensional quantities, given an array with normalised
    values.
    """

    def __init__(self, pranges=None):
        if pranges.ndim == 1:
            pranges = np.expand_dims(pranges, axis=0)
        self.pranges = pranges
        # Count true number of parameters: their ranges must be non-zero
        npar = 0
        for prange in pranges:
            if prange.min() != prange.max():
                npar = npar + 1
        self.npar = npar

    def normalise(self, x=None):
        "Convert dimensional input (x) into the corresponding normalised values."
        ans = x.copy()
        for i, (lo, hi) in enumerate(self.pranges):
            if hi == lo:
                ans[i] = 0.5
            else:
                ans[i] = (x[i] - lo) / (hi - lo)
        return ans

    def inv(self, x=None):
        "Convert normalised input (x) into the corresponding dimensional values."
        ans = x.copy()
        for i, (lo, hi) in enumerate(self.pranges):
            ans[i] = lo + x[i] * (hi - lo)
        return ans


def load_and_prepare_esm_data(
    path, model, experiment, recalc_emis, ninit, smoothing_alg
):
    """
    Load and prepare the ESM data for calibration.

    It calls the load_esm_data on the relevant ESM input file (as specified by the
    combination of path, model and experiment) and derives the associated pre-industrial
    quantities.

    Input:
    - path: filepath to the folder where ESM input data files are saved.
    - model: ESM to run the calibration against.
    - experiment: scenario to run calibration against.
    - recalc_emis: whether to recalculate emissions based on carbon balance.
    - ninit: number of initial datapoints (years) to use in the derivation of the
             pre-industrial quantities.
    - smoothing_alg: smoothing algorithm to use when loading the ESM data.

    return: dictionary with loaded ESM data, under the appropriate model and
            scenario entries.
    """
    file_path = path + model + "_" + experiment + ".txt"
    print("Loading file: ", file_path)

    esm_input = load_esm_data(
        file_path, recalc_emis, ninit=ninit, smoothing_pars=smoothing_alg
    )

    # Get initial figures for the relevant quantities for the model
    # Stocks are directly the first ESM data points, whereas we take the average
    # of the first ninit years for fluxes, to try and denoise the signal.
    cveg0 = esm_input.cveg[0]
    csoil0 = esm_input.csoil[0]

    gpp0 = np.mean(esm_input.gpp[:ninit])
    npp0 = np.mean(esm_input.npp[:ninit])
    lit0 = np.mean(esm_input.lit[:ninit])

    catm0 = np.mean(esm_input.catm[:ninit])
    lu0 = np.mean(esm_input.lu[:ninit])

    sres0 = np.mean(esm_input.rh[:ninit])
    vres0 = np.mean(esm_input.ra[:ninit])

    # Flux efficiency parameters
    vres0_par = vres0 / cveg0
    sres0_par = sres0 / csoil0
    lit0_par = lit0 / cveg0

    loaded_data = {
        "catm": esm_input.catm,
        "csoil": esm_input.csoil,
        "cveg": esm_input.cveg,
        "dtglb": esm_input.dtglb,
        "dtocn": esm_input.dtocn,
        "emis": esm_input.emis,
        "gpp": esm_input.gpp,
        "lit": esm_input.lit,
        "lu": esm_input.lu,
        "npp": esm_input.npp,
        "oflux": esm_input.oflux,
        "ra": esm_input.ra,
        "rh": esm_input.rh,
        "time": esm_input.time,
        "fcvegout": esm_input.fcvegout,
        "fcsoilout": esm_input.fcsoilout,
        "gpp0": gpp0,
        "npp0": npp0,
        "cveg0": cveg0,
        "csoil0": csoil0,
        "lit0": lit0,
        "catm0": catm0,
        "lu0": lu0,
        "rh0": sres0,
        "ra0": vres0,
        "vres0": vres0_par,
        "sres0": sres0_par,
        "litp0": lit0_par,
    }
    return {model: {experiment: loaded_data}}


def get_vars_from_experiments(esm_data, model, x):
    """
    Get all instances of a given variable and model from a dictionary.

    input:
    - esm_data: dictionary storing the esm data.
    - model: model to extract the variable data for.
    - x: name of the variable to extract.

    return: array with all instances of the variable "x" in "esm_data".
    """
    x_list = []
    for experiment in esm_data[model].keys():
        x_list.append(esm_data[model][experiment][x])

    return x_list


def get_general_average(esm_data, model, var_name, nround):
    """
    Get the average of a given variable across different experiments for a model.

    input:
    - esm_data: dictionary storing the esm data.
    - model: model to extract the variable data for.
    - var_name: name of the variable to extract.
    - nround: number of decimal points to use in rounding.

    return: average of the variable across the different experiments.
    """
    return round(
        np.average(
            get_vars_from_experiments(
                esm_data,
                model,
                var_name,
            )
        ),
        nround,
    )


def run_minimisation(func, p0, inargs, xlo, xhi, ftol=1e-6, attempts=5):
    """
    Run the cost minimisation routine 'attempts' times for the 'func' function
    with different starting points.

    input:
    - func: function to minimise, returning a measure of cost.
    - p0: inititial guess for the parameter values.
    - inargs: input arguments for the function
    - xlo: low bounds for the function
    - xhi: high bounds for the function
    - ftol: function tolerance for variantions to find the minimum cost.
    - attempts: number of times to attempt the cost minimisation.

    return:
    - p_bar: array with parameter values that minimise the function cost.
    - costmin: associated minimum cost for p_bar.
    - results_vec: list of cost for each attempt.
    """
    maxiter = 100000 * len(p0)
    # Get bounds sequence
    if len(xlo) != len(xhi):
        raise AssertionError(
            f"Array of lower bounds is not the same length as array of higher bounds: "
            f"{len(xlo)} - {len(xhi)}"
        )
    bounds = []
    for idx, xlo_instance in enumerate(xlo):
        bounds.append((xlo_instance, xhi[idx]))

    outmin = minimize_parallel(
        func,
        p0,
        args=inargs,
        bounds=bounds,
        tol=ftol,
        options={"maxiter": maxiter, "maxfun": maxiter},
    )
    #   parallel={'max_workers': 1})

    p_bar = outmin.x
    costmin = outmin.fun

    print(
        "\n> Iteration=",
        1,
        ", costmin=",
        costmin,
        "termination reason: ",
        outmin.message,
    )
    print("  p_bar=", p_bar)

    # p_bar is the normalized solution for parameters between 0 and 1.

    # After first search above, do some more searches to see perhaps if we are in a local
    # minimum. Brute-force approach: create attempts random start points and use these
    # as initial solutions. Pick the final solution from the different start points with
    # lowest minimum cost.

    # Array to store costs
    results_vec = np.zeros(attempts)
    if attempts > 1:
        results_vec[0] = costmin

        # Create a matrix of random starting point, where each row is a calibration
        # attempt and each column is a variable
        nvar = p0.shape[0]
        nran = attempts * nvar
        ran_starting_points = np.reshape(np.random.uniform(size=nran), (attempts, nvar))

        # Run for each starting point
        for ii in range(attempts - 1):
            p0 = ran_starting_points[ii, :]
            outmin = minimize_parallel(
                func,
                p0,
                args=inargs,
                bounds=bounds,
                tol=ftol,
                options={"maxiter": maxiter, "maxfun": maxiter},
            )
            #   parallel={'max_workers': 1})

            p_out = outmin.x
            costout = outmin.fun
            results_vec[ii + 1] = costout

            # Only keep the new par values if cost is lower
            if costout < costmin:
                p_bar = p_out
                costmin = costout
            elif np.isnan(costmin):
                p_bar = p_out
                costmin = costout
            print(
                "\n> Iteration=", ii + 2, ", costmin=", costmin, ", costout=", costout
            )
            print("  p_bar=", p_bar)

    print(f"Final p_bar = {p_bar}")
    print(f"Final costmin = {costmin}")

    return p_bar, costmin, results_vec


def hi_freq_variance(x, cutoff=30):
    """
    Obtain variance from the high-frequency component of a data array.

    Pass the data by a butterworth filter to obtain the low-frequency signal,
    subtract that signal from the data to obtain the high-frequency component, and
    finally calculate the variance of that noise.

    input:
    - x: data to find the high-frequency variance for.
    - cutoff: cutoff period for the butterworth filter.

    return: variance of the high-frequency component of the signal.
    """
    padlen = min([cutoff, int(x.shape[0] / 2.0)])  # Padding
    high_f_comp = x - butterworth(x, cutoff, padlen=padlen)
    return np.var(high_f_comp, ddof=1)


def calculate_cost_gen_func_cross_experiment(
    param,
    normalizer,
    esm_data,
    model,
    flux0,
    stock,
    esm_flux,
):
    """
    Calculate the "cost" of the "gen_func" for the associated parameters and flux
    across all experiments for a given model.

    This "cost" is essentially a measure of how close our approximation by the gen_func
    is to the original esm flux timseries.

    input:
    - param: parameters to define the instance of "gen_func" to calculate the cost of.
    - normaliser: normaliser object to translate par values from dimenional to
                  normalised and vice versa.
    - esm_data: dictionary containing the esm data for the specific model.
    - model: model to use for input to the cost calculation. Essentially, the model
             we are calibrating against.
    - flux0: pre-industrial value for the carbon flux we are applying the "gen_func"
             approximation to. (GtC/year)
    - stock: timeseries of the carbon stocks of the relevant pool for the experiment
            we are calibrating against. (GtC)
    - esm_flux: timeseries of the relevant carbon flux for the experiment we are
                calibrating against. (GtC/year)

    return: total cost across experiments.
    """
    cutoff = 30
    par_t_l, par_t_e, par_c_l, par_c_half = normalizer.inv(param)
    cost = 0
    for experiment in esm_data[model].keys():
        cost += cost_gen_func(
            esm_data[model][experiment][flux0],
            par_t_l,
            par_t_e,
            par_c_l,
            par_c_half,
            esm_data[model][experiment]["dtglb"],
            esm_data[model][experiment]["catm"],
            esm_data[model][experiment][stock],
            esm_data[model][experiment][esm_flux],
            cutoff=cutoff,
        )
    return np.log(cost)


def calculate_cost_gen_func(param, normalizer, flux0, catm, stock, dtglb, esm_flux):
    """
    Calculate the cost of the "gen_func" for the associated parameters and flux
    across all experiments for a given model.

    This "cost" is essentially a measure of how close our approximation by the gen_func
    is to the original esm flux timseries.

    input:
    - param: parameters to define the instance of "gen_func" to calculate the cost of.
    - normaliser: normaliser object to translate par values from dimenional to
                  normalised and vice versa.
    - flux0: pre-industrial value for the carbon flux we are applying the "gen_func"
             approximation to. (GtC/year)
    - catm: timeseries of the carbon concentration of the atmosphere for the experiment
            we are calibrating against. (ppm)
    - stock: timeseries of the carbon stocks of the relevant pool for the experiment
            we are calibrating against. (GtC)
    - dtglb: timeseries of the temperature anomaly for the experiment we are calibrating
            against. (K)
    - esm_flux: timeseries of the relevant carbon flux for the experiment we are
                calibrating against. (GtC/year)

    return: total cost for the experiment.
    """
    # param are normalized (range [0,1]), so use inv method to 'de-normalize'
    par_t_l, par_t_e, par_c_l, par_c_half = normalizer.inv(param)
    ans1 = cost_gen_func(
        flux0, par_t_l, par_t_e, par_c_l, par_c_half, dtglb, catm, stock, esm_flux
    )
    return np.log(ans1)


def cost_gen_func(
    flux0,
    par_t_l,
    par_t_e,
    par_c_l,
    par_c_half,
    dtglb,
    catm,
    stock,
    esm_flux,
    cutoff=30,
):
    """
    Calculate the cost of the "gen_func" for the associated parameters and flux
    across all experiments for a given model.

    This "cost" is essentially a measure of how close our approximation by the gen_func
    is to the original esm flux timseries.

    input:
    - flux0: pre-industrial value for the carbon flux we are applying the "gen_func"
             approximation to. (GtC/year)
    - par_t_l: parameter associated with the linear dependence on temperature of gen_func
    - par_t_e: parameter associated with the exponential dependence on temperature of
               gen_func
    - par_c_l: parameter associated with the linear dependence on carbon stocks of
               gen_func
    - par_c_half: parameter associated with the atmospheric carbon saturation component
                  of gen_func
    - dtglb: timeseries of the temperature anomaly for the experiment we are calibrating
            against. (K)
    - catm: timeseries of the carbon concentration of the atmosphere for the experiment
            we are calibrating against. (ppm)
    - stock: timeseries of the carbon stocks of the relevant pool for the experiment
            we are calibrating against. (GtC)
    - esm_flux: timeseries of the relevant carbon flux for the experiment we are
                calibrating against. (GtC/year)
    - cutoff: cutoff period for the butterworth filter.

    return: total cost for the experiment.
    """
    scc_val = general_calibration_fun(
        flux0,
        par_t_l,
        par_t_e,
        par_c_l,
        par_c_half,
        stock,
        stock[0],
        catm[0],
        dtglb,
        catm,
    )
    variance = hi_freq_variance(esm_flux, cutoff=cutoff)
    cost = np.sum((scc_val - esm_flux) ** 2)
    n = float(esm_flux.shape[0])
    ans = cost / (n * variance)
    return ans


# ============================
# OCEAN COST FUNCTIONS
# ============================


def calculate_cost_ocean_cross_experiment(param, normalizer, esm_data, model, dtime0):
    """
    Calculate the "cost" of the ocean emulation for the associated parameters and flux
    across all experiments for a given model.

    This "cost" is essentially a measure of how close our approximation by our ocean
    module is to the original esm flux timseries.

    input:
    - param: parameters to define the instance of "gen_func" to calculate the cost of.
    - normalizer: normaliser object to translate par values from dimensional to
                  normalised and vice versa.
    - esm_data: dictionary containing the esm data for the specific model. (GtC/year)
    - model: model to use for input to the cost calculation. Essentially, the model
             we are calibrating against.
    - dtime0: time step size for the ocean emulation. (year)

    return: total cost across experiments.
    """
    cutoff = 30
    docn, docnfac, ocntemp, docntemp = normalizer.inv(param)
    cost = 0
    for experiment in esm_data[model].keys():
        num_steps = round(
            1
            + (
                esm_data[model][experiment]["time"][-1]
                - esm_data[model][experiment]["time"][0]
            )
            / dtime0
        )
        cost += cost_docn(
            esm_data[model][experiment]["catm0"],
            dtime0,
            num_steps,
            docn,
            docnfac,
            ocntemp,
            docntemp,
            esm_data[model][experiment]["time"],
            esm_data[model][experiment]["catm"],
            esm_data[model][experiment]["oflux"],
            esm_data[model][experiment]["dtocn"],
            cutoff=cutoff,
        )
    return np.log(cost)


def costdocn1(
    param, normalizer, catm0, dtime0, num_steps, intime, catm, esm_oflux, dtocn
):
    """
    Calculate the "cost" of the ocean emulation for the associated parameters and flux
    for a given experiment and model.

    This "cost" is essentially a measure of how close our approximation by our ocean
    module is to the original esm flux timseries.

    input:
    - param: parameters to define the instance of "gen_func" to calculate the cost of.
    - normalizer: normaliser object to translate par values from dimensional to
                  normalised and vice versa.
    - dtime0: timestep size. (year)
    - num_steps: number of time steps for the ocean emulation.
    - intime: array with ESM time.
    - catm: array with atmospheric carbon concentrations. (ppm)
    - esm_flux: array with atmosphere-ocean carbon exchange values from the ESM.
                This is what we are trying to emulate. (GtC/year)
    - esm_data: dictionary containing the esm data for the specific model. (GtC/year)
    - dtocn: global mean near surface ocean temperature. (K)

    return: total cost across experiments.
    """
    cutoff = 30
    docn, docnfac, ocntemp, docntemp = normalizer.inv(param)
    ans1 = cost_docn(
        catm0,
        dtime0,
        num_steps,
        docn,
        docnfac,
        ocntemp,
        docntemp,
        intime,
        catm,
        esm_oflux,
        dtocn,
        cutoff=cutoff,
    )
    return np.log(ans1)


def cost_docn(
    catm0,
    dtime0,
    num_steps,
    docn,
    docnfac,
    ocntemp,
    docntemp,
    intime,
    catm,
    esm_oflux,
    dtocn,
    cutoff=30,
):
    """
    Calculate the "cost" of the ocean emulation for the associated parameters and flux
    for a given experiment and model.

    This "cost" is essentially a measure of how close our approximation by our ocean
    module is to the original esm flux timseries.

    input:
    - catm0: pre-industrial atmospheric carbon concentration.
    - dtime0: timestep size.
    - num_steps: number of time steps for the ocean emulation.
    - docn: initial mixing depth for CO2 uptake. Calibrated parameter. (m)
    - docnfac: temperature dependence for change in mixing depth for CO2 uptake.
               Calibrated parameter. (dimensionless).
    - ocntemp: temperature calibration parameter. (1/K).
    - intime: array with ESM time.
    - catm: array with atmospheric carbon concentrations. (ppm)
    - esm_flux: array with atmosphere-ocean carbon exchange values from the ESM.
                This is what we are trying to emulate. (GtC/year)
    - esm_data: dictionary containing the esm data for the specific model. (GtC/year)
    - dtocn: global mean near surface ocean temperature (K)
    - cutoff: cutoff period for the butterworth filter used to smooth calculate
              variance.

    return: total cost across experiments.
    """
    scc_oflux = docn_func(
        catm0,
        dtime0,
        num_steps,
        docn,
        docnfac,
        ocntemp,
        docntemp,
        intime,
        catm,
        dtocn,
    )
    variance = hi_freq_variance(esm_oflux, cutoff=cutoff)
    rss = np.sum((scc_oflux - esm_oflux) ** 2)
    n = float(esm_oflux.shape[0])
    ans = rss / (n * variance)
    return ans


def docn_func(
    catm0, dtime0, num_steps, docn, docnfac, ocntemp, docntemp, intime, catm, dtocn
):
    """
    Use the ocean scheme (Joos et al. 1996) to emulate the ocean carbon exchange with
    the atmosphere.

    input:
    - catm0: pre-industrial atmospheric carbon concentration.
    - dtime0: timestep size. (year)
    - num_steps: number of time steps for the ocean emulation.
    - docn: initial mixing depth for CO2 uptake. Calibrated parameter. (m).
    - docnfac: temperature dependence for change in mixing depth for CO2 uptake.
               Calibrated parameter. (dimensionless)
    - ocntemp: temperature calibration parameter. (1/K)
    - intime: array with ESM time.
    - catm: array with atmospheric carbon concentrations. (ppm)
    - dtocn: global mean near surface ocean temperature. (K)

    return: total cost across experiments.
    """
    pars = {
        "docn": docn,
        "docnfac": docnfac,
        "ocntemp": ocntemp,
        "docntemp": docntemp,
        "catm0": catm0,
    }
    oceancycle = OceanCarbonCycle(dtime0, num_steps, **pars)

    # Interpolate data to ocean model time points
    time_u = np.linspace(intime[0], intime[-1], num=num_steps, endpoint=True)
    catm_u = np.interp(time_u, intime, catm)
    dtocn_u = np.interp(time_u, intime, dtocn)

    ntime = time_u.shape[0]
    ocn_uptake = np.zeros(ntime, "f")
    n4occ = 1  # Ok since time_u is already made using dtime0
    rjoos = joos_response(time_u)
    deltacocn = np.zeros(ntime, "f")

    # Run the ocean carbon emulation
    for i in range(1, ntime):
        dco2_ocn, dummy = oceancycle.update(
            catm_u[i], catm_u[i - 1], i - 1, ocn_uptake, rjoos, dtocn_u[i], n4occ
        )
        deltacocn[i] = dco2_ocn

    uocn = deltacocn / (dtime0 / n4occ)

    # Interp back to input time to give output of same shape as input, and convert to Gt
    ans = np.interp(intime, time_u, uocn)
    return ans


def prepare_contour_data(
    par1, par2, cost_func, normaliser, normaliser_par_ind, nor_sol, esm_vals
):
    """
    Return a matrix with the costs associated to running cost_func for each pair of 'par1'
    and 'par2' values.

    input:
    - par1: one of the parmaters to vary in the contour plot.
    - par2: one of the parmaters to vary in the contour plot.
    - cost_func: cost function to calculate the cost associated with each pair of
                 parameters.
    - normaliser: normaliser object to translate par values from dimensional to
                  normalised and vice versa.
    - normaliser_par_ind: indeces of the parameters to vary in the normaliser object.
    - nor_sol: best fit solution coming from the calibration process (to superimpose on
               contour plot as a dot).
    - esm_vals: data from the ESM.

    otput:
    - par1dim: Array of values for the first parameter to vary (for the contour plot)
    - par2dim: Array of values for the second parameter to vary (for the contour plot)
    - cost_array: array storing the cost associated with each pair of parameter values.
    """
    # unpack values and prepare arrays
    par1dim = par1.copy()
    par2dim = par2.copy()
    cost_array = np.zeros((par1.shape[0], par2.shape[0]))

    # We want to substitute every par1 and par2 value in the array containing the
    # (dimensional) solutions for the parameters, and then obtain the cost, leaving all
    # other parameters in the soulution fixed. The posistions for those par1 and par2
    # values are passed through the normaliser_par_ind variable.
    for i, p1 in enumerate(par1):
        array1 = nor_sol.copy()
        array1[normaliser_par_ind[0]] = p1
        par1dim[i] = normaliser.inv(array1)[normaliser_par_ind[0]]
        for j, p2 in enumerate(par2):
            array2 = nor_sol.copy()
            array2[normaliser_par_ind[1]] = p2
            par2dim[j] = normaliser.inv(array2)[normaliser_par_ind[1]]
            param = nor_sol.copy()
            param[normaliser_par_ind[0]] = p1
            param[normaliser_par_ind[1]] = p2
            ans = cost_func(param, normaliser, *esm_vals)
            cost_array[i, j] = ans

    return par1dim, par2dim, cost_array


def plot_diagnostic(
    out_dir,
    esm_in,
    allfit,
    outplot=None,
    fontsize=11,
    xsize=16.0,
    ysize=22.0,
    hspace=0.35,
    wspace=0.35,
    top=0.94,
    bottom=0.10,
    right=0.97,
    left=0.14,
):
    """
    Function to plot diagnostic plots from the calibration process to pdf files.

    input:
    - out_dir: folder to save individual plots.
    - esm_in: object storing data from the ESM run.
    - allfit: object storing information related to the calibration of the model.
    - outplot: path and name of the file with output plots.
    - fontsize: fontsize to use in plots.
    (rest of arguments are related to output file dimensions)

    return: None
    """
    if outplot is None:
        outplot = os.path.join(os.getcwd(), "scc_calibration_plot.pdf")

    plt.close("all")
    # Defining some properties for the document to output
    plot_fontsize = fontsize
    matplotlib.rcParams.update({"font.size": plot_fontsize})
    pdfpages = PdfPages(outplot)
    lw = 1.0
    colfit1 = "red"
    colgcm = "k"
    alpha = 0.75
    lw = 0.8
    lwgcm = 0.8
    handlelength = 1.1
    borderaxespad = 0.35
    handletextpad = 0.35
    labelspacing = 0.25
    legsize = 0.8
    x_size_con = xsize
    y_size_con = xsize - 3.0
    figsize = (xsize / 2.54, ysize / 2.54)  # convert input size (cm) to inches
    consize = (x_size_con / 2.54, y_size_con / 2.54)
    legsize = legsize * fontsize
    legloc = "best"

    ##########
    ifig = 0
    nrow = 4
    ncol = 2
    alpha = 0.9

    plt.figure(ifig, figsize=figsize)
    plt.clf()
    plt.subplots_adjust(
        hspace=hspace, wspace=wspace, top=top, bottom=bottom, right=right, left=left
    )

    # ESM DATA
    # First make one page of subplots with 8 key input ESM variables
    title = ", ".join(allfit[0].title[0:2])
    lablist = ["catm", "Tglb", "cveg", "csoil", "gpp", "lit", "resp", "oflux"]
    try:
        esm_list = [
            esm_in.catm,
            esm_in.dtglb,
            esm_in.cveg,
            esm_in.csoil,
            esm_in.gpp,
            esm_in.lit,
            esm_in.rh,
            esm_in.oflux,
        ]
        for isub, gcmdata, ylab in zip(
            np.arange(1, nrow * ncol + 1, 1), esm_list, lablist
        ):
            ax = plt.subplot(nrow, ncol, isub)
            plt.xlabel("Year")
            plt.ylabel(ylab)
            plt.plot(esm_in.time, gcmdata, color=colgcm, linewidth=lw, alpha=alpha)
            if isub in [1, 2]:
                plt.title(title)
    except AttributeError:
        esm_list = [
            esm_in["catm"],
            esm_in["dtglb"],
            esm_in["cveg"],
            esm_in["csoil"],
            esm_in["gpp"],
            esm_in["lit"],
            esm_in["rh"],
            esm_in["oflux"],
        ]
        for isub, gcmdata, ylab in zip(
            np.arange(1, nrow * ncol + 1, 1), esm_list, lablist
        ):
            ax = plt.subplot(nrow, ncol, isub)
            plt.xlabel("Year")
            plt.ylabel(ylab)
            plt.plot(esm_in["time"], gcmdata, color=colgcm, linewidth=lw, alpha=alpha)
            if isub in [1, 2]:
                plt.title(title)

    pdfpages.savefig()

    # Loop over the various fluxes, showing first the ESM flux/variable, and then
    # the corresponding SCC values
    nrow = 3
    ncol = 2
    for fit in allfit:

        ifig = ifig + 1
        plt.figure(ifig, figsize=figsize)
        plt.clf()
        tit = ", ".join(fit.title)

        # DIFFERENCE PLOTS
        for ix, x1 in enumerate(fit.xlist):

            ax = plt.subplot(nrow, ncol, 2 * ix + 1)
            plt.subplots_adjust(
                hspace=hspace,
                wspace=wspace,
                top=top,
                bottom=bottom,
                right=right,
                left=left,
            )
            plt.xlabel(fit.xlabel[ix])
            plt.ylabel(fit.title[-1])
            plt.title(tit)
            c1 = "Fit"
            plt.plot(
                x1, fit.gcm, color=colgcm, linewidth=lwgcm, alpha=alpha, label="ESM"
            )
            plt.plot(x1, fit.scc, color=colfit1, linewidth=lw, alpha=alpha, label=c1)
            leg = plt.legend(
                loc=legloc,
                fontsize=legsize,
                handlelength=handlelength,
                borderaxespad=borderaxespad,
                handletextpad=handletextpad,
                labelspacing=labelspacing,
            )
            leg.draw_frame(False)

            ax = plt.subplot(nrow, ncol, 2 * ix + 2)
            plt.subplots_adjust(
                hspace=hspace,
                wspace=wspace,
                top=top,
                bottom=bottom,
                right=right,
                left=left,
            )
            plt.xlabel(fit.xlabel[ix])
            plt.ylabel(fit.title[-1] + ", ESM - FIT")
            plt.title(tit)
            plt.plot(
                x1,
                fit.gcm - fit.scc,
                color=colfit1,
                linewidth=lw,
                alpha=alpha,
                label="ESM-Fit",
            )
            plt.axhline(0.0, ls=":", color="k", lw=0.75)
            leg = plt.legend(
                loc=legloc,
                fontsize=legsize,
                handlelength=handlelength,
                borderaxespad=borderaxespad,
                handletextpad=handletextpad,
                labelspacing=labelspacing,
            )
            leg.draw_frame(False)

        pdfpages.savefig()

        # CONTOUR PLOTS
        for par1, par2, cost, jj in zip(
            fit.par1arr, fit.par2arr, fit.costarr, fit.jarr
        ):
            ifig = ifig + 1
            plt.figure(ifig, figsize=consize)
            plt.clf()
            cmap = plt.cm.jet
            origin = "lower"
            nlev = 31
            colone = "lightpink"
            ax = plt.subplot(1, 1, 1)
            # contourf wants 2D array with y coord first, so need to transpose
            # (strange python choice).
            cp = plt.contourf(
                par1, par2, cost.T, nlev, alpha=1.0, cmap=cmap, origin=origin
            )
            _ = plt.colorbar(cp)
            j1 = jj[0]
            j2 = jj[1]
            plt.plot(
                fit.pstar[j1], fit.pstar[j2], marker="o", markersize=8, color=colone
            )
            ccc = "  " + fit.title[0].split(", ")[0]
            plt.text(
                fit.pstar[j1],
                fit.pstar[j2],
                ccc,
                ha="left",
                va="bottom",
                fontsize=8,
                color=colone,
            )
            plt.xlabel(fit.conlabel[j1])
            plt.ylabel(fit.conlabel[j2])
            tit1 = tit + ", ln(MSE/SIGMA)"
            tit2 = f"({','.join(f'{par:9.4f}' for par in fit.pstar)})"
            # tit2 = "(" + "".join(["%9.4f" % par + "," for par in fit.pstar])[:-1] + ")"
            ax.set_title(tit1 + "\n " + tit2)
            pdfpages.savefig()

    ##########

    pdfpages.close()
    plt.clf()
    plt.close("all")
    print("Close outplot=", outplot)

    # Also print an individual plot with the ESM-SCM comparison
    matplotlib.rcParams.update({"font.size": 22})
    for fit in allfit:

        ifig = ifig + 1
        plt.figure(ifig, figsize=figsize)
        plt.clf()
        tit = ", ".join(fit.title)

        # DIFFERENCE PLOTS
        plt.xlabel(fit.xlabel[0])
        # plt.ylabel(fit.title[-1])
        plt.title(tit)
        c1 = "Fit"
        plt.plot(
            fit.xlist[0],
            fit.gcm,
            color=colgcm,
            linewidth=lwgcm,
            alpha=alpha,
            label="ESM",
        )
        plt.plot(
            fit.xlist[0], fit.scc, color=colfit1, linewidth=lw, alpha=alpha, label=c1
        )
        leg = plt.legend(
            loc=legloc,
            fontsize=3 * legsize,
            handlelength=handlelength,
            borderaxespad=borderaxespad,
            handletextpad=handletextpad,
            labelspacing=labelspacing,
        )
        leg.draw_frame(False)

        # Remove $ at both sides and \
        # title = re.sub(r'\$(.*?)\$', r'\1', fit.title[-1]).replace('\\', '')
        title = fit.title[-1].replace("$", "").replace("\\", "")
        plt.savefig(
            f"{out_dir}/ALL/individual_flux_{fit.title[0].replace(', ', '_')}_{title}.png"
        )
