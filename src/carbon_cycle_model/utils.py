"""
Script gathering miscelaneous utilities used in the model.
"""

import numpy as np
import scipy
from scipy.signal import savgol_filter
import os

from carbon_cycle_model.constants import KELVIN_0, OCEAN_FRAC, PPM2GT


class Data:
    """
    Container class used to store GCM/ESM data and other objects.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def keys(self):
        "Return name of the stored objects in the container."
        return list(self.__dict__.keys())

    def __str__(self):
        return repr(self)

    def __repr__(self):
        s = str(self.__class__).split(self.__module__)[1]
        if s[0] == ".":
            s = s[1:]
        if s[-1] == ">":
            s = s[:-1]
        if s[-1] == "'":
            s = s[:-1]
        attr = [a for a in dir(self) if a[0] != "_"]
        ans = s + ", public attributes: " + str(attr)
        return ans


def calc_emissions(
    time,
    catm,
    cveg,
    csoil,
    oflux,
    smoothing_pars={"type": "butterworth", "pars": [5]},
):
    """
    This estimates the total emissions needed to conserve CO2 given land and ocean
    stores. Can be noisy, so smoothing is recommended.

    To estimate, we calculate the derivative of the difference in C02 quantities
    (summed up across atmosphere, land and ocean) at each timepoint.
    """
    # This is a cumulative integral calculating the total historical carbon uptake by the
    # ocean at each timestep
    cocn = scipy.integrate.cumtrapz(oflux, x=time, initial=0.0)
    ctot = catm * PPM2GT + cveg + csoil + cocn

    # Implied emissions are the derivative of the difference.
    # Calculate using np.gradient (centre-differencing, except at endpoints).
    deriv = np.gradient(ctot, time)

    # Apply smoothing, cutoff=1 implies no smoothing.
    ffi = apply_smoothing(deriv, smoothing_pars)

    return ffi


# =============================================
#          BUTTERWORTH SMOOTHING
# =============================================


def smooth(x, cutoff=30, padlen=None):
    "Entry point for butterworth smoothing."
    if padlen is None:
        padlen = min([cutoff, int(x.shape[0] / 2)])
    return butterworth(x, cutoff, padlen=padlen)


def predict_linear_fit(y, new_x):
    "Predict values of y in the new range of X values with a linear fit"
    if y.ndim != 1:
        raise AssertionError(f"Y is {y.ndim} but should be a vector")
    ny = y.shape[0]
    x = np.array([np.ones(ny), np.arange(ny, dtype=float)])
    betas = np.linalg.lstsq(x.T, y, rcond=None)[0]
    return betas[0] + betas[1] * new_x


def use_linear_trend_to_extend_ends_1d(y, padlen):
    "Add padding and call predict_linear_fit at both halves of the y vector"
    if y.ndim != 1:
        raise AssertionError(f"Array must be 1d but is shape {y.ndim}")
    if padlen is None or padlen == 0:
        return y
    else:
        if 2 * padlen > y.shape[0]:
            raise AssertionError(
                f"padlen = {padlen} but this cannot be greater"
                f"than half length of y which is {y.shape[0] / 2.0}"
            )
        y1 = predict_linear_fit(y[0:padlen], np.linspace(-padlen, -1, padlen))
        y2 = predict_linear_fit(
            y[-padlen:], np.linspace(padlen, 2 * padlen - 1, padlen)
        )
        return np.concatenate((y1, y, y2))


def use_linear_trend_to_extend_ends(y, padlen, axis=0):
    """
    Flat data to 1D if dimension is larger than 1D, and then apply
    use_linear_trend_to_extend_ends_1d.
    """
    if y.ndim < 2:
        return use_linear_trend_to_extend_ends_1d(y, padlen)
    yy = y.reshape((y.shape[0], -1))
    ans = np.apply_along_axis(use_linear_trend_to_extend_ends_1d, axis, yy, padlen)
    return ans


def butterworth(x, period, axis=-1, order=4, padlen=3 * (4 + 1), high_pass=False):
    """
    High values for order (eg 8), give more end effects, while a default value of 4 seems
    ok. Default padlen=3*(4+1) here, since we follow scipy.signal.filtfilt, which also
    implements padding. This assumes default length of 3*max(len(a),len(b)) where b,
    a = scipy.signal.butter(order, wn). Here one finds len(a)=len(b)=order+1, hence
    default padlen=3*(4+1) above. Higher period for cutoff -> Smaller frequencies for
    cutoff -> more higher frequencies are dropped -> more smoothing.
    """
    if period <= 1:
        ans = x
    else:
        n0 = x.shape[axis]
        padlen = int(round(padlen))
        x = use_linear_trend_to_extend_ends(x, padlen, axis=axis)
        # The critical frequency.
        # For a Butterworth filter, this is the point at which the gain drops to
        # 1/sqrt(2) that of the passband (the “-3 dB point”).
        # i.e. roughly the frequency at which we start filtering (higher frequencies
        # are dropped). I got the code with a 2 in there, not sure where that comes from,
        # but leaving for consistency.
        wn = 2 / float(period)
        if wn > 1.0:
            padlen = 0
            y = x
        else:
            # Numerator (b) and denominator (a) polynomials of the IIR filter
            b, a = scipy.signal.butter(order, wn)
            # filtfilt: Apply a digital filter forward and backward to a signal.
            # Note: padding just makes reference to extending the signal at the edges
            # before running the filer.
            # Native padding in filtfilt just duplicates data, which is not very good.
            # Better to use linear extension above, and then set no padding (padlen=0) in
            # filtfilt.
            y = scipy.signal.filtfilt(b, a, x, axis=axis, padlen=0)
        # Create a slicer to obtain the smoothed original data points
        # (i.e. without the slicing)
        slicer = [slice(None)] * x.ndim
        slicer[axis] = slice(padlen, padlen + n0)
        # We have applied a low pass filder, so if you want the high pass substract
        # the smoothed signal from original
        if high_pass:
            ans = x[tuple(slicer)] - y[tuple(slicer)]
        else:
            ans = y[tuple(slicer)]
    return ans


# =============================================
#          END BUTTERWORTH SMOOTHING
# =============================================


def apply_smoothing(data, smoothing_pars):
    """
    Apply a smoothing algorithm to the data based on the passed parameters.

    Two algorithms are supported: butterworth and savgol filter. The former
    requires one parameter to specify smoothing strength, while the latter
    requires two: one of the the size of the rolling window of points to evaluate
    and a second for the order of the polynomial to use to approximate them.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)
    """
    if smoothing_pars["type"] == "butterworth":
        smoothed_data = smooth(data, smoothing_pars["pars"][0])
    elif smoothing_pars["type"] == "savgol":
        smoothed_data = savgol_filter(data, *smoothing_pars["pars"])
    else:
        raise ValueError

    return smoothed_data


def load_esm_data(
    esm_data_file,
    recalc_emis=True,
    ocean_frac=OCEAN_FRAC,
    ninit=1,
    smoothing_pars={"type": "butterworth", "pars": [1]},
):
    """
    Read in ESM data from an existing file at 'esm_data_file'.

    These files contain lots of global mean ESM diagnostics for the carbon cycle in
    column-formatted text format, with each row corresponding to a yearly mean.
    They are read with np.genfromtxt, and the output is written to a 'Data' object
    (see above) with name 'gcm' here.
    Attributes of this objects are things like 'gcm.time', 'gcm.catm' or 'gcm.npp'.

    INPUT
    esm_data_file: String, full path to the input file, eg sum4_MRI-ESM1.txt
    recalc_emis:   Boolean. Switch to recalculate the emissions from mass
                   conservation, rather than using the input data.
    ocean_frac:  Float, fraction of the earth surface covered by oceans. Default 0.710369
                 is the HadCM3 ocean_frac. This value was actully used in calibration for
                 all models. ocean_frac is used to estimate the global mean ocean near
                 surface temperature from input Tglb and Tland (Ben's data only has these)
    ninit:       Integer, the number of initial time points (years) for establishing a
                 temperature baseline againt which we define the anomalies of surface and
                 ocean.
    smoothing_pars Scheme to use to smoothing the read in data. The variable must be a
                   dictionary with the type of algorithm to run (butterworth or savgol),
                   and the parameters necessary for that algorithm. The butterworth filter
                   only needs a parameter specifying the strength of the smoothing (1 is
                   equal to no smoothing), whereas the savgol smoothing requires two, the
                   number of points included in the rolling window and the degree of the
                   polynomial used to fit the points (see here for more information:
                   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)
                   Examples:
                   {"type": butterworth, "pars": [1]}, {"type": savgol, "pars": [21, 3]}

    OUTPUT
    output is written to a 'Data' object (see above) with name 'gcm' here.


    Columns in the input esm_data_file files with the following names represent the
    following quantities:
    time       year
    emission   CO2 emissions. Units: GtC/yr. It equal to -999.0, this represents missing
               data.
    CO2        atmospheric CO2 conc. Units: ppm
    tas        global mean near-surface temparature. Units: Kelvin
    tas_land   global land mean near-surface temparature. Units: Kelvin
    fgco2      global mean flux of carbon to the oceans. Positive represents CO2 uptake
               to the oceans. Units: GtCO2/yr.
               Notice that we are feeding here in CO2 units, not C. This is what the
               model expects, and converts it internally to C.
               So the output of the ocean model is already converted to GtCs.
    fgpp       global mean gross primary production (GPP). Units: GtC/yr.
    fnpp       global mean net primary production (NPP). Units: GtC/yr.
    fvegsoil   global mean flux of carbon from vegetation to the soil (litterfall flux).
               Units: GtC/yr.
    frh        global mean 'heterotrophic' respiration (soil respiration). Units: GtC/yr.
    ra         global mean 'autotrophic' respiration (vegetation respiration).
               Units: GtC/yr.
    fluc       global mean land-use emissions. Units: GtC/yr.
    cveg       total global carbon in the land vegetation. Units: GtC.
    csoil      total global carbon in the soil. Units: GtC.
    cprod      global mean carbon in forestry and agricultural products due to
               anthropogenic land-use. Not used here. For some models this may not be
               included, in which case it is just 0. Units: GtC.
    fcvegout   Residual global carbon flux from the vegetation to the atmosphere, which is
               not covered by any of the fluxes above.
    fcsoilout  Residual global carbon flux from the soil to the atmosphere, which is
               not covered by any of the fluxes above.
    fcvegoutcsoilin  Residual global carbon flux from the vegetation to the soil,
                     which is not covered by any of the fluxes above.
    """

    with open(esm_data_file, "r") as f1:
        esm_data = np.genfromtxt(f1, names=True)

    gcm_time = apply_smoothing(esm_data["time"], smoothing_pars)
    gcm_emis = apply_smoothing(esm_data["emission"], smoothing_pars)
    gcm_catm = apply_smoothing(esm_data["CO2"], smoothing_pars)
    gcm_tglb = apply_smoothing(esm_data["tas"] - KELVIN_0, smoothing_pars)
    gcm_tlnd = apply_smoothing(esm_data["tas_land"] - KELVIN_0, smoothing_pars)
    gcm_cveg = apply_smoothing(esm_data["cveg"], smoothing_pars)
    gcm_csoil = apply_smoothing(esm_data["csoil"], smoothing_pars)
    gcm_oflux = apply_smoothing(esm_data["fgco2"], smoothing_pars)
    gcm_lu = apply_smoothing(esm_data["fluc"], smoothing_pars)
    gcm_npp = apply_smoothing(esm_data["fnpp"], smoothing_pars)
    gcm_gpp = apply_smoothing(esm_data["fgpp"], smoothing_pars)
    gcm_lit = apply_smoothing(esm_data["fvegsoil"], smoothing_pars)
    gcm_rh = apply_smoothing(esm_data["frh"], smoothing_pars)
    gcm_ra = gcm_gpp - gcm_npp

    # The following are optional fluxes, that files may or may not include
    try:
        gcm_fcvegout = apply_smoothing(esm_data["fcvegout"], smoothing_pars)
    except ValueError:
        gcm_fcvegout = None
    try:
        gcm_fcsoilout = apply_smoothing(esm_data["fcsoilout"], smoothing_pars)
    except ValueError:
        gcm_fcsoilout = None
    try:
        gcm_fcvegoutcsoilin = apply_smoothing(
            esm_data["fcvegoutsoilin"], smoothing_pars
        )
    except ValueError:
        gcm_fcvegoutcsoilin = None

    # Calculate temperature anomalies of surface and ocean againt the baseline defined
    # by the number of initial datapoints supplied in ninit
    dtglb = gcm_tglb - np.mean(gcm_tglb[:ninit])
    dtlnd = gcm_tlnd - np.mean(gcm_tlnd[:ninit])

    # Derive the global mean near surface ocean temperature, given the means from the
    # total and land.
    dtocn = (dtglb - (1.0 - ocean_frac) * dtlnd) / ocean_frac

    # Often we don't have the GCM emissions in the input files.
    missing = False
    if gcm_emis[0] == -999.0:
        # -999. is mdi for these files, reset gcm_emis to zero and calculate later from
        # mass conservation.
        gcm_emis[:] = 0.0
        missing = True

    # If missing or recalc_emis is True, use mass conservation to calculate implied
    # emissions. Need the change in carbon in atm, veg, soil, ocean and land-use
    # to do the integration. (This assumes that these categories comprise all relevant
    # carbon inventories in the underlying model)
    if recalc_emis or missing:
        emis = calc_emissions(
            gcm_time,
            gcm_catm,
            gcm_cveg,
            gcm_csoil,
            gcm_oflux,
            smoothing_pars=smoothing_pars,
        )
    else:
        emis = gcm_emis

    gcm = Data(
        time=gcm_time,
        catm=gcm_catm,
        cveg=gcm_cveg,
        csoil=gcm_csoil,
        oflux=gcm_oflux,
        emis=emis,
        gcmemis=gcm_emis,
        lu=gcm_lu,
        dtglb=dtglb,
        dtocn=dtocn,
        dtlnd=dtlnd,
        gpp=gcm_gpp,
        npp=gcm_npp,
        lit=gcm_lit,
        rh=gcm_rh,
        ra=gcm_ra,
        fcvegout=gcm_fcvegout,
        fcsoilout=gcm_fcsoilout,
        fcvegoutcsoilin=gcm_fcvegoutcsoilin,
    )

    return gcm


def make_all_dirs(fullfilename):
    "Create necessary directories to host fullfilename."
    dirarr = fullfilename.split("/")
    dirarr = dirarr[1:-1]
    subdir = ""
    for subname in dirarr:
        subdir = subdir + "/" + subname
        if not os.path.exists(subdir):
            print("making dir: ", subdir)
            os.makedirs(subdir)
    return 1
