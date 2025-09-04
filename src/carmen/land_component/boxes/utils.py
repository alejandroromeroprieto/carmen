"""
File to store any auxiliary functions used by the land cycle boxes.

Right now the only function stored is the general_calibration_fun() which
codifies the modulation of land-based carbon fluxes to climate variables.
"""

import numpy as np


def general_calibration_fun(
    flux0,
    par_t_l,
    par_t_e,
    par_c_l,
    par_c_half,
    par_c_e,
    par_hyst,
    par_c_tan,
    par_alpha_fast,
    stock_ano,
    stock0,
    catm0,
    dtglb,
    catm,
    hyst_signal=None,
    f_inertia_factor=None,
):
    """
    Funtion used to emulate carbon fluxes through the modulation of said fluxes through
    variations in temperature and carbon stocks.

    The formula is:
    Flux = flux0 * (1 + par_t_l*T)*e^(par_t_e*T)
                 * (1 + par_c_l * stock_ano /(1 + stock_ano^2)) * e^(par_c_e * stock_ano)
                 * (catm / (catm + par_c_half)) / ((catm_0 + par_c_half) / catm0)
                 * (1 + par_hyst * (peak_temperature - T))
                 * 1 + par_inertia * EMA(T)

    where:
    - flux0: pre-industrial flux.
    - par_t_l: parameter controlling linear modulation with temperature.
    - par_t_e: parameter controlling exponential modulation with temperature.
    - par_c_l: parameter controlling linear modulation with carbon stock.
    - par_c_half: parameter controlling fertilisation effects of Co2. This parameter is
                  also known as half-saturation constant.
    - par_c_e: parameter controlling exponential modulation with carbon stock.
    - par_hyst: parameter controlling hysteresis-associated modulated with difference with
                respect to the maximum experienced temperature.
    - par_c_tan: parameter controlling the modulation with the exponential moving
                 average of temperature anomaly.
    - par_alpha_fast: parameter controlling the "memory" of the exponential
                      moving average of temperatuer anomaly.
    - stock_ano: carbon stock anomaly for the relevant ecosystem box.
    - stock0: pre-industrial carbon stock.
    - catm0: pre-industrial atmospheric carbon concentration (ppm)
    - dtglb (T): temperature anomaly from pre-industrial (Kelvin/celsius).
    - catm: atmospheric carbon concentration (ppm)
    - hyst_signal: difference between current and maximum experienced temperatures.
    - f_inertia_factor: if supplied, use this inertia factor instead of computing a new one.
    """
    # Temperature factor
    f_temp = (1.0 + par_t_l * dtglb) * np.exp(par_t_e * dtglb)

    # Stock factor
    stock_pro = stock_ano / stock0

    f_stock = (1.0 + (par_c_l * stock_pro) / (1 + stock_pro**2)) * np.minimum(
        5, np.exp(par_c_e * stock_pro)
    )

    # CO2 fertilization factor
    # Notice we divide by the initial factor value to normalise it to 1 at pre-industrial
    f_co2 = (catm / (catm + par_c_half)) * ((catm0 + par_c_half) / catm0)

    # If hysteresis signal was not supplied, calculate it
    # This signal is just the difference in temperature anomaly with respect
    # to the highest anomaly the model has seen in this run
    if hyst_signal is None:
        dtglb_array = np.atleast_1d(dtglb)
        peak_t = np.maximum.accumulate(dtglb_array)
        hyst_signal = peak_t - dtglb_array

    f_hyst = 1 + par_hyst * hyst_signal

    if f_inertia_factor is None:
        # Initialize inertia variables
        f_inertia_fast = np.zeros_like(dtglb)  # fast response timescale (~decades)

        for t in range(1, len(dtglb)):
            f_inertia_fast[t] = (
                par_alpha_fast * f_inertia_fast[t - 1] + (1 - par_alpha_fast) * dtglb[t]
            )

        f_inertia = 1.0 + par_c_tan * f_inertia_fast
    else:
        f_inertia = f_inertia_factor

    return flux0 * np.clip(f_temp * f_stock * f_co2 * f_hyst * f_inertia, 0.2, 5.0)
