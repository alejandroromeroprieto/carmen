"""
File to store any auxiliary functions used by the land cycle boxes.
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
    par_alpha_slow,
    par_c_tan2,
    stock_ano,
    stock0,
    catm0,
    dtglb,
    catm,
    hyst_signal=None,
    f_inertia_factor=None,
    testing=False,
):
    """
    Funtion used to emulate carbon fluxes through the modulation of said fluxes through
    variations in temperature and carbon stocks.

    The formula is:
    Flux = flux0 * (1 + par_t_l*T)*e^(-par_t_e*T)
                 * (1 + par_c_l * stock_ano /(1 + stock_ano^2)) * e^(par_c_e * stock_ano)
                 * (catm / (catm + par_c_half)) / ((catm_0 + par_c_half) / catm0)
                 * (1 + par_hyst * (peak_temperature - T))

    where:
    - flux0: pre-industrial flux.
    - stock_ano: carbon stock anomaly for the relevant ecosystem box.
    - stock0: pre-industrial carbon stock.
    - dtglb (T): temperature anomaly from pre-industrial (Kelvin/celsius).
    - stock_ano: normalised carbon stock anomaly from pre-industrial
                 i.e., (stock - stock0)/stock0)
    - catm: atmospheric carbon concentration (ppm)
    - catm0: pre-industrial atmospheric carbon concentration (ppm)
    - par_t_l: parameter controlling linear modulation with temperature.
    - par_t_e: parameter controlling exponential modulation with temperature.
    - par_c_l: parameter controlling linear modulation with carbon stock.
    - par_c_half: parameter controlling fertilisation effects of Co2. This parameter is
                  also known as half-saturation constant.
    - par_c_e: parameter controlling exponential modulation with carbon stock.
    - par_hyst: parameter controlling hysteresis-associated modulated with difference with
                respect to the maximum experienced temperature.
    - hyst_signal: difference between current and maximum experienced temperatures.
    """

    if f_inertia_factor is None:
        # Initialize inertia variables
        f_inertia_fast = np.zeros_like(dtglb)
        f_inertia_slow = np.zeros_like(dtglb)

        # par_alpha_fast = par_par_alpha_fast  # fast response timescale (~decades)
        # par_alpha_slow = par_par_alpha_slow  # slow response timescale (~centuries)

        for t in range(1, len(dtglb)):
            f_inertia_fast[t] = par_alpha_fast * f_inertia_fast[t-1] + (1 - par_alpha_fast) * dtglb[t]
            f_inertia_slow[t] = par_alpha_slow * f_inertia_slow[t-1] + (1 - par_alpha_slow) * dtglb[t]

        f_inertia = 1.0 + par_c_tan * f_inertia_fast + par_c_tan2 * f_inertia_slow
    else:
        f_inertia = f_inertia_factor
    
    # Temperature factor
    f_temp = (1.0 + par_t_l * dtglb) * np.exp(par_t_e * dtglb)

    # Stock factor
    stock_pro = stock_ano / stock0
    # f_stock = (1.0 + (par_c_l * stock_pro) / (1 + stock_pro**2)) * np.exp(
    #     par_c_e * np.maximum(-0.2, stock_pro)
    # )

    f_stock = (1.0 + (par_c_l * stock_pro) / (1 + stock_pro**2) ) * np.minimum(5, np.exp(
        par_c_e * stock_pro
    ))

    # stock_pro = stock_ano / stock0
    # f_stock = (1 + par_c_l* np.tanh(par_c_e * stock_pro))
    # f_stock = (1 + par_c_l* np.tanh(par_c_e * stock_pro + np.exp(par_c_e * stock_pro)  ))
    # f_stock = (1.0 + (par_c_l * stock_pro) / (1 + 2*stock_pro**2))
    # f_stock = (1.0 + (par_c_l * stock_pro))* np.exp(
    #     par_c_e * stock_pro
    # )

    # f_stock = 0.5 +0.5*np.exp(-np.abs(par_c_l)*stock_pro**2)

    # f_stock = 1

    # CO2 fertilization factor
    # Notice we divide by the initial factor value to normalise it to 1 at pre-industrial
    f_co2 = (catm / (catm + par_c_half)) * ((catm0 + par_c_half) / catm0)

    # If hysteresis signal was not supplied, calculate it
    # This signal is just the difference in temperature anomaly with respect
    # to the highest anomaly the model has seen in this run
    if hyst_signal is None:
        dtglb_array = np.atleast_1d(dtglb)
        peak_T = np.maximum.accumulate(dtglb_array)
        hyst_signal = peak_T - dtglb_array

        # IMplement logistic model with non-0 start?

    f_hyst = 1 + par_hyst * hyst_signal

    # print(flux0, f_temp, f_stock, f_co2, f_hyst, f_inertia)
    # f_temp + "asd"

    return flux0 * np.clip(f_temp * f_stock * f_co2 * f_hyst * f_inertia, 0.2, 5.0)


# def calculate_intertia_factor(dtglb_t, par_c_tan, par_c_tan2, f_fast_prev, f_slow_prev, alpha_fast, alpha_slow, timestep_sub_annual=None):
#     if timestep_sub_annual:
#         alpha_fast_run = alpha_fast ** timestep_sub_annual
#         alpha_slow_run = alpha_slow ** timestep_sub_annual

#         # Update f_fast, f_slow
#         f_fast_new = alpha_fast_run * f_fast_prev + (1 - alpha_fast_run) * dtglb_t
#         f_slow_new = alpha_slow_run * f_slow_prev + (1 - alpha_slow_run) * dtglb_t

#         gain_fast_cal = 1.0 / (1 - alpha_fast)
#         gain_fast_run = 1.0 / (1 - alpha_fast_run)
#         scaling_fast = gain_fast_cal / gain_fast_run
#         par_c_tan = par_c_tan * scaling_fast

#         gain_slow_cal = 1.0 / (1 - alpha_slow)
#         gain_slow_run = 1.0 / (1 - alpha_slow_run)
#         scaling_slow = gain_slow_cal / gain_slow_run
#         par_c_tan2 = par_c_tan2 * scaling_slow
#     else:
#         f_fast_new = alpha_fast * f_fast_prev + (1 - alpha_fast) * dtglb_t
#         f_slow_new = alpha_slow * f_slow_prev + (1 - alpha_slow) * dtglb_t

#     f_inertia = 1.0 + par_c_tan * f_fast_new + par_c_tan2 * f_slow_new

#     return f_inertia, 