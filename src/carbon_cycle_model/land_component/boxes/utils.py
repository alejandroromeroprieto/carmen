"""
File to store any auxiliary functions used by the land cycle class.
"""

import numpy as np


def general_calibration_fun(
    flux0, par_t_l, par_t_e, par_c_l, par_c_half, stock, stock0, catm0, dtglb, catm
):
    """
    Funtion used to emulate carbon fluxes through the modulation of said fluxes through
    variations in temperature and carbon stocks.

    The formula is:
    Flux = flux0 * (1 + par_t_l*T)*e^(-par_t_e*T)
                 * (1 + par_c_l * stock_ano) * 1 / e^(stock_ano^2)
                 * (catm / (catm + par_c_half)) / ((catm_0 + par_c_half) / catm0)

    where:
    - T: temperature anomaly from pre-industrial (Kelvin/celsius) (dtglb).
    - stock_ano: normalised carbon stock anomaly from pre-industrial
                 i.e., (stock - stock0)/stock0)
    - catm: atmospheric carbon concentration (ppm)
    - catm0: pre-industrial atmospheric carbon concentration (ppm)
    - par_t_l: parameter controlling linear modulation with temperature.
    - par_t_e: parameter controlling exponential modulation with temperature.
    - par_c_l: parameter controlling linear modulation with carbon stock.
    - par_c_half: parameter controlling fertilisation effects of Co2. This parameter is
                  also known as half-saturation constant.
    """
    f_temp = (1.0 + par_t_l * dtglb) * np.exp(-par_t_e * dtglb)
    stock_pro = (stock - stock0) / stock0
    f_stock = 1 + (par_c_l * stock_pro) / (np.exp(stock_pro**2))
    # Divide by the initial factor value to normalise it to 1 at pre-industrial
    f_co2 = (catm / (catm + par_c_half)) * ((catm0 + par_c_half) / catm0)

    return flux0 * f_temp * f_stock * f_co2
