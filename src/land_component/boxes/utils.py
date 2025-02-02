"""
File to store any auxiliary functions used by the land cycle
"""

import numpy as np


def general_calibration_fun(
    flux0, par_t_l, par_t_e, par_c_l, par_c_half, stock, stock0, catm0, dtglb, catm
):
    """
    General funtion used to calibrate the modulation of carbon
    fluxes based on temperature and carbon stocks.
    """
    f_temp = (1.0 + par_t_l * dtglb) * np.exp(-par_t_e * dtglb)
    stock_pro = (stock - stock0) / stock0
    f_stock = 1 + (par_c_l * stock_pro) / (np.exp(stock_pro**2))
    # Divide by the initial parameter value to normalise it to 1 at pre-industrial
    f_co2 = (catm / (catm + par_c_half)) * ((catm0 + par_c_half) / catm0)

    return flux0 * f_temp * f_stock * f_co2
