"""
File to store any auxiliary functions for the ocean carbon cycle.
"""

import math
import numpy as np


def joos_response(time_array):
    """
    Return an array of the same length as the input time array with the value for the
    impulse response function -- r_s in Joos et al. (1996) -- fitted to the Princeton 3D
    model at each timepoint in time. This IRF is used to approximate the export of carbon
    to the deep ocean from the mixed layer.
    """
    ans = np.zeros(time_array.shape[0])
    for i in range(0, time_array.shape[0]):
        ttt = time_array[i] - time_array[0]
        if ttt >= 1.0:
            ans[i] = (
                0.014819
                + 0.703670 * math.exp(-ttt / 0.70177)
                + 0.249660 * math.exp(-ttt / 2.3488)
                + 0.066485 * math.exp(-ttt / 15.281)
                + 0.038344 * math.exp(-ttt / 65.359)
                + 0.019439 * math.exp(-ttt / 347.55)
            )
        else:
            ans[i] = 1.0 + ttt * (
                -2.2617
                + ttt
                * (
                    14.002
                    + ttt * (-48.770 + ttt * (82.986 + ttt * (-67.527 + ttt * 21.037)))
                )
            )
    return ans
