"""
Defaults for model parameter values. They were taken from an UKESM1-0-LL cross-experiment
calibration. The model will default to using these numbers if the relevant quantity is not
supplied by the user (although not entirely sure how useful that is if the user-supplied
values are from a radically different carbon cycle model).
"""

# Stocks
CVEG0_DEFAULT = 581.64545517
CSOIL0_DEFAULT = 1766.34032645
CATM0_DEFAULT = 284.17452497

# Initial fluxes
GPP0_DEFAULT = 116.94204612
LIT0_DEFAULT = 57.83547102
NPP0_DEFAULT = 58.83742481
VRES0_DEFAULT = 58.24303196
SRES0_DEFAULT = 57.85194986

# Parameter values
GPP_C_HALF = 145.28992779
GPP_C_L = 0.21474753
GPP_T_E = 0.02881768
GPP_T_L = 0.07661808

VRES_C_HALF = 125.05320426
VRES_C_L = -0.59067791
VRES_T_E = 0.00089676
VRES_T_L = 0.05400144

NPP_C_HALF = 156.41774418
NPP_C_L = 0.00526843
NPP_T_E = 0.04929246
NPP_T_L = 0.09369002

LIT_C_HALF = 111.34304853
LIT_C_L = -0.8934007
LIT_T_E = 0.02584605
LIT_T_L = 0.06023067

SRES_C_HALF = 78.72382148
SRES_C_L = -0.49845692
SRES_T_E = -0.03421489
SRES_T_L = 7.2e-07

DOCN_DEFAULT = 42.72954548
DOCNFAC_DEFAULT = 1.99787663
OCNTEMP_DEFAULT = 0.02276231
DOCNTEMP_DEFAULT = -0.0120341
