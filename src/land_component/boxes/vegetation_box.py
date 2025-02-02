"""
Vegetation box code.

This box keeps track of its own carbon stock, and calculates
four associated fluxes:
    - get_vres(temp_ano, catm): vegetation (autotrophic) respiration (GtC/y).
    - get_gpp(temp_ano, catm): Gross primary production (GtC/y).
    - get_npp(temp_ano, catm): net primary production (GtC/y).
    - get_litterfall(temp_ano, catm): litterfall (GtC/y).
where:
    - temp_ano: temperature anomaly. (kelvin/celsius)
    - catm: atmospheric concentration of caron (ppm)
"""

from land_component.boxes.abstract_box import AbstractLandBox
from land_component.boxes.utils import general_calibration_fun

# Defaults
# TODO: GET RIGHT DEFAULTS
CVEG0_DEFAULT = 200
CATM0_DEFAULT = 100
GPP0_DEFAULT = 100
LIT0_DEFAULT = 100
NPP0_DEFAULT = 100
VRES0_DEFAULT = 100

VRES_C_HALF = 100
VRES_C_L = 100
VRES_T_E = 1
VRES_T_L = 100

GPP_C_HALF = 100
GPP_C_L = 100
GPP_T_E = 1
GPP_T_L = 100

NPP_C_HALF = 100
NPP_C_L = 100
NPP_T_E = 1
NPP_T_L = 100

LIT_C_HALF = 100
LIT_C_L = 100
LIT_T_E = 1
LIT_T_L = 100


class VegetationBox(AbstractLandBox):
    """Vegetation box class."""

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("stock", CVEG0_DEFAULT))
        self.catm0 = kwargs.get("catm0", CATM0_DEFAULT)
        self.gpp0 = kwargs.get("gpp0", GPP0_DEFAULT)
        self.lit0 = kwargs.get("lit0", LIT0_DEFAULT)
        self.npp0 = kwargs.get("npp0", NPP0_DEFAULT)
        self.vres0 = kwargs.get("vres0", VRES0_DEFAULT)

        self.vres_c_half = kwargs.get("vres_c_half", VRES_C_HALF)
        self.vres_c_l = kwargs.get("vres_c_l", VRES_C_L)
        self.vres_t_e = kwargs.get("vres_t_e", VRES_T_E)
        self.vres_t_l = kwargs.get("vres_t_l", VRES_T_L)

        self.gpp_c_half = kwargs.get("gpp_c_half", GPP_C_HALF)
        self.gpp_c_l = kwargs.get("gpp_c_l", GPP_C_L)
        self.gpp_t_e = kwargs.get("gpp_t_e", GPP_T_E)
        self.gpp_t_l = kwargs.get("gpp_t_l", GPP_T_L)

        self.npp_c_half = kwargs.get("npp_c_half", NPP_C_HALF)
        self.npp_c_l = kwargs.get("npp_c_l", NPP_C_L)
        self.npp_t_e = kwargs.get("npp_t_e", NPP_T_E)
        self.npp_t_l = kwargs.get("npp_t_l", NPP_T_L)

        self.lit_c_half = kwargs.get("lit_c_half", LIT_C_HALF)
        self.lit_c_l = kwargs.get("lit_c_l", LIT_C_L)
        self.lit_t_e = kwargs.get("lit_t_e", LIT_T_E)
        self.lit_t_l = kwargs.get("lit_t_l", LIT_T_L)

    def get_vres(self, temp_ano, catm):
        """Vegetation respiration coefficient."""

        return general_calibration_fun(
            self.vres0,
            self.vres_t_l,
            self.vres_t_e,
            self.vres_c_l,
            self.vres_c_half,
            self.stock,
            self.stock0,
            self.catm0,
            temp_ano,
            catm,
        )

    def get_gpp(self, temp_ano, catm):
        """Gross primary production."""

        return general_calibration_fun(
            self.gpp0,
            self.gpp_t_l,
            self.gpp_t_e,
            self.gpp_c_l,
            self.gpp_c_half,
            self.stock,
            self.stock0,
            self.catm0,
            temp_ano,
            catm,
        )

    def get_npp(self, temp_ano, catm):
        """Net primary production. (GPP in vres in theory)."""

        return general_calibration_fun(
            self.npp0,
            self.npp_t_l,
            self.npp_t_e,
            self.npp_c_l,
            self.npp_c_half,
            self.stock,
            self.stock0,
            self.catm0,
            temp_ano,
            catm,
        )

    def get_litterfall(self, temp_ano, catm):
        """Litterfall. (GPP in vres in theory)."""

        return general_calibration_fun(
            self.lit0,
            self.lit_t_l,
            self.lit_t_e,
            self.lit_c_l,
            self.lit_c_half,
            self.stock,
            self.stock0,
            self.catm0,
            temp_ano,
            catm,
        )
