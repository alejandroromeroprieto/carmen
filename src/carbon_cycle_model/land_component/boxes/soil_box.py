"""
Soil box code.

This box keeps track of its own carbon stock, and calculates
one associated fluxes:
    - get_sres(temp_ano, catm): soil (heterotrophic) respiration (GtC/y).
where:
    - temp_ano: temperature anomaly. (kelvin/celsius)
    - catm: atmospheric concentration of caron (ppm)
"""

from carbon_cycle_model.land_component.boxes.abstract_box import AbstractLandBox
from carbon_cycle_model.land_component.boxes.utils import general_calibration_fun

# Defaults
# TODO: GET RIGHT DEFAULTS
CSOIL0_DEFAULT = 200
CATM0_DEFAULT = 100
SRES0_DEFAULT = 100

SRES_C_HALF = 100
SRES_C_L = 100
SRES_T_E = 1
SRES_T_L = 100


class SoilBox(AbstractLandBox):
    """Soil box class."""

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("stock", CSOIL0_DEFAULT))
        self.catm0 = kwargs.get("catm0", CATM0_DEFAULT)
        self.sres0 = kwargs.get("sres0", SRES0_DEFAULT)

        self.sres_c_half = kwargs.get("sres_c_half", SRES_C_HALF)
        self.sres_c_l = kwargs.get("sres_c_l", SRES_C_L)
        self.sres_t_e = kwargs.get("sres_t_e", SRES_T_E)
        self.sres_t_l = kwargs.get("sres_t_l", SRES_T_L)

    def get_sres(self, temp_ano, catm):
        """Soil respiration coefficient."""

        return general_calibration_fun(
            self.sres0,
            self.sres_t_l,
            self.sres_t_e,
            self.sres_c_l,
            self.sres_c_half,
            self.stock,
            self.stock0,
            self.catm0,
            temp_ano,
            catm,
        )
