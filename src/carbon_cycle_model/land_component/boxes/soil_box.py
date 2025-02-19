"""
Soil box class.

This box keeps track of the carbon content in the soil, and calculates one associated
flux:
    - get_sres(temp_ano, catm): soil (heterotrophic) respiration (GtC/y).
"""

from carbon_cycle_model.land_component.boxes.abstract_box import AbstractLandBox
from carbon_cycle_model.land_component.boxes.utils import general_calibration_fun
from carbon_cycle_model import defaults


class SoilBox(AbstractLandBox):
    """Soil box class."""

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("csoil0", defaults.CSOIL0_DEFAULT))
        self.catm0 = kwargs.get("catm0", defaults.CATM0_DEFAULT)
        self.sres0 = kwargs.get("sres0", defaults.SRES0_DEFAULT)
        self.sres0_par = self.sres0 / self.stock0

        self.sres_c_half = kwargs.get("sres_c_half", defaults.SRES_C_HALF)
        self.sres_c_l = kwargs.get("sres_c_l", defaults.SRES_C_L)
        self.sres_t_e = kwargs.get("sres_t_e", defaults.SRES_T_E)
        self.sres_t_l = kwargs.get("sres_t_l", defaults.SRES_T_L)

    def get_sres(self, temp_ano, catm):
        """
        Soil respiration coefficient. It should be multiplied by the soil carbon content
        to obtain the total soil respiration flux.

        input:
        - temp_ano: temperature anomaly from pre-industrial (kelvin/celsius).
        - catm: atmospheric concentration of carbon dioxide (ppm).
        """

        return general_calibration_fun(
            self.sres0_par,
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
