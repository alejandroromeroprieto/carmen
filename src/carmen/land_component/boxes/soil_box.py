"""
Soil box class.

This box keeps track of the carbon content in the soil, and calculates one associated
flux:
    - get_sres(temp_ano, catm): soil (heterotrophic) respiration (GtC/y).
"""

from carmen.land_component.boxes.abstract_box import AbstractLandBox
from carmen.land_component.boxes.utils import general_calibration_fun
from carmen import defaults


class SoilBox(AbstractLandBox):
    """Soil box class."""

    def __init__(self, timestep, **kwargs):
        super().__init__(kwargs.get("csoil0", defaults.CSOIL0_DEFAULT))
        self.catm0 = kwargs.get("catm0", defaults.CATM0_DEFAULT)
        self.sres0 = kwargs.get("sres0", defaults.SRES0_DEFAULT)
        self.sres0_par = self.sres0 / self.stock0

        self.sres_c_half = kwargs.get("sres_c_half", defaults.SRES_C_HALF)
        self.sres_c_l = kwargs.get("sres_c_l", defaults.SRES_C_L)
        self.sres_t_e = kwargs.get("sres_t_e", defaults.SRES_T_E)
        self.sres_t_l = kwargs.get("sres_t_l", defaults.SRES_T_L)
        self.sres_c_e = kwargs.get("sres_c_e")
        self.sres_hyst = kwargs.get("sres_hyst")

        # inertia parameters
        self.sres_fast_inertia = kwargs.get("sres_fast", 1) ** timestep
        self.sres_slow_inertia = kwargs.get("sres_slow", 1) ** timestep
        self.sres_prev_fast_inertia = 0.0
        self.sres_prev_slow_inertia = 0.0

        self.sres_c_tan = kwargs.get("sres_c_tan")
        self.sres_c_tan2 = kwargs.get("sres_c_tan2", 0)

    def get_sres(self, temp_ano, catm, hyst_signal_t):
        """
        Soil respiration coefficient. It should be multiplied by the soil carbon content
        to obtain the total soil respiration flux.

        input:
        - temp_ano: temperature anomaly from pre-industrial (kelvin/celsius).
        - catm: atmospheric concentration of carbon dioxide (ppm).
        - hyst_signal_t: difference between current temperature anomaly and highest
                         temperature anomaly experienced by the model thus far
                         (kelvin/celsius).
        """

        # Calculate and store the inertia-associated factor recursively
        self.sres_prev_fast_inertia = (
            self.sres_fast_inertia * self.sres_prev_fast_inertia
            + (1 - self.sres_fast_inertia) * temp_ano
        )

        f_inertia = 1.0 + self.sres_c_tan * self.sres_prev_fast_inertia

        return general_calibration_fun(
            self.sres0_par,
            self.sres_t_l,
            self.sres_t_e,
            self.sres_c_l,
            self.sres_c_half,
            self.sres_c_e,
            self.sres_hyst,
            None,
            None,
            self.stock - self.stock0,
            self.stock0,
            self.catm0,
            temp_ano,
            catm,
            hyst_signal=hyst_signal_t,
            f_inertia_factor=f_inertia,
        )
