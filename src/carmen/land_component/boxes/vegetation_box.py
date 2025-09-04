"""
Vegetation box code.

This box keeps track of the carbon content in vegetation, and calculates
four associated fluxes:
    - get_vres(temp_ano, catm): vegetation (autotrophic) respiration (GtC/y).
    - get_gpp(temp_ano, catm): Gross primary production (GtC/y).
    - get_npp(temp_ano, catm): net primary production (GtC/y).
    - get_litterfall(temp_ano, catm): litterfall (GtC/y).
"""

from carmen.land_component.boxes.abstract_box import AbstractLandBox
from carmen.land_component.boxes.utils import general_calibration_fun
from carmen import defaults


class VegetationBox(AbstractLandBox):
    """Vegetation box class."""

    def __init__(self, timestep, **kwargs):
        super().__init__(kwargs.get("cveg0", defaults.CVEG0_DEFAULT))

        # Store pre-industrial values
        self.catm0 = kwargs.get("catm0", defaults.CATM0_DEFAULT)
        self.gpp0 = kwargs.get("gpp0", defaults.GPP0_DEFAULT)
        self.lit0 = kwargs.get("lit0", defaults.LIT0_DEFAULT)
        self.npp0 = kwargs.get("npp0", defaults.NPP0_DEFAULT)
        self.vres0 = kwargs.get("vres0", defaults.VRES0_DEFAULT)

        # parameter values (divided by stocks, which is what we used
        # for training the parameters)
        self.lit0_par = self.lit0 / self.stock0
        self.vres0_par = self.vres0 / self.stock0

        # vres parameters
        self.vres_c_half = kwargs.get("vres_c_half", defaults.VRES_C_HALF)
        self.vres_c_l = kwargs.get("vres_c_l", defaults.VRES_C_L)
        self.vres_t_e = kwargs.get("vres_t_e", defaults.VRES_T_E)
        self.vres_t_l = kwargs.get("vres_t_l", defaults.VRES_T_L)
        self.vres_c_e = kwargs.get("vres_c_e")
        self.vres_hyst = kwargs.get("vres_hyst")

        # inertia vres parameters
        self.vres_fast_inertia = kwargs.get("vres_fast", 1) ** timestep
        self.vres_prev_fast_inertia = 0.0
        self.vres_c_tan = kwargs.get("vres_c_tan")

        # GPP parameters
        self.gpp_c_half = kwargs.get("gpp_c_half", defaults.GPP_C_HALF)
        self.gpp_c_l = kwargs.get("gpp_c_l", defaults.GPP_C_L)
        self.gpp_t_e = kwargs.get("gpp_t_e", defaults.GPP_T_E)
        self.gpp_t_l = kwargs.get("gpp_t_l", defaults.GPP_T_L)
        self.gpp_c_e = kwargs.get("gpp_c_e")
        self.gpp_hyst = kwargs.get("gpp_hyst")

        # inertia gpp parameters
        self.gpp_fast_inertia = kwargs.get("gpp_fast", 1) ** timestep
        self.gpp_prev_fast_inertia = 0.0
        self.gpp_c_tan = kwargs.get("gpp_c_tan")

        # NPP parameters
        self.npp_c_half = kwargs.get("npp_c_half", defaults.NPP_C_HALF)
        self.npp_c_l = kwargs.get("npp_c_l", defaults.NPP_C_L)
        self.npp_t_e = kwargs.get("npp_t_e", defaults.NPP_T_E)
        self.npp_t_l = kwargs.get("npp_t_l", defaults.NPP_T_L)
        self.npp_c_e = kwargs.get("npp_c_e")
        self.npp_hyst = kwargs.get("npp_hyst")

        # inertia NPP parameters
        self.npp_fast_inertia = kwargs.get("npp_fast", 1) ** timestep
        self.npp_prev_fast_inertia = 0.0
        self.npp_c_tan = kwargs.get("npp_c_tan")

        # Litterfall parameters
        self.lit_c_half = kwargs.get("lit_c_half", defaults.LIT_C_HALF)
        self.lit_c_l = kwargs.get("lit_c_l", defaults.LIT_C_L)
        self.lit_t_e = kwargs.get("lit_t_e", defaults.LIT_T_E)
        self.lit_t_l = kwargs.get("lit_t_l", defaults.LIT_T_L)
        self.lit_c_e = kwargs.get("lit_c_e")
        self.lit_hyst = kwargs.get("lit_hyst")

        # inertia litterfall parameters
        self.lit_fast_inertia = kwargs.get("lit_fast", 1) ** timestep
        self.lit_prev_fast_inertia = 0.0
        self.lit_c_tan = kwargs.get("lit_c_tan")

    def get_vres(self, temp_ano, catm, hyst_signal_t):
        """
        Vegetation respiration coefficient. It should be multiplied by the vegetation
        carbon content to obtain the total vegetation respiration flux.

        input:
        - temp_ano: temperature anomaly from pre-industrial (kelvin/celsius).
        - catm: atmospheric concentration of carbon dioxide (ppm).
        - hyst_signal_t: difference between current temperature anomaly and highest
                         temperature anomaly experienced by the model thus far
                         (kelvin/celsius).
        """

        # Calculate and store the inertia-associated factor recursively
        self.vres_prev_fast_inertia = (
            self.vres_fast_inertia * self.vres_prev_fast_inertia
            + (1 - self.vres_fast_inertia) * temp_ano
        )

        f_inertia = 1.0 + self.vres_c_tan * self.vres_prev_fast_inertia

        return general_calibration_fun(
            self.vres0_par,
            self.vres_t_l,
            self.vres_t_e,
            self.vres_c_l,
            self.vres_c_half,
            self.vres_c_e,
            self.vres_hyst,
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

    def get_gpp(self, temp_ano, catm, hyst_signal_t):
        """
        Gross primary production.

        input:
        - temp_ano: temperature anomaly from pre-industrial (kelvin/celsius).
        - catm: atmospheric concentration of carbon dioxide (ppm).
        - hyst_signal_t: difference between current temperature anomaly and highest
                         temperature anomaly experienced by the model thus far
                         (kelvin/celsius).
        """

        # Calculate and store the inertia-associated factor recursively
        self.gpp_prev_fast_inertia = (
            self.gpp_fast_inertia * self.gpp_prev_fast_inertia
            + (1 - self.gpp_fast_inertia) * temp_ano
        )

        f_inertia = 1.0 + self.gpp_c_tan * self.gpp_prev_fast_inertia

        return general_calibration_fun(
            self.gpp0,
            self.gpp_t_l,
            self.gpp_t_e,
            self.gpp_c_l,
            self.gpp_c_half,
            self.gpp_c_e,
            self.gpp_hyst,
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

    def get_npp(self, temp_ano, catm, hyst_signal_t):
        """
        Net primary production. In theory, this should be equal to GPP - vres.

        input:
        - temp_ano: temperature anomaly from pre-industrial (kelvin/celsius).
        - catm: atmospheric concentration of carbon dioxide (ppm).
        - hyst_signal_t: difference between current temperature anomaly and highest
                         temperature anomaly experienced by the model thus far
                         (kelvin/celsius).
        """

        # Calculate and store the inertia-associated factor recursively
        self.npp_prev_fast_inertia = (
            self.npp_fast_inertia * self.npp_prev_fast_inertia
            + (1 - self.npp_fast_inertia) * temp_ano
        )

        f_inertia = 1.0 + self.npp_c_tan * self.npp_prev_fast_inertia

        return general_calibration_fun(
            self.npp0,
            self.npp_t_l,
            self.npp_t_e,
            self.npp_c_l,
            self.npp_c_half,
            self.npp_c_e,
            self.npp_hyst,
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

    def get_litterfall(self, temp_ano, catm, hyst_signal_t):
        """
        Litterfall coefficient. It should be multiplied by the vegetation carbon content
        to obtain the total litterfall flux.

        input:
        - temp_ano: temperature anomaly from pre-industrial (kelvin/celsius).
        - catm: atmospheric concentration of carbon dioxide (ppm).
        - hyst_signal_t: difference between current temperature anomaly and highest
                         temperature anomaly experienced by the model thus far
                         (kelvin/celsius).
        """

        # Calculate and store the inertia-associated factor recursively
        self.lit_prev_fast_inertia = (
            self.lit_fast_inertia * self.lit_prev_fast_inertia
            + (1 - self.lit_fast_inertia) * temp_ano
        )

        f_inertia = 1.0 + self.lit_c_tan * self.lit_prev_fast_inertia

        return general_calibration_fun(
            self.lit0_par,
            self.lit_t_l,
            self.lit_t_e,
            self.lit_c_l,
            self.lit_c_half,
            self.lit_c_e,
            self.lit_hyst,
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
