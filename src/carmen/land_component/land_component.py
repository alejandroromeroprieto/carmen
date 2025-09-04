"""
Class implementing the land component of the carbon cycle.

It includes two boxes: vegetation and soil. These boxes interact between them
and with an abstract atmosphere box which is simply tracked as a number - the
carbon dioxide concentration in the atmosphere.

"""

import numpy as np

from carmen.land_component.boxes.vegetation_box import VegetationBox
from carmen.land_component.boxes.soil_box import SoilBox
from carmen import defaults


class LandCarbonCycle:
    """
    Class implementing the land component of the carbon cycle.

    input:
    - dt: timestep size.
    - num_steps: number of steps the model is expected to run.
    - kwargs: dictionary with parameter values.

    This class stores as attributes the historical timeseries of:
    - cveg: carbon content in the vegetation pool.
    - csoil: carbon content in the soil pool.
    - gpp: gross primary production flux.
    - npp: net primary production flux.
    - vres: autotrophic respiration flux.
    - lit: litterfall flux.
    - sres: heterotrophic respiration flux.
    - fcva: exogenous carbon flux from vegetation to atmosphere. This includes
            any anthropogenic disturbances (e.g. fire, harvesting, etc...)
    - fcsa: exogenous carbon flux from soil to atmosphere. This includes
            any anthropogenic disturbances (e.g. fire, etc...)
    - fcvs: exogenous carbon flux from vegetation to soil. This includes
            any anthropogenic disturbances.

    as well as the following quantities:
    - catm0: pre-industrial atmospheric concentration of carbon (ppm).
    - dt: timestep size (years).
    - num_steps: number of steps the model will be run for.
    - peak_t: maximum experienced temperature anomaly by the model in this run.
    - old_t: temperature anomaly in previous timestep.
    """

    def __init__(self, dt, num_steps, **kwargs):
        # Create and initialise boxes
        self.veg_box = VegetationBox(timestep=dt, **kwargs)
        self.soil_box = SoilBox(timestep=dt, **kwargs)

        # pre-industrial value for atmos CO2 conc. Units: ppm.
        self.catm0 = kwargs.get("catm0", defaults.CATM0_DEFAULT)
        # Timestep. Unit: years
        self.dt = dt
        # Number of timesteps in the simulation. Unit: dimensionless.
        # (needed to avoid dynamic allocation, although this could change in the future)
        self.num_steps = num_steps

        # Create arrays to store results
        self.cveg = np.zeros(num_steps)
        self.csoil = np.zeros(num_steps)
        self.gpp = np.zeros(num_steps)
        self.npp = np.zeros(num_steps)
        self.vres = np.zeros(num_steps)
        self.lit = np.zeros(num_steps)
        self.sres = np.zeros(num_steps)
        self.fcva = np.zeros(num_steps)
        self.fcsa = np.zeros(num_steps)
        self.fcvs = np.zeros(num_steps)

        # Variable keeping track of the maximum temperature anomaly
        # experience by the model (required for flux emulation)
        self.peak_t = -np.inf

        self.old_t = 0

        # Populate first values
        self.cveg[0] = kwargs.get("cveg0", defaults.CVEG0_DEFAULT)
        self.csoil[0] = kwargs.get("csoil0", defaults.CSOIL0_DEFAULT)
        self.gpp[0] = kwargs.get("gpp0", defaults.GPP0_DEFAULT)
        self.npp[0] = kwargs.get("npp0", defaults.NPP0_DEFAULT)
        self.vres[0] = kwargs.get("vres0", defaults.VRES0_DEFAULT)
        self.lit[0] = kwargs.get("lit0", defaults.LIT0_DEFAULT)
        self.sres[0] = kwargs.get("sres0", defaults.SRES0_DEFAULT)

        # Current timestep index
        self.timestep_ind = 0

    def update(
        self,
        temp_ano,
        catm,
        npp_flag=True,
        fcva=0,
        fcsa=0,
        fcvs=0,
    ):
        """Run the model one time step into the future.

        This requires the following input:
        - temp_ano: current temperature difference from pre-industrial (Kelvin/celsius).
        - catm: atmospheric concentration of carbon dioxide (ppm).
        - npp_flag: whether to use the npp flux, or the GPP/vres fluxes.
        - fcva: exogenous flux of carbon from vegetation to atmosphere.
        - fcsa: exogenous flux of carbon from soil to atmosphere.
        - fcvs: exogenous flux of carbon from vegetation to soil.

        return values:
        - Increment of carbon stock in the vegetation pool.
        - Increment of carbon stock in the soil pool.
        """
        # Update temperature hysteresis factor
        self.peak_t = max(self.peak_t, temp_ano)
        hyst_signal_t = self.peak_t - temp_ano

        # Calculate new fluxes
        if npp_flag:
            npp = self.veg_box.get_npp(temp_ano, catm, hyst_signal_t)
        else:
            gpp = self.veg_box.get_gpp(temp_ano, catm, hyst_signal_t)
            vres = self.veg_box.get_vres(temp_ano, catm, hyst_signal_t)

        lit = self.veg_box.get_litterfall(temp_ano, catm, hyst_signal_t)
        sres = self.soil_box.get_sres(temp_ano, catm, hyst_signal_t)

        # Calculate new carbon stocks using an implicit euler scheme
        if npp_flag:
            cvegnew = (
                self.cveg[self.timestep_ind] * (1.0 + self.dt * (0 - lit))
                + self.dt * npp
                - self.dt * (fcva + fcvs)
            )
        else:
            cvegnew = (
                self.cveg[self.timestep_ind] * (1.0 + self.dt * (0 - lit - vres))
                + self.dt * gpp
                - self.dt * (fcva + fcvs)
            )
        csoilnew = (
            self.csoil[self.timestep_ind] * (1.0 - self.dt * sres)
            + lit * self.dt * cvegnew
            + self.dt * (fcvs - fcsa)
        )

        # Save new values
        self.veg_box.stock = self.cveg[self.timestep_ind + 1] = cvegnew
        self.soil_box.stock = self.csoil[self.timestep_ind + 1] = csoilnew
        self.lit[self.timestep_ind + 1] = lit * cvegnew
        self.sres[self.timestep_ind + 1] = sres * csoilnew
        if npp_flag:
            self.npp[self.timestep_ind + 1] = npp
        else:
            self.gpp[self.timestep_ind + 1] = gpp
            self.vres[self.timestep_ind + 1] = vres * cvegnew
            self.npp[self.timestep_ind + 1] = (
                self.gpp[self.timestep_ind + 1] - self.vres[self.timestep_ind + 1]
            )

        self.fcva[self.timestep_ind + 1] = fcva
        self.fcsa[self.timestep_ind + 1] = fcsa
        self.fcvs[self.timestep_ind + 1] = fcvs

        self.old_t = temp_ano

        self.timestep_ind += 1

        return (
            cvegnew - self.cveg[self.timestep_ind - 1],
            csoilnew - self.csoil[self.timestep_ind - 1],
        )
