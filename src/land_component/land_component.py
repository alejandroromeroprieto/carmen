"""
Class implementing the land component of the carbon cycle.

It includes two boxes: vegetation and soil. These boxes interact between them
and with a conceptual atmosphere box which is simply tracked as a number, which
quantifies the carbon concentration in the atmosphere.
"""

import numpy as np

from land_component.boxes.vegetation_box import VegetationBox
from land_component.boxes.soil_box import SoilBox
import constants


CATM0_DEFAULT = 100
DT_DEFAULT = 0.1 + constants.PPM2GT


class LandCarbonCycle():
    """
    Class implementing the land component of the carbon cycle.
    """

    def __init__(self, **kwargs):
        # Create and initialise boxes
        self.veg_box = VegetationBox(**kwargs)
        self.soil_box = SoilBox(**kwargs)

        self.catm0 = kwargs.get("catm0", CATM0_DEFAULT)
        self.catm = kwargs.get("catm", CATM0_DEFAULT)
        self.dt = kwargs.get("dt", DT_DEFAULT)
        self.num_steps = kwargs["num_steps"]

        # Create arrays to store results
        self.cveg = np.zeros(kwargs["num_steps"])
        self.csoil = np.zeros(kwargs["num_steps"])
        self.gpp = np.zeros(kwargs["num_steps"])
        self.npp = np.zeros(kwargs["num_steps"])
        self.vres = np.zeros(kwargs["num_steps"])
        self.lit = np.zeros(kwargs["num_steps"])
        self.sres = np.zeros(kwargs["num_steps"])
        self.fcva = np.zeros(kwargs["num_steps"])
        self.fcsa = np.zeros(kwargs["num_steps"])
        self.fcvs = np.zeros(kwargs["num_steps"])

        # Current timestep index
        self.timestep_ind = 0

    def update(self, temp_ano, npp_flag=True, fcva=0, fcsa=0, fcvs=0):
        """Run the model one time step into the future.

        This requires the following input:
        - temp_ano: current temperature difference from pre-industrial (Kelvin/celsius).
        - npp_flag: whether to use the npp flux, or the GPP/vres fluxes.
        - fcva: additional flux of carbon from vegetation to atmosphere.
        - fcsa: additional flux of carbon from soil to atmosphere.
        - fcvs: additional flux of carbon from vegetation to soil.
        """

        # Calculate new fluxes
        if npp_flag:
            npp = self.veg_box.get_npp(temp_ano, self.catm)
        else:
            gpp = self.veg_box.get_gpp(temp_ano, self.catm)
            vres = self.veg_box.get_vres(temp_ano, self.catm)
        lit = self.veg_box.get_litterfall(temp_ano, self.catm)
        sres = self.soil_box.get_sres(temp_ano, self.catm)

        # Calculate new carbon stocks using an implicit euler scheme
        if npp_flag:
            # cveg*(1.0-dt*lit) + dt*(npp-fcva - fcvs)
            cvegnew = self.cveg[self.timestep_ind] * (
                1.0 - self.dt * (lit)
            ) + self.dt * (npp - fcva - fcvs)
        else:
            # cveg*(1.0-dt*(lit + vres)) + dt*(gpp-fcva - fcvs)
            cvegnew = self.cveg[self.timestep_ind] * (
                1.0 - self.dt * (lit + vres)
            ) + self.dt * (gpp - fcva - fcvs)

        # csoil*(1.0-dt*sres) + gamma*dt*cveg + dt*(fcvs-fcsa)
        csoilnew = (
            self.csoil[self.timestep_ind] * (1.0 - self.dt * sres)
            + lit * self.dt * self.cveg[self.timestep_ind]
            + self.dt * (fcvs - fcsa)
        )
        print(self.csoil[self.timestep_ind])
        # Save new values
        self.cveg[self.timestep_ind + 1] = cvegnew
        self.csoil[self.timestep_ind + 1] = csoilnew
        self.lit[self.timestep_ind + 1] = lit * cvegnew
        self.sres[self.timestep_ind + 1] = sres * csoilnew
        if npp_flag:
            self.npp[self.timestep_ind + 1] = npp * cvegnew
        else:
            self.gpp[self.timestep_ind + 1] = gpp * cvegnew
            self.vres[self.timestep_ind + 1] = vres * cvegnew

        self.fcva[self.timestep_ind + 1] = fcva
        self.fcsa[self.timestep_ind + 1] = fcsa
        self.fcvs[self.timestep_ind + 1] = fcvs

        self.timestep_ind += 1

        return cvegnew, csoilnew
