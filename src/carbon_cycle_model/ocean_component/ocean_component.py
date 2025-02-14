"""
Class implementing the ocean component of the carbon cycle.

It is based on the scheme developed by Joos et al (1996) where a carbonate
scheme simulates the dissolution of carbon from the atmosphere to the ocean
mixed layer, and an Impulse-response function is used to approximate the carbon
export towards the deep ocean
"""

import math
import numpy as np

from carbon_cycle_model.constants import GAS_EXCHANGE_COEF, OCEAN_AREA, PPM2GT
from carbon_cycle_model import defaults


class OceanCarbonCycle:
    """
    Class implementing the ocean component of the carbon cycle, based on
    the work of Joos et al. (1996).

    The class has the following attributes:
        catm0:   pre-industrial value for atmos CO2 conc. Units: ppm.
        docn:    initial mixing depth for CO2 uptake. Calibrated parameter. Units: m.
        docnfac: temperature dependence for change in mixing depth for CO2 uptake.
                 Calibrated parameter. Units: dimensionless.
        ocntemp: temperature calibration parameter. Units: 1/K.

    This was originally developed by Glen Harris, at the met office, who had the
    following information:
        Based on Joos et al. (1996) as implemented in IMOGEN (Huntingford et al 2010).
        IMOGEN source code (Fortran) that was used to inform this is in
            /home/h04/hadgh/AM/decks_from_hadley/all_decks/analogue_decks/ANLG_DRIVER.dk
            /home/h04/hadgh/AM/decks_from_hadley/all_decks/analogue_decks/OCEAN_CO2.dk
        and other files here.

    Note that this implementation can sometimes be numerically unstable for large
    timesteps. IMOGEN gets around this by using using a smaller timestep. We follow
    the same approach, by specifying a timestep 'dtoccmax' outside the call to this
    object, and then calculating the number of smaller ocean timesteps 'n4occ' required
    for each main timestep. This is input to the Update method. We find that dtocc=0.1
    years is generally stable. This is less than ideal, in that the ocean carbon cycle
    estimation is a bit slow.
    """

    def __init__(self, dt, num_steps, **kwargs):
        self.catm0 = kwargs.get("catm0", defaults.CATM0_DEFAULT)
        self.carbon_increase = np.zeros(num_steps)
        self.dt = dt
        self.num_steps = num_steps

        # Parameters
        self.docn = kwargs.get("docn", defaults.DOCN_DEFAULT)
        self.docnfac = kwargs.get("docnfac", defaults.DOCNFAC_DEFAULT)
        self.ocntemp = kwargs.get("ocntemp", defaults.OCNTEMP_DEFAULT)
        self.docntemp = kwargs.get("docntemp", defaults.DOCNTEMP_DEFAULT)

        # Current timestep index
        self.timestep_ind = 0

    def molcon_to_ppm(self, dmol):
        """
        Calultates the ocean carbon partial pressure. Eq. 6b in Joos et al. (1996).

        dmol is the dissolved inorganic carbon. Units: micromol/Kg.
        """

        t0slab = 18.0  # temperature of mixed later (assumed constant)
        dcutoff = 292.6039793
        pcutoff = 1319.9999999117306
        derivcut = 12.01493683
        a1 = 1.5568 - (1.39930 * 0.01 * t0slab)
        a2 = 7.4706 - (0.20207 * t0slab)
        a3 = 1.2748 - (0.12015 * t0slab)
        a4 = 2.4491 - (0.12639 * t0slab)
        a5 = 1.5468 - (0.15326 * t0slab)
        if dmol >= dcutoff:
            ppm = pcutoff + derivcut * (dmol - dcutoff)
        else:
            # ppm = dmol*0.1*(a1*10. + a2*dd - a3*math.pow(dd,2)+ a4*math.pow(dd,3) -
            #       a5*0.1*math.pow(dd,4) )
            # This gives the same but is a bit faster
            dd = dmol / 100.0
            ppm = (
                dmol
                * 0.1
                * (a1 * 10.0 + dd * (a2 + dd * (-a3 + dd * (a4 - a5 * 0.1 * dd))))
            )
        return ppm

    def update(self, catm, catm1, im1, uptake, rjoos, d_t, n4occ):
        """
        Update ocean carbon cycle for one timestep.
        INPUT
        dt:     float, timestep, Units: year
        catm:   float, atmospheric CO2 conc (ppm) at current timestep.
        catm1:  float, atmospheric CO2 conc (ppm) at previous timestep.
        im1:    integer, index of previous timestep.
        uptake: np.array, ocean CO2 uptake for all times (on shorter ocean timeteps).
        rjoos:  Joos response function for all times.
        d_t:  float, global mean temperature change wrt pre-industrial (degC)
        n4occ:  integer, number of ocean timesteps for each model timestep.

        OUTPUT
        dco2_ocn:  change in ocean carbon over the timestep
        uptakenew: flux of carbon to ocean at end of timestep

        The input/output array uptake is also updated with new values for the ocean
        carbon flux for all ocean timesteps in the current main timestep.
        uptake is re-used in succeeding call to the Update method.
        """

        mol_units_converter = 1.722e17

        # following Oscar (gasser 2017)
        docntemp = self.docn * (
            1.0
            + max(-0.5, min(1.0, self.docnfac * (math.exp(self.docntemp * d_t) - 1)))
        )
        cmol = mol_units_converter / docntemp

        dtstep = self.dt / n4occ
        grad_catm = (catm - catm1) / self.dt
        dco2_ocn = 0.0

        # This rest of this function is as taken by Glen Harris code
        for k0 in range(n4occ):
            k = k0 + 1
            catmk = catm1 + grad_catm * dtstep * (k - 0.5)  # (k-0.5) midpoint of dtstep
            istep = im1 * n4occ + k  # istep is end of the occ prediction timestep

            # Warning, Python processing is much slower if one doesn't multiply cmol*
            # uptake INSIDE the loop below. Note that uptake here is typically very small:
            # O(6*10**-15), and cmol very large: O(6*10**16), so I suppose multiplying
            # keeps things closer to 1. This is still a bit unexpected (python 'feature'?)
            msum = 0.0
            # msum = np.sum(np.convolve(cmol*uptake[:istep], rjoos[istep:0:-1]))

            # much faster than the original
            msum = np.sum(cmol * uptake[:istep] * rjoos[istep:0:-1])

            msum = dtstep * msum
            psum = self.molcon_to_ppm(msum)  # psum has units of ppm
            # ! Net C0_2 flux
            # ! I guess that we take catm0 as in pre-industrial the partial pressures
            # ! would be the same
            # ! due to the equilibrium assumption.
            cocn = (self.catm0 + psum) * math.exp(self.ocntemp * d_t)

            uptakenew = (GAS_EXCHANGE_COEF / OCEAN_AREA) * (catmk - cocn)
            uptake[istep] = uptakenew
            dco2_ocn = dco2_ocn + uptakenew * OCEAN_AREA * dtstep * PPM2GT

        self.timestep_ind += 1

        self.carbon_increase[self.timestep_ind] = (
            self.carbon_increase[self.timestep_ind - 1] + dco2_ocn
        )
        return (dco2_ocn, uptakenew)
