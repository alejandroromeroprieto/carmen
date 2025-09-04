"""
Class implementing the ocean component of the carbon cycle.

It is based on the scheme developed by Joos et al. (1996), where a carbonate
scheme simulates the dissolution of carbon from the atmosphere to the ocean
mixed layer, and an impulse-response function is used to approximate the carbon
export towards the deep ocean.
"""

import numpy as np

from carmen.constants import (
    GAS_EXCHANGE_COEF,
    OCEAN_AREA,
    PPM2GT,
)
from carmen import defaults
from carmen.ocean_component.utils import joos_response


class OceanCarbonCycle:
    """
    Class implementing the ocean component of the carbon cycle, based on
    the work of Joos et al. (1996).

    The class has the following attributes:
        - catm0: pre-industrial value for atmos CO2 conc. (ppm).
        - oflux: timeseries of carbon uptake by the ocean (positive into the ocean).
                 (Gtc/yr)
        - carbon_increase: cumulative carbon uptake by the ocean (positive into the
                           ocean). (GtC)
        - dt: timestep used for the general carbon cycle emulator. (year)
        - dt_ocean: timestep used for the ocean component of the carbon cycle emulator.
                    (year)
        - num_steps: number of steps the general carbon cycle emulator is expected to run.
                     (dimensionless)
        - docn: parameter representing pre-industrial mixing depth for CO2 uptake. (m)
        - ocntemp: parameter controlling the modulation of the ocean partial carbon
                   pressure on sea surface temperature. This is an addition to the base
                   Joos model. (1/K)
        - docntemp: parameter controlling the modulation of the ocean mixed layer depth on
                    sea surface temperature. This is an addition to the base Joos model
                    (1/K)
        - timestep_ind: index pointing at the current timestep.
        - n4occ: number of ocean steps per parent-model step.
        - rjoos: joos response function for all timesteps.
        - ocn_uptake: timeseries of carbon uptake by the ocean for each ocean timestep
                      (positive into the ocean) (Gtc/yr).


    The initialisation function takes the following arguments:
    - dt_model: timestep used for the general carbon cycle emulator. (year)
    - dt_ocean: timestep used for the ocean component of the carbon cycle emulator. (year)
    - num_steps: number of steps the general carbon cycle emulator is expected to run.
                 (dimensionless)
    - t0: initial year of the emulation.
    - kwargs: dictionary with parameter values for the emulator.

    The code was originally developed by Glen Harris, from the met office, with the
    following information:
        Based on Joos et al. (1996) as implemented in IMOGEN (Huntingford et al 2010).
        IMOGEN source code (Fortran) that was used to inform this is in
            /home/h04/hadgh/AM/decks_from_hadley/all_decks/analogue_decks/ANLG_DRIVER.dk
            /home/h04/hadgh/AM/decks_from_hadley/all_decks/analogue_decks/OCEAN_CO2.dk
        and other files here.

        Note that this implementation can sometimes be numerically unstable for large
        timesteps. IMOGEN gets around this by using using a smaller timestep. We follow
        the same approach, by specifying a timestep 'dtoccmax' outside the call to this
        object, and then calculating the number of smaller ocean timesteps 'n4occ'
        required for each main timestep. We find that dtocc=0.1
        years is generally stable. This is less than ideal, in that the ocean carbon cycle
        estimation is a bit slow.
    """

    def __init__(self, dt_model, dt_ocean, num_steps, t0, **kwargs):
        self.catm0 = kwargs.get("catm0", defaults.CATM0_DEFAULT)
        self.oflux = np.zeros(num_steps)
        self.carbon_increase = np.zeros(num_steps)
        self.dt = dt_model
        self.dt_ocean = dt_ocean
        self.num_steps = num_steps

        # Parameters
        self.docn = kwargs.get("docn", defaults.DOCN_DEFAULT)
        self.ocntemp = kwargs.get("ocntemp", defaults.OCNTEMP_DEFAULT)
        self.docntemp = kwargs.get("docntemp", defaults.DOCNTEMP_DEFAULT)

        # Current timestep index
        self.timestep_ind = 0

        # To avoid instabilities, you may want to use a smaller timestep for
        # the ocean component
        ocean_steps_per_model_steps = max([1, int(dt_model / dt_ocean)])
        self.n4occ = ocean_steps_per_model_steps
        dt4occ = dt_model / ocean_steps_per_model_steps

        # save the joos response function at all timesteps for later use
        timeocc = np.arange(t0, t0 + num_steps * dt4occ, dt4occ)
        self.rjoos = joos_response(timeocc)

        # Create an array to store ocean uptake values for each ocean step.
        self.ocn_uptake = np.zeros(len(timeocc))

    def molcon_to_ppm(self, dmol):
        """
        Calultates the ocean carbon partial pressure. Eq. 6b in Joos et al. (1996).

        dmol is the dissolved inorganic carbon. Units: micromol/Kg.
        """

        t0slab = 18.3  # temperature of mixed layer (assumed constant)
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
            dd = dmol / 100.0
            ppm = (
                dmol
                * 0.1
                * (a1 * 10.0 + dd * (a2 + dd * (-a3 + dd * (a4 - a5 * 0.1 * dd))))
            )
        return ppm

    def update(self, catm, catm1, dt_ocn):
        """
        Update ocean carbon cycle for one timestep.
        INPUT
        catm:   float, atmospheric CO2 conc (ppm) at current timestep.
        catm1:  float, atmospheric CO2 conc (ppm) at previous timestep.
        dt_ocn: float, global mean sea surface temperature change wrt pre-industrial
                (Kelvin/Celsius)

        OUTPUT
        dco2_ocn: carbon uptake by the ocean over the timestep (positive into ocean)
                  (GtC/year)
        """

        mol_units_converter = 1.722e17

        docntemp = self.docn * (1 / (1 + np.exp(self.docntemp * dt_ocn)) + 0.5)

        cmol = mol_units_converter / (docntemp * OCEAN_AREA)

        dtstep = self.dt / self.n4occ
        grad_catm = (catm - catm1) / self.dt

        # The rest of this function is as originally coded by Glen Harris
        for k0 in range(self.n4occ):
            total_uptake = 0  # Total uptake for this dtstep
            k = k0 + 1
            catmk = catm1 + grad_catm * dtstep * (k - 0.5)  # (k-0.5) midpoint of dtstep
            istep = (
                self.timestep_ind
            ) * self.n4occ + k  # istep is end of the occ prediction timestep

            # Warning, Python processing is much slower if one doesn't multiply cmol*
            # uptake INSIDE the loop below. Note that uptake here is typically very small:
            # O(6*10**-15), and cmol very large: O(6*10**16), so I suppose multiplying
            # keeps things closer to 1. This is still a bit unexpected (python 'feature'?)
            msum = 0.0
            # msum = np.sum(np.convolve(cmol*uptake[:istep], rjoos[istep:0:-1]))

            # much faster than the original
            msum = np.sum(cmol * self.ocn_uptake[:istep] * self.rjoos[istep:0:-1])

            msum = dtstep * msum
            psum = self.molcon_to_ppm(msum)  # psum has units of ppm
            # Net C0_2 flux

            cocn = (self.catm0 + psum) * np.exp(self.ocntemp * dt_ocn)  # original

            uptakenew = GAS_EXCHANGE_COEF * (catmk - cocn)
            self.ocn_uptake[istep] = uptakenew
            total_uptake += uptakenew * dtstep * PPM2GT

        self.timestep_ind += 1

        self.carbon_increase[self.timestep_ind] = (
            self.carbon_increase[self.timestep_ind - 1] + total_uptake
        )
        self.oflux[self.timestep_ind] = total_uptake / self.dt

        return total_uptake
