#!/usr/bin/env python

"""
author: Joshua C Chang
email: joshchang@ucla.edu

"""

from numpy import power, exp, log, sqrt, sum
from params import *
from species import *
import numpy as np
from customdict import *
import math


def scalar_mult_dict(dictionary, scalar):
    return {key: scalar * value for key, value in dictionary.iteritems()}


class Channel(object):
    system_state_offset = 0
    N = 1
    name = "Generic Channel"  # overwrite this
    """Generic ion channel class (also emcompasses pumps)
        Args:
            name (String): name of channel
            species ([Species]): species that are possibly permeable
            gmax ([float]): maximum conductance for each species (for single channel)
    """

    def current(self, system_state=None, V_m = None,invalues = None, outvalues = None):
        """
            Return the current as a dict with ion: current pairs
            If V_m is defined, use it as the potential. Otherwise, use
            V_m()
        """
        return

    def water_permeability(self, system_state=None):
        """
        Water permeability of channel as a function of V_m
        Units of permeability are L/Molarity/s for each single channel
        """
        if self.N > 1:
            return np.zeros(self.N)
        return 0.0

    def current_infty(self, system_state=None):
        """ This is the current if the opening/closing is able to immediately adjust to the environment
        Override this method for GHK channels

        """
        return self.current(system_state=system_state)

    def equilibriate(self):
        """
        Equilibriate the internal state
        """
        print(str(self) + " single-channel rest-current: " + str(self.current()) + "A")
        return

    def getInternalVars(self): return None

    def setInternalVars(self, system_state): return None

    def vectorizevalues(self):
        pass

    def __str__(self):
        return self.name


class GHKChannel(Channel):
    m = 0.0
    h = 1.0
    p = 0
    q = 0
    max_permeability = None
    quasi_steady = False

    def __init__(self, quasi_steady=False):
        self.quasi_steady = quasi_steady

    def current(self, system_state=None, V_m = None, invalues=None, outvalues=None): #@TODO: FIX FOR non-standard channels
        if V_m is None: V_m = self.membrane.phi(system_state)
        if invalues is None: invalues = self.membrane.inside.get_val_dict(system_state)
        if outvalues is None: outvalues = self.membrane.outside.get_val_dict(system_state)

        permeability = self.permeability(system_state, V_m, invalues,
                                         outvalues) if not self.quasi_steady else self.permeability_infty(system_state,
                                                                                                          V_m, invalues,
                                                                                                          outvalues)

        # For safety, but SLOW!!!
        if not hasattr(V_m, '__iter__'):
            # V_m is not iterable, use ternary operator

            I = [permeability[species]  * F * species.z * species.z * V_m / phi \
                 * ((invalues[species] - exp(-V_m * species.z / phi) * outvalues[species]) / (
                1.0 - exp(-V_m * species.z / phi)) \
                        if V_m * species.z > 0 else \
                        (invalues[species] * exp(V_m * species.z / phi) - outvalues[species]) / (
                            exp(V_m * species.z / phi) - 1.0)) \
                 for species in self.species]
        else:

            I = [
                permeability[species]* F * species.z * species.z * V_m / phi * ((invalues[species] - exp(-V_m * species.z / phi) * outvalues[species]) / (
                    1.0 - exp(-V_m * species.z / phi)) if species.z < 0 else \
                                     (invalues[species] * exp(V_m * species.z / phi) - outvalues[species]) / (
                                         exp(V_m * species.z / phi) - 1.0)) \
                for species in self.species
                ]
        currents = customdict(float)
        for ion, current in zip(self.species, I):
            currents[ion] = current
        return currents

    def get_h(self, system_state=None):
        """
        m^p h^q
        :param system_state: Vector of the system state
        :return: numpy array of h value or ones
        """
        if self.q == 0:
            return np.ones(self.N)
        if system_state is None or self.quasi_steady:
            V_m = self.membrane.phi(system_state)
            return self.hinfty(V_m)
        if self.p == 0:
            return system_state[self.system_state_offset:self.system_state_offset + self.N]
        else:
            return system_state[self.system_state_offset + self.N:self.system_state_offset + 2 * self.N]

    def get_m(self, system_state=None):
        """


        :rtype : np.array
        :param system_state: numpy array of the system state
        :return:
        """
        if self.p == 0:
            return np.ones(self.N)
        if system_state is None or self.quasi_steady:
            V_m = self.membrane.phi(system_state)
            return self.minfty(V_m)
        return system_state[self.system_state_offset:self.system_state_offset + self.N]

    def permeability(self, system_state=None, V_m = None, invalues=None, outvalues=None):
        if self.quasi_steady:
            return self.permeability_infty(system_state, V_m, invalues, outvalues)
        if V_m is None: V_m = self.membrane.phi(system_state)
        h = self.get_h(system_state) if self.q>0 else 1.0
        m = self.get_m(system_state) if self.p>0 else 1.0
        gate = power(np.abs(m), self.p) * power(np.abs(h), self.q)
        return {ion: permeability * gate for permeability, ion in zip(self.max_permeability, self.species)}

    def permeability_infty(self, system_state=None, V_m = None, invalues=None, outvalues=None):
        if V_m is None: V_m = self.membrane.phi(system_state)

        alpham = self.alpham(V_m)
        betam = self.betam(V_m)
        alphah = self.alphah(V_m)
        betah = self.betah(V_m)
        m_infty = alpham / (alpham + betam)
        h_infty = alphah / (alphah + betah)
        gate = power(m_infty, self.p) * power(h_infty, self.q)
        return {ion: permeability * gate for permeability, ion in zip(self.max_permeability, self.species)}

    def current_infty(self, system_state=None, V_m = None, invalues = None, outvalues = None):
        """
        This is the current when the gates have equilibriated to V_m, and the
        given intra- and extra- cellular concentrations
        invalues and outvalues are dicts taken species are arguments
        """
        if V_m is None: V_m = self.membrane.phi(system_state)
        if invalues is None: invalues = self.membrane.inside.get_val_dict(system_state)
        if outvalues is None: outvalues = self.membrane.outside.get_val_dict(system_state)

        permeability = self.permeability_infty(system_state, V_m)
        # Maybe let's not compute this multiple times, once for each channel!
        if not hasattr(V_m, '__iter__'):
            # V_m is not iterable, use ternary operator
            I = [ permeability[species] * F * species.z * species.z * V_m / phi \
                 * ((invalues[species] - exp(-V_m * species.z / phi) * outvalues[species]) / (1.0 - exp(-V_m * species.z / phi)) \
                        if V_m * species.z > 0 else \
                        (invalues[species] * exp(V_m * species.z / phi) - outvalues[species]) / (exp(V_m * species.z / phi) - 1.0)) \
                 for species in self.species]
        else:
            I = [
                permeability[species] * F * species.z * species.z * V_m / phi \
                    * ((invalues[species] - exp(-V_m * species.z / phi) * outvalues[species]) / (
                    1.0 - exp(-V_m * species.z / phi)) if species.z < 0 else \
                                     (invalues[species] * exp(V_m * species.z / phi) - outvalues[species]) / (
                                         exp(V_m * species.z / phi) - 1.0)) \
                for species in self.species
                ]

        currents = customdict(float)
        for ion, current in zip(self.species, I):
            currents[ion] = current
        return currents

    def mdot(self, V_m=None, m=None):
        """ Compute dm/dt
        """

        if V_m is None: V_m = self.membrane.phi()
        if m is None: m = self.m
        return self.alpham(V_m) * (1.0 - m) - self.betam(V_m) * m

    def hdot(self, V_m=None, h=None):
        """ Compute dh/dt
        """
        if V_m is None: V_m = self.membrane.phi()
        if h is None: h = self.h

        return self.alphah(V_m) * (1.0 - h) - self.betah(V_m) * h

    def minfty(self, V_m=None):
        if V_m is None: V_m = self.membrane.phi()
        am = self.alpham(V_m)
        bm = self.betam(V_m)
        return am / (am + bm)

    def hinfty(self, V_m=None):
        if V_m is None: V_m = self.membrane.phi()
        ah = self.alphah(V_m)
        bh = self.betah(V_m)
        return ah / (ah + bh)

    def equilibriate(self, V_m=None):
        if V_m is None: V_m = self.membrane.phi()
        if not self.p == 0:
            self.m = self.minfty(V_m)
        else:
            self.m = 1.0
        if not self.q == 0:
            self.h = self.hinfty(V_m)
        else:
            self.h = 1.0
        print(str(self) + " single-channel rest-current: " + str(self.current()) + "A")

    def alpham(self, system_state=None):
        """
        Over-ride this method
        """
        return 1.0

    def betam(self, system_state=None):
        """
        Over-ride this method
        """
        return 1.0

    def alphah(self, system_state=None):
        return 1.0

    def betah(self, system_state=None):
        return 1.0

    def equilibriate_gates(self, system_state=None):
        V_m = self.membrane.phi(system_state)
        self.m = self.minfy(V_m)
        self.h = self.hinfty(V_m)

    def vectorizevalues(self):
        if not hasattr(self.m, '__iter__') and self.m is not None:
            self.m = np.ones(self.N) * self.m
        if not hasattr(self.h, '__iter__') and self.h is not None:
            self.h = np.ones(self.N) * self.h

    # Internal vars for the standard GHK equation are the gating variables
    # [p, q]
    # expect that system_state offset is set??
    def getInternalVars(self, system_state=None):
        if self.quasi_steady: return None
        if self.p == 0 and self.q == 0:
            return None
        elif self.q == 0:
            return self.get_m(system_state)
        elif self.q == 0:
            return self.get_h(system_state)
        return np.array([self.get_m(system_state), self.get_h(system_state)]).flatten()

    def setInternalVars(self, system_state):
        if self.quasi_steady: return
        if self.p == 0:
            self.h = self.get_h(system_state)
        elif self.q == 0:
            self.m = self.get_m(system_state)
        else:
            self.m = self.get_m(system_state)
            self.h = self.get_h(system_state)

    def get_dot_InternalVars(self, system_state, t, V_m = None):
        if self.quasi_steady: return None
        if self.p == 0 and self.q == 0:
            return None
        elif self.p == 0:
            temp = self.hdot(self.membrane.phi(system_state), self.get_h(system_state))
        elif self.q == 0:
            temp = self.mdot(self.membrane.phi(system_state), self.get_m(system_state))
        else:
            temp = np.zeros(2 * self.N)
            temp[:self.membrane.N] = self.mdot(self.membrane.phi(system_state), self.get_m(system_state))
            temp[self.membrane.N:] = self.hdot(self.membrane.phi(system_state), self.get_h(system_state))
        return temp


class LeakChannel(GHKChannel):
    p = 0
    q = 0
    species = []
    max_permeability = []
    def __init__(self, species):
        self.species = [species]
        self.max_permeability = [1.0]

    def permeability(self, system_state=None, V_m = None, invalues=None, outvalues=None):
        return {self.species[0] : np.ones(self.N)*self.max_permeability[0] }

    def permeability_infty(self, system_state=None, V_m = None, invalues=None, outvalues=None):
        return self.permeability(system_state=system_state, V_m = V_m, invalues=invalues, outvalues=outvalues)

    def set_permeability(self, permeability):
        self.max_permeability = [permeability]