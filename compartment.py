__author__ = 'Josh Chang'
__email__ = 'joshchang@ucla.edu'

from numpy import sum, exp, sqrt, log
from params import *
from channels import *
from species import *
import collections
import operator
from collections import defaultdict
from customdict import *

class Compartment(object):
    """Defines a spatial compartment

    Note:
        All species in a compartment are intensive variables

    Attributes:
        name (String): Name of compartment
        species ([Species]): Species that are tracked in the compartment
        values ([double]): Corresponding values for each of the species in the compartment
        names ([String]): Names of the species in LaTeX
        diffusivities ([double]): Values in m^2/s

    """

    def __init__(self, name="", minvolume=0.05, maxvolume=0.9):
        self.name = name
        self.species = set()
        self.values = {}
        self.names = {}
        self.diffusivities = {}
        self.reactions = []
        self.diffusive = False
        self.porosity_adjustment = False
        self.internalVars = []
        self.system_state_offset = 0
        self.species_internal_lookup = {}
        self.N = 1
        self.minvolume = minvolume
        self.maxvolume = maxvolume
        self.onedimension = False
        self.density = 1 # normalization density for computing fluxes ecs is 1. otherwise, it is the number density
        self.initial_v_frac = 1.0

    def __str__(self):
        return self.name

    def addSpecies(self, species, value, diffusivity=None, name=None):
        """Add a given species to the compartment
        Args:
            species(Species): An ion species
            value(double): Value associated (concentration)
            diffusivity(double): Diffusivity at steady state for species in compartment
            name(String): Name of species in LaTeX

        """
        if species in self.species:
            self.values[species] += value
            return
        self.species.add(species)
        self.values[species] = value
        if diffusivity is None:
            self.diffusivities[species] = species.diffusivity
        else:
            self.diffusivities[species] = diffusivity
        if name is None:
            self.names[species] = species.name
        else:
            self.names[species] = name

        if self.diffusivities[species] > 0: self.diffusive = True

    def value(self, species, system_state=None):
        if system_state is None: return self.values[species]
        return system_state[self.system_state_offset \
                            + self.species_internal_lookup[species]:self.system_state_offset \
                                                                    + self.species_internal_lookup[species] + self.N]

    def value_matrix(self, species, system_state):
        return system_state[:, self.system_state_offset \
                               + self.species_internal_lookup[species]:self.system_state_offset \
                                                                       + self.species_internal_lookup[species] + self.N]

    def get_val_dict(self, system_state=None):
        """

        :rtype : dict
        """
        if system_state is None: return self.values
        # else, return corresponding entries in system_state
        return {species: system_state[self.system_state_offset \
                                      + self.species_internal_lookup[species]:self.system_state_offset \
                                                                              + self.species_internal_lookup[
                                                                                  species] + self.N] for species in
                self.species}

    def get_val_matrix_dict(self, system_state):
        """

        :rtype : dict
        """
        if system_state is None: return self.values
        # else, return corresponding entries in system_state
        return {species: system_state[:, self.system_state_offset \
                                         + self.species_internal_lookup[species]:self.system_state_offset \
                                                                                 + self.species_internal_lookup[
                                                                                     species] + self.N] for species in
                self.species}

    def setInternalVars(self, values):
        j = 0
        for key, val in self.values.iteritems():
            self.values[key] = values[j:(j + len(val))]
            j += len(val)
        pass

    def getInternalVars(self):
        # return np.array(self.values.values()).flatten()
        temp = np.zeros(sum([item[1] for item in self.internalVars]))
        for (key, length, index) in self.internalVars:
            if type(key) is Species:
                temp[index:index + length] = self.value(key)
            else:
                temp[index:index + length] = key.getInternalVars()

        return temp

    def get_dot_InternalVars(self, system_state, fluxes=customdict(float), invalues = None, volumefraction=1.0, dotvolumefraction=0.0, t=None, dx=0):
        """
        Expect a dictionary of fluxes into the compartment
        The fluxes are already adjusted for the volume of the compartment
        """
        temp = np.zeros(sum([item[1] for item in self.internalVars]))
        concentrations = self.get_val_dict(system_state) if invalues is None else invalues


        for (key, length, index) in self.internalVars:
            if type(key) is Species:
                concentration = concentrations[key]
                sourceterm = fluxes[key]
                volumeterm = dotvolumefraction * concentration
                diffusionterm = 0
                electrodiffusion = 0
                if self.onedimension and dx > 0 and self.diffusive:
                    # Add diffusion somehow...
                    dconc = np.zeros(self.N)
                    dconc[1:-1] = (concentration[2:] - concentration[:-2]) / dx
                    ddconc = np.zeros(self.N)
                    ddconc[1:-1] = (concentration[2:] - 2 * concentration[1:-1] + concentration[:-2]) / dx ** 2
                    ddconc[0] = dconc[1] / dx
                    ddconc[-1] = -dconc[-2] / dx
                    dvolumefraction = np.zeros(self.N)
                    dvolumefraction[1:-1] = (volumefraction[2:] - volumefraction[:-2]) / dx
                    dvolumefraction[0] = (volumefraction[1] - volumefraction[0]) / dx
                    dvolumefraction[-1] = (volumefraction[-1] - volumefraction[-2]) / dx
                    diffusionterm = self.diffusivities[key] * volumefraction * ddconc + self.diffusivities[
                                                                                            key] * dvolumefraction * dconc if not self.porosity_adjustment \
                        else self.diffusivities[key] * power(volumefraction, 2) / self.initial_v_frac * ddconc + 2 * \
                                                                                                                 self.diffusivities[
                                                                                                                     key] \
                                                                                                                 * volumefraction / self.initial_v_frac * dvolumefraction * dconc
                    phi = self.phi(system_state)
                    try:
                        """
                        dphi= np.gradient(phi,dx,edge_order=2)
                        ddphi = np.gradient(dphi,dx,edge_order=2)
                        """

                        dphi = np.zeros(self.N)
                        ddphi = np.zeros(self.N)
                        dphi[1:-1] = (phi[2:] - phi[:-2]) / dx
                        ddphi[1:-1] = (phi[2:] + 2 * phi[1:-1] - phi[:-2]) / dx ** 2
                        ddphi[0] = dphi[1] / dx
                        ddphi[-1] = -dphi[-1] / dx

                    except:
                        dphi = 0
                        ddphi = 0

                temp[index: (index + length)] = (
                                                sourceterm + diffusionterm + electrodiffusion - volumeterm) / volumefraction

            else:
                temp[(self.system_state_offset + index):(
                self.system_state_offset + index + length)] = key.get_dot_InternalVars(system_state)

        return temp

    def phi(self, system_state):
        return 0

    def setValue(self, species, value):
        self.values[species] = value

    def addReaction(self, reaction):
        reaction.compartment = self
        reaction.equilibriate()
        self.reactions.extend([reaction])

    def reactionFluxes(self):
        pass

    def printvalues(self):
        for species, value in self.values.iteritems():
            print("%s: %.3f" % (self.names[species], value))

    def tonicity(self, system_state=None, invalues=None):
        if invalues is not None:
            return np.sum(list(invalues.values()), axis=0)
        if system_state is None:
            return np.sum(list(self.values.values()), axis=0)
        else:
            return np.sum([self.value(species, system_state) for species in self.species], axis=0)

    def charge(self, system_state=None):
        if system_state is None:
            return np.sum([value * species.z for species, value in self.values.iteritems()], axis=0)
        else:
            return np.sum([self.value(species, system_state) * species.z for species in self.species], axis=0)

    def balanceCharge(self):
        """
        Add nonspecific cation and anion species to current compartmnet in order to
        balance both the tonicity and the charge
        """
        charge = self.charge()
        if charge > 1e-20:
            self.addSpecies(Anion, -charge / Anion.z, 0, "Anion")
        elif charge < -1e-20:
            self.addSpecies(Cation, -charge / Anion.z, 0, "Cation")

    def balanceWith(self, compartment):
        # Add anion and cation species to this compartment to achieve electroneutrality in this compartment
        # and isotonicity with outer compartment
        charge_deficit = np.float64(compartment.charge() - self.charge())
        tonicity_deficit = np.float64(compartment.tonicity() - self.tonicity())
        if tonicity_deficit < 0.0:
            print("Tonicity is too high in this compartment already to do anything")
            print("Add some particles to " + compartment.name)
            return
        if abs(charge_deficit) > 1e-20:
            self.addSpecies(Cation, np.float64((charge_deficit + tonicity_deficit) * 0.5), 0, "Cation")
            self.addSpecies(Anion, np.float64((tonicity_deficit - charge_deficit) * 0.5), 0, "Anion")
        elif tonicity_deficit > 1e-20:
            # Add in equal proportions to balance things
            self.addSpecies(Anion, np.float64(-tonicity_deficit * 0.5), 0, "Anion")
            self.addSpecies(Cation, np.float64(-tonicity_deficit * 0.5), 0, "Cation")


class CellCompartment(Compartment):
    """ Cellular compartment
    Args:
        density (float): number of cells per unit volume (length in 1-d, area in 2-d)
            units are SI
        name (String): name of compartment
    """

    def __init__(self, name, density):
        super(self.__class__, self).__init__(name)
        self.density = density  # number of these compartments per unit volume (SI units)

    def volume(self, system_state):
        '''

        :param system_state: The entire system state as a numpy vector
        :return: the volume of this cell compartment on a per-cell basis
        '''
        return 0
