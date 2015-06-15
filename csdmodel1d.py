#!/usr/bin/env python
"""
author: Joshua C Chang

Import first our libraries
"""
from csdmodel import *
from compartment import *
from channels import *
from compartment import *
from reaction import *
from membrane import *
from species import *

"""
Import other libraries, including dolfin.
"""
from collections import Counter
import itertools
from scipy.integrate import ode


def scalar_mult_dict(dictionary, scalar):
    return {key: scalar * value for key, value in dictionary.items()}


class CSDModelInterval(CSDModel):
    def __init__(self, N, dx):
        # Each compartment is associated with a volume fraction
        self.compartments = []
        self.volfrac = {}  # \sum v_j = 1, for v_j <=1
        """
         Store the volume fractions of each of the compartments
        """

        # membranes
        self.membranes = []

        self.numdiffusers = 0
        self.numpoisson = 0
        self.N = N
        self.dx = dx
        self.x = np.arange(N) * dx + dx / 2.0
        self.num_membrane_potentials = 0
        self.isAssembled = False

    def assembleSystem(self):
        super(CSDModelInterval, self).assembleSystem()
        # 1D???

    def ode_rhs(self, t, system_state):
        """
        Take the native model variables and create a single vector
        Add diffusion for the diffusing variables!
        return:
            ydot(numpy.ndarray)
            ordering: internal vars for each variable in self.internalVars
        """

        temp = np.zeros(self.Nobject + self.Nvfrac)
        """
        Loop through once to do necessary pre-computation. This code relies on the
        membranes coming first in self.internalVars for the model. Otherwise, the
        fluxes will all be zero!
        """
        compartmentfluxes = {compartment: collections.Counter() for compartment in self.compartments}
        waterflows = {compartment: 0.0 for compartment in self.compartments}
        volumefractions = self.volumefractions(system_state)
        # the fluxes are now zero

        for (key, length, index) in self.internalVars:
            if type(key) is Membrane:
                (temp[index:(index + length)], flux, current) = key.get_dot_InternalVars(system_state, t)
                compartmentfluxes[key.outside].update(flux)
                compartmentfluxes[key.inside].update(scalar_mult_dict(flux, -1.0))
                wf = key.waterFlow(system_state)
                condition = (volumefractions[key.outside]<0.95)*(volumefractions[key.inside]<0.95) \
                    *(volumefractions[key.outside]>0.05)*(volumefractions[key.outside]>0.05)
                waterflows[key.outside] += wf*condition
                waterflows[key.inside] -= wf*condition
            elif type(key) is not Compartment and type(key) is not CellCompartment:
                temp[index:(index + length)] = key.get_dot_InternalVars(system_state, t)


        for (key, length, index) in self.internalVars:
            if type(key) is Compartment or type(key) is CellCompartment:
                temp[index:(index + length)] = key.get_dot_InternalVars(system_state, compartmentfluxes[key],volumefractions[key],0.0, t, dx = self.dx)

        # Also compute the fluxes and determine the changes in the concentrations

        for j in xrange(self.numcompartments - 1):
            temp[(self.Nobject + j * self.N):(self.Nobject + (j + 1) * self.N)] = waterflows[self.compartments[j]]

        return temp

