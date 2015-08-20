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
from customdict import *
import itertools
from scipy.integrate import ode


def scalar_mult_dict(dictionary, scalar):
    return {key: scalar * value for key, value in dictionary.iteritems()}


class CSDModelInterval(CSDModel):

    def __init__(self, N, dx):
        """

        :param N: number of grid points
        :param dx: spacing
        :return:
        """
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
        for compartment in self.compartments:
            compartment.onedimension = True
        # 1D???

    def ode_rhs(self, t, system_state, debug = False):
        """
        Take the native model variables and create a single vector
        Add diffusion for the diffusing variables!
        return:
            ydot(numpy.ndarray)
            ordering: internal vars for each variable in self.internalVars

        """
        temp = np.zeros(self._N_internal_object + self._N_volumefraction)
        """
        Loop through once to do necessary pre-computation. This code relies on the
        membranes coming first in self.internalVars for the model. Otherwise, the
        fluxes will all be zero!
        """
        compartmentfluxes = {compartment: customdict(float) for compartment in self.compartments}
        waterflows = {compartment: 0.0 for compartment in self.compartments}
        volumefractions = self.volumefractions(system_state=system_state)
        # the fluxes are now zero

        concentrations = {compartment: compartment.get_val_dict(system_state) for compartment in self.compartments}

        for (key, length, index) in self.internalVars:
            if type(key) is Membrane:
                (temp[index:(index + length)], flux, current) = key.get_dot_InternalVars(system_state, t) # output ydot, ion flux, current
                inside_t = key.inside.tonicity(system_state, invalues = concentrations[key.inside])
                outside_t = key.outside.tonicity(system_state, invalues = concentrations[key.outside])

                compartmentfluxes[key.outside].update(flux)
                compartmentfluxes[key.inside].update(scalar_mult_dict(flux, -1.0))
                wf = key.waterFlow(system_state, tonicity = inside_t - outside_t)
                # No waterflow into compartment if it is too full
                # No waterflow out of compartment if it is too empty
                condition1 = (volumefractions[key.outside]<=key.outside.minvolume)
                condition2 = (volumefractions[key.outside]>=key.outside.maxvolume)
                condition3 = (volumefractions[key.inside]<=key.inside.minvolume)
                condition4 = (volumefractions[key.inside]>=key.inside.maxvolume)
                condition = wf > 0
                waterflows[key.outside] += condition1*condition4*condition*wf + condition2*condition3*(1-condition)*wf
                waterflows[key.inside] -= condition1*condition4*condition*wf - condition1*condition3*(1-condition)*wf

            elif type(key) is Reaction:
                flux = key.flux(system_state)
                compartmentfluxes[key.compartment].update(flux)

            elif not issubclass(type(key), Compartment):
                try:
                    temp[index:(index + length)] = key.get_dot_InternalVars(system_state, t)
                except:
                    print(key.name)


        for (key, length, index) in self.internalVars:
            if issubclass(type(key), Compartment):
                temp[index:(index + length)] = key.get_dot_InternalVars(system_state=system_state, invalues = concentrations[key] \
                        , fluxes=compartmentfluxes[key],volumefraction=volumefractions[key],dotvolumefraction=0.0,\
                        t=t, dx = self.dx)
            elif type(key) is CellCompartment:
                print "WTF"

        # Also compute the fluxes and determine the changes in the concentrations

        """
        Volume shifts

        """
        for j in xrange(self.numcompartments - 1):
            temp[(self._N_internal_object + j * self.N):(self._N_internal_object + (j + 1) * self.N)] = waterflows[self.compartments[j]]

        if debug: return temp, compartmentfluxes
        return temp

