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

        membranecurrents = customdict(float)
        membranefluxes = customdict(float)

        for (key, length, index) in self.internalVars:
            if type(key) is Membrane:
                (temp[index:(index + length)], flux, current) = key.get_dot_InternalVars(system_state, t) # output ydot, ion flux, current
                membranecurrents[key] = current
                membranefluxes[key] = flux

                inside_t = key.inside.tonicity(system_state, invalues = concentrations[key.inside])
                outside_t = key.outside.tonicity(system_state, invalues = concentrations[key.outside])

                compartmentfluxes[key.outside].update(scalar_mult_dict(flux,key.inside.density*key.outside.density)) # The flux is per cell
                compartmentfluxes[key.inside].update(scalar_mult_dict(flux,-key.inside.density*key.outside.density)) # inside updates on a per-cell basis
                wf = key.waterFlow(system_state, tonicity = outside_t-inside_t)*key.inside.density*key.outside.density # per-cell waterflow times compartment densities
                '''
                Compute the water flows here. If compartment is over maxvolume, don't let it get bigger
                If a compartment is below minvolume don't let it get smaller
                '''
                out_small = 1-(volumefractions[key.outside]<=key.outside.minvolume) #outside too small
                out_big = 1-(volumefractions[key.outside]>=key.outside.maxvolume) #outside too big
                in_small = 1-(volumefractions[key.inside]<=key.inside.minvolume) # inside too small
                in_big = 1-(volumefractions[key.inside]>=key.inside.maxvolume) # inside too big

                have_flow = out_small*out_big*in_small*in_big # need all conditions to be satisfied

                waterflows[key.outside] += wf*have_flow
                waterflows[key.inside] -= wf*have_flow
                '''
                Settle for now on the cludgy solution of putting all reactions into model.internalvars
                '''
            elif issubclass(type(key), Reaction):
                if issubclass(type(key), MembraneReaction):
                    flux = customdict(float)
                    if length > 0:
                        ydot, flux = key.get_dot_InternalVars(system_state,
                                                              invalues=concentrations[key.membrane.inside],
                                                              outvalues=concentrations[key.membrane.outside])
                        temp[index:(index + length)] = ydot
                    else:
                        flux = key.flux(system_state, invalues=concentrations[key.membrane.inside],
                                        outvalues=concentrations[key.membrane.outside])
                    compartmentfluxes[key.membrane.outside].update(
                        scalar_mult_dict(flux, key.membrane.inside.density * key.membrane.outside.density))
                    compartmentfluxes[key.membrane.inside].update(
                        scalar_mult_dict(flux, -key.membrane.inside.density * key.membrane.outside.density))

                elif issubclass(type(key), CompartmentReaction):

                    if length > 0:
                        ydot, flux = key.get_dot_InternalVars(system_state,
                                                              volfraction=volumefractions[key.compartment],
                                                              dotvolfraction=waterflows[key.compartment],
                                                              invalues=concentrations[key.compartment])
                        temp[index:(index + length)] = ydot
                    else:
                        flux = key.flux(system_state)

                    compartmentfluxes[key.compartment].update(scalar_mult_dict(flux, key.compartment.density))


            elif not issubclass(type(key), Compartment):
                try:
                    temp[index:(index + length)] = key.get_dot_InternalVars(system_state, t)
                except:
                    print(key.name)

        # Also compute the fluxes and determine the changes in the concentrations

        """
        Volume shifts

        """
        for j in xrange(self.numcompartments - 1):
            temp[(self._N_internal_object + j * self.N):(self._N_internal_object + (j + 1) * self.N)] = waterflows[
                self.compartments[j]]


        for (key, length, index) in self.internalVars:
            if issubclass(type(key), Compartment):
                temp[index:(index + length)] = key.get_dot_InternalVars(system_state=system_state, invalues = concentrations[key] \
                                                                        , fluxes=compartmentfluxes[key],
                                                                        volumefraction=volumefractions[key],
                                                                        dotvolumefraction=waterflows[key], \
                                                                        t=t, dx = self.dx)
            elif type(key) is CellCompartment:
                print "WTF"



        if debug: return temp, compartmentfluxes, waterflows
        return temp

