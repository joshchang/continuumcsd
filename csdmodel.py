#!/usr/bin/env python
"""
author: Joshua C Chang

Import first our libraries
"""
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
    return { key: scalar*value for key,value in dictionary.items()}


class CSDModel(object):
    def __init__(self, N):
        self.N = N

        # Each compartment is associated with a volume fraction
        self.compartments = []
        self.volfrac = {} # \sum v_j = 1, for v_j <=1
        """
         Store the volume fractions of each of the compartments
        """

        # membranes
        self.membranes = []

        self.numdiffusers = 0
        self._N_internal_object = 0
        self._N_volumefraction = 0
        self.numpoisson = 0
        self.num_membrane_potentials = 0
        self.isAssembled = False

    def addCompartment(self, comp, fraction):

        self.compartments.extend([comp])
        self.volfrac[comp] = fraction

    def assembleSystem(self):
        """Assemble the FEM system. This is only run a single time before time-stepping. The values of the coefficient
        fields need to be updated between time-steps
        """
        # Loop through the entire model and composite the system of equations
        self.diffusors = []  # [[compartment, species, diffusivity of species],[ ...],[...]]
        """
           Diffusors have source terms
        """
        self.electrostatic_compartments = [] # (compartment) where electrostatic equations reside
        """
           Has source term
        """
        self.potentials = [] # (membrane)
        """
            No spatial derivatives, just construct ODE
        """
        self.channelvars = [] #(membrane, channel, ...)

        for compartment in self.compartments:
            s = 0
            for species in compartment.species:
                if compartment.diffusivities[species] < 1e-10: continue
                self.diffusors.extend([ [compartment,species,compartment.diffusivities[species]] ])
                s+=compartment.diffusivities[species]*abs(species.z)
            if s>0:
                self.electrostatic_compartments.extend([compartment])
                # Otherwise, there are no mobile charges in the compartment


        # the number of potentials is the number of spatial potentials + number of membrane potentials
        self.numdiffusers = len(self.diffusors)
        self.numpoisson = len(self.electrostatic_compartments)
        self.numcompartments = len(self.compartments)


        for membrane in self.membranes:
            self.potentials.extend([membrane])
            membrane.phi_m = np.ones(self.N)*membrane.phi_m
        self.num_membrane_potentials = len(self.potentials) # improve this

        """
            Assemble the equations for the system
            Order of equations:
            for compartment in compartments:
                for species in compartment.species
                    diffusion (in order)
                volume
                potential for electrodiffusion
            Membrane potentials


        Vectorize the values that aren't already vectorized
        """
        for compartment in self.compartments:
            for j,(species, val) in enumerate(compartment.values.items()):
                try:
                    length = len(val)
                    compartment.internalVars.extend([(species,self.N,j*self.N)])
                    compartment.species_internal_lookup[species] = j*self.N
                except:
                    compartment.values[species]= np.ones(self.N)*val
                    compartment.species_internal_lookup[species] = j*self.N

        """
        Set indices for the "internal variables"
        List of tuples
        (compartment/membrane, num of variables)

        Each compartment or membrane has method
        getInternalVars()
        get_dot_InternalVars(t,values)
        get_jacobian_InternalVars(t,values)
        setInternalVars(values)

        The internal variables for each object are stored starting in
        y[obj.system_state_offset]
        """
        self.internalVars = []
        index = 0
        for membrane in self.membranes:
            tmp = membrane.getInternalVars()
            if tmp is not None:
                self.internalVars.extend( [(membrane,len(tmp),index)] )
                membrane.system_state_offset = index # @TODO FIX!!
                index += len(tmp)
                membrane.N = self.N

        for membrane in self.membranes:
            for channel in membrane.channels:
                channel.N = self.N
                # Register internal variables for the channels involved
                channeltmp = channel.getInternalVars()
                if channeltmp is not None:
                    self.internalVars.extend([ (channel, len(channeltmp),index)])
                    channel.system_state_offset = index
                    channel.internalLength = len(channeltmp)
                    index+=len(channeltmp)

            # Register any reactions associated with the given membrane?
        """
        Compartments at the end, so we may reuse some computations
        """

        for compartment in self.compartments:
            index2 = 0
            compartment.system_state_offset = index  # offset for this object in the overall state
            for species, value in compartment.values.items():
                compartment.internalVars.extend([(species,len(compartment.value(species)),index2)])
                index2 += len(value)
            tmp = compartment.getInternalVars()
            self.internalVars.extend( [(compartment,len(tmp),index)] )
            index += len(tmp)
            compartment.N = self.N

        for compartment in self.compartments:
            for reaction in compartment.reactions:
                tmp = reaction.getInternalVars()
                if tmp is not None:
                    self.internalVars.extend([ (reaction, len(tmp),index)])
                    reaction.system_state_offset = index
                    reaction.internalLength = len(tmp)
                    index += len(tmp)

        for key, val in self.volfrac.items():
            self.volfrac[key] = val*np.ones(self.N)

        self._N_internal_object = sum ([item[1] for item in self.internalVars])
        self._N_volumefraction = (len(self.compartments)-1)*self.N

        """
        ODE integrator here. Add ability to customize the parameters in the future
        """
        self.t = 0.0
        self.odesolver = ode(self.ode_rhs) #
        self.odesolver.set_integrator('lsoda', nsteps=3000, first_step=1e-6, max_step=5e-3 )
        self.odesolver.set_initial_value(self.getInternalVars(),self.t)



        self.isAssembled = True

    def addMembrane(self,membrane):
        self.membranes.extend([membrane])
        if self.isAssembled:
            # need to register the internal variables for this membrane to the model!
            test = membrane.getInternalVars()

            pass

    def advance_one_timestep(self,dt,system_state=None):
        """
        Take result from computation and update the coefficient fields (sources, volume fractions) for dolfin, before next FEM step
        Solve the parabolic PDE
        Assume that all of the source functions are already updated, so we can just solve the system
        """
        self.dt.assign(dt)

        if system_state is not None:
            # set prev_value__ based on system_state
            pass

        self.prev_value__.vector()[:] = self.functions__.vector().array()  # FIX!!!! @TODO!!!!
        #fluxes = self.updateSources()
        self.pdesolver.solve()  # solves the diffusive compartments

        """
        for compartment, flux in fluxes.items():
            if not compartment.diffusive:
                for species, val in flux.items():
                    compartment.values[species]+=val*dt/self.volfrac[compartment]
            #else:
            #    print flux[K]*dt
        """

        newvals = self.functions__.vector().array()

        for j,(compartment,species,D) in enumerate(self.diffusors):
            # update these
            compartment.values[species] = np.array(newvals[self.dofs_is[j]])

        # loop through and update non-diffusive compartments

    def ode_rhs(self,t,system_state):
        """
        Take the native model variables and create a single vector

        return:
            ydot(numpy.ndarray)
            ordering: internal vars for each variable in self.internalVars
        """
        temp = np.zeros(self._N_internal_object+self._N_volumefraction)
        """
        Loop through once to do necessary pre-computation. This code relies on the
        membranes coming first in self.internalVars for the model. Otherwise, the
        fluxes will all be zero!
        """
        compartmentfluxes = {compartment:collections.Counter() for compartment in self.compartments}
        waterflows = {compartment:0.0 for compartment in self.compartments}
        # the fluxes are now zero
        for (key, length, index) in self.internalVars:
            if type(key) is Membrane:
                (temp[index:(index+length)] , flux, current) = key.get_dot_InternalVars(system_state,t)
                compartmentfluxes[key.outside].update(flux)
                compartmentfluxes[key.inside].update(scalar_mult_dict(flux,-1.0))
                wf = key.waterFlow(system_state)
                waterflows[key.outside] += wf
                waterflows[key.inside] -= wf
            elif type(key) is not Compartment and type(key) is not CellCompartment:
                temp[index:(index+length)] = key.get_dot_InternalVars(system_state,t)

        #for key, val in compartmentfluxes.items():
        #    compartmentfluxes[key] = scalar_mult_dict(compartmentfluxes[key] , 1.0/self.volumefraction(key,system_state))

        for (key, length, index) in self.internalVars:
            if type(key) is Compartment or type(key) is CellCompartment:
                temp[index:(index+length)] = key.get_dot_InternalVars(system_state,compartmentfluxes[key],t)

        #Also compute the fluxes and determine the changes in the concentrations

        for j in xrange(self.numcompartments-1):
            temp[(self._N_internal_object+j*self.N):(self._N_internal_object+(j+1)*self.N)] = waterflows[self.compartments[j]]

        return temp

    def getInternalVars(self):
        """
        Ordering: Concentrations: internal vars
        """
        temp = np.zeros(self._N_internal_object+self._N_volumefraction)
        for (key, length, index) in self.internalVars:
            temp[index:(index+length)] = key.getInternalVars()
        for j in xrange(self.numcompartments-1):
            temp[(self._N_internal_object+j*self.N):(self._N_internal_object+(j+1)*self.N)] = self.volumefraction(self.compartments[j])
        return temp

    def ode_jacobian(self,t,system_state):
        """
        TODO: PROGRAM A JACOBIAN
        Havign an explicit Jacobian would reduce the number of function evals needed
        as well as improve the situation wrt numerical instabilities
        """
        pass

    def volumefraction(self,compartment,system_state=None): #@TODO
        if system_state is not None:
            # compute the offset in the system_state vector for when volume fractions are stored
            # find out which compartment is indexed
            index = self.compartments.index(compartment)
            if index < len(self.compartments):
                pass
            else:
                pass
        return self.volfrac[compartment]

    def volumefractions(self,system_state=None):
        vfrac = {}
        if system_state is None:
            return self.volfrac
        for j in xrange(self.numcompartments-1):
            vfrac[self.compartments[j]] = system_state[(self._N_internal_object+j*self.N):(self._N_internal_object+(j+1)*self.N)]
        totalfrac = sum(vfrac.values(),axis = 0)
        vfrac[self.compartments[self.numcompartments-1]]= 1.0-totalfrac
        return vfrac


    def setInternalVars(self,system_state):
        for (key, length, index) in self.internalVars:
            key.setInternalVars(system_state[index:(index+length)])
        for j in xrange(self.numcompartments-1):
            self.volfrac[self.compartments[j]] = system_state[(self._N_internal_object+j*self.N):(self._N_internal_object+(j+1)*self.N)]
        return

    def __str__(self):
        return self.y

    def setBoundaryConditions(self,bcs):
        self.bcs = bcs

    def sanity(self,system_state=None):
        """
        Make sure the system is still physical
        All concentrations are non-negative
        All potentials are not NaN
        """
        return True

