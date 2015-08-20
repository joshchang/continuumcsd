from channels import *
from compartment import *
from reaction import *
from species import *
from collections import defaultdict
from customdict import *
from itertools import chain
import warnings
import operator
import numpy as np

def scalar_mult_dict(dictionary, scalar):
    """

    :type scalar: scalar number
    """
    return { key: scalar*value for key,value in dictionary.iteritems()}

def dictmult(dict1,dict2):
    return {key: value*dict2[key] for key,value in dict1.iteritems()}

class Coupling(object):
    """
    Define a general coupling between two compartments
    This may be used for instance to couple neurons with dendrites
    """
    def __init__(self,name,inside,outside):
        self.name = name
        self.inside = inside
        self.outside = outside


class Membrane(Coupling):
    """ Cell membrane separating two compartments
        All parameters are for a single cell. Current gives the current per cell
        Surface area is per cell, capacitance is per cell, etc
    Args:
        name (String): Name for membrane
        inside (Compartment): Compartment inside the membrane
        outside (Compartment): Compartment outside the membrane
        Cm (double): Capacitance of the membrane in Farads
        S (double): surface area in m^2
        phi0 (double): Resting potential difference across membrane (In-Out) in Volts
        water_permeable (double): total permability

    """
    def __init__(self,name,inside,outside,Cm,phi_m):
        self.name = name
        self.inside = inside
        self.outside = outside
        self.compartments = [self.inside,self.outside] # pointer for convenience
        self.channels = []
        self.channeldensity = {}
        self.reactions = []
        self.species = set()
        self.Cm = Cm
        self.phi_m = phi_m
        self.internalVars = []
        self.N = 1

    def setInternalVars(self,system_state):
        """
        InternalVars are the membrane potentials, and the ion variables
        """
        self.phi_m = system_state[:self.N]
        index = len(self.phi_m)
        for key, val in self.internalVars:
            key.setInternalVars(system_state[index:(index+len(val))])
            index += len(val)

    def getInternalVars(self,system_state = None):
        temp = []
        temp.extend(self.phi_m) # add the membrane potential first
        for key, value in self.internalVars:
           temp.extend(key.getInternalVars())
        return np.array(temp)

    # Currents go as phidot
    def get_dot_InternalVars(self,system_state = None,t = None):
        """
        InternalVars are the membrane potentials, and the ion variables
        Return dot currents, single cell fluxes, single cell currents
        """
        V_m = self.phi(system_state)
        invalues = self.inside.get_val_dict(system_state)
        outvalues = self.outside.get_val_dict(system_state)

        currents = self.currents(system_state=system_state, V_m = V_m, invalues = invalues, outvalues = outvalues)
        fluxes = {species: current/F/species.z for species, current in currents.iteritems() }

        totalcurrents = np.sum(currents.values(),axis = 0)

        # Channels need internal vars too!
        return -totalcurrents/self.Cm, fluxes, currents

    def get_jacobian_InternalVars(self,system_state = None,t = None):
        """
        :param system_state:
        :param t:
        :return: Jacobian matrix, rectangular matrix
        """
        pass

    def addReaction(self,reaction):
        reaction.membrane = self
        reaction.equilibriate()
        self.reactions.extend([reaction])

    def addChannel(self,channel,density):
        """ Add a single channel to the membrane
        Args:
            channel (Channel): The channel to add
            density (np.ndarray(self.N)): the number of channels per cell
        """

        if type(channel) is LeakChannel:
            # make this channel set steady system_state
            if channel.species in self.species:
                currents = self.currents()
                residual = currents[channel.species]
                gmax = -residual/(self.phi()-self.phi_ion(channel.species))
                if channel.gmax<0:
                    print("Not adding leak because it doesn't balance anything. Try adding a pump first")
                    return
                normalcurrent = channel.current()
                permability = -residual/normalcurrent[channel.species]
                channel.set_permeability(permability)

        channel.membrane = self
        channel.equilibriate()
        self.channels.extend([channel])
        try:
                self.species = self.species.union(channel.species) # if list
        except:
                self.species.add(channel.species) # if atom
        self.channeldensity[channel]=density
        # if channel is a leak channel, find its parameters by equilibriation


    def equilibriate(self):
        for channel in self.channels:
            channel.equilibriate()
        pass

    def addLeakChannels(self):
        """
           Find permeabilities for GHK leak currents
        """
        currents = self.currents()  # these are the residual currents to balance
        for species, residual in currents.iteritems():
            if abs(self.phi()-self.phi_ion(species))<1e-20:
                continue
            gmax = -residual/(self.phi()-self.phi_ion(species))
            if gmax<0:
                print("Adding a leak channel does nothing to balance " + str(species))
                continue

            channel = LeakChannel(species)
            channel.membrane = self

            normalcurrent = channel.current()
            permability = -residual/normalcurrent[species]

            channel.set_permeability(permability)
            self.channeldensity[channel] = 1.0
            self.channels.extend([channel])
            print ("Ion: %s, P_leak: %8.2E" %(str(species),permability))

    def currents(self, system_state = None, V_m = None, invalues = None, outvalues = None):
        """ Compute the instantaneous total currents through the membrane for single cells

        Value:
            dict species: current
        """
        if V_m is None: V_m = self.phi(system_state)
        if invalues is None: invalues = self.inside.get_val_dict(system_state)
        if outvalues is None: outvalues = self.outside.get_val_dict(system_state)

        # Compute first ion-specific terms for GHK

        # Compute ghkcurrents
        ghkcurrents = {}
        try:
            ghkcurrents = {species: np.where( V_m*species.z<0, F*V_m*species.z**2/phi*(invalues[species]*exp(V_m*species.z/phi)-outvalues[species])/(exp(V_m*species.z/phi)-1.0),
                F*V_m*species.z**2/phi*(invalues[species]-outvalues[species]*exp(-V_m*species.z/phi))/(1.0-exp(-V_m*species.z/phi))) \
                for species in self.species}
        except:
            pass

        counter = customdict(float)

        for channel in self.channels:
            if issubclass(type(channel),GHKChannel):
                counter.update( dictmult(channel.permeability(system_state=system_state, V_m = V_m,invalues = invalues, outvalues = outvalues), scalar_mult_dict(ghkcurrents,self.channeldensity[channel])))
            else:
                counter.update(scalar_mult_dict(channel.current(system_state=system_state, V_m = V_m,invalues = invalues, outvalues = outvalues),self.channeldensity[channel]))

        return counter


    def fluxes(self, system_state = None):
        """ Compute the instantaneous fluxes through the membrane
        # These fluxes are per total volume and not yet adjusted for volume fraction
        """
        currents = self.currents(system_state=system_state)
        fluxes = {key: current/F/key.z*self.inside.density for key, current in currents.iteritems() }
        return fluxes

    def waterFlow(self,system_state = None, V_m = None, invalues = None, outvalues = None, tonicity = None):
        """
        Compute rate of water flow through the channel in units L/s
        """

        totalpermeability = sum([channel.water_permeability(system_state)*self.channeldensity[channel] for channel in self.channels])

        if tonicity is not None: return totalpermeability*tonicity

        inside_t = self.inside.tonicity(system_state, invalues)
        outside_t = self.outside.tonicity(system_state, outvalues)

        return totalpermeability*(inside_t-outside_t)  # flow from in to out

    def currents_and_fluxes(self, system_state = None):
        """
        call this to compute the source terms and phidot
        invalues and outvalues are dicts
        """

        if system_state is not None:
            V_m = self.phi(system_state)

        currents = self.currents(V_m=V_m, system_state=system_state)
        fluxes = {key: current/F/key.z*self.inside.density for key, current in currents.iteritems() }
        return currents, fluxes

    def removeChannel(self,channel):
        """
          Use this to remove a channel after temporarily removing it
        """
        self.channels.remove(channel)
        self.channeldensity.pop(channel,None)

    def phi(self,system_state=None):
        if system_state is None: return self.phi_m;
        return system_state[self.system_state_offset:self.system_state_offset+self.N]

    def phi_matrix(self,system_state):
        return system_state[:,self.system_state_offset:self.system_state_offset+self.N]

    def set_phi(self,phi_m): self.phi_m = phi_m

    def phi_ion(self, species, system_state=None):
        Ce = self.outside.value(species,system_state)
        Ci = self.inside.value(species,system_state)
        return phi/species.z*(np.log(Ce)-np.log(Ci))
