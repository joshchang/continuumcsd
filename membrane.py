from channels import *
from compartment import *
from reaction import *
from species import *
import collections
import warnings
import operator
import numpy as np

def scalar_mult_dict(dictionary, scalar):
    return { key: scalar*value for key,value in dictionary.items()}

def dictmult(dict1,dict2):
    return {key: value*dict2[key] for key,value in dict1.items()}

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
        """
        V_m = self.phi(system_state)
        invalues = self.inside.get_val_dict(system_state)
        outvalues = self.outside.get_val_dict(system_state)

        currents = self.currents(system_state=system_state)
        fluxes = {species: current/F/species.z for species, current in currents.items() }

        totalcurrents = np.sum(currents.values(),axis = 0)

        # Channels need internal vars too!
        return -totalcurrents/self.Cm, fluxes, currents

    def get_jacobian_InternalVars(self,system_state = None,t = None):
        pass


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
                #print residual
                gmax = -residual/(self.phi()-self.phi_ion(channel.species))/density
                if channel.gmax<0:
                    print "Not adding leak because it doesn't balance anything. Try adding a pump first"
                    return
                channel.set_gmax(gmax)

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
           Add leak channels for each of the ion species that go through the channel in order to achieve balance
        """
        currents = self.currents()  # these are the residual currents to balance
        for species, residual in currents.items():
            if abs(self.phi()-self.phi_ion(species))<1e-20:
                gmax = 0.0
            else: gmax = -residual/(self.phi()-self.phi_ion(species))
            if gmax<0:
                print "Adding a leak channel does nothing to balance " + str(species)
                continue
            else:
                print ("Ion: %s, g_leak: %8.2E" %(str(species),gmax))
                channel = LeakChannel(species)
                channel.membrane = self
                channel.set_gmax(gmax)
                self.channeldensity[channel] = 1.0
                self.channels.extend([channel])

    def currents(self, system_state = None):
        """ Compute the instantaneous total currents through the membrane

        Value:
            dict species: current
        """
        V_m = self.phi(system_state)
        
        # Compute first ion-specific terms for GHK

        # Compute ghkcurrents
        '''
        # SUPER SLOW!!!
        if hasattr(V_m,'__iter__'):
            
            ghkcurrents = { species: F*V_m*species.z/phi*np.fromiter( [ (cin-exp(-vm*species.z/phi)*cout)/(1.0-exp(-vm*species.z/phi)) \
                if vm*species.z>0 else  \
                (cin*exp(vm*species.z/phi)-cout)/(exp(vm*species.z/phi)-1.0)
                for (vm,cin,cout) in zip(V_m,self.inside.value(species,system_state),self.outside.value(species,system_state)) ], np.float64)\
                for species in self.species}
            #ghkcurrents = {species: F*V_m*species.z/phi*(cin*exp(vm*species.z/phi)-cout)/(exp(vm*species.z/phi)-1.0) } \
        else:
            ghkcurrents = {species: F*V_m*species.z/phi*(self.inside.value(species,system_state)*exp(V_m*species.z/phi)-self.outside.value(species,system_state))/(exp(V_m*species.z/phi)-1.0) \
                for species in self.species}
        
        ghkcurrents = {species: F*V_m*species.z/phi*(self.inside.value(species,system_state)*exp(V_m*species.z/phi)-self.outside.value(species,system_state))/(exp(V_m*species.z/phi)-1.0) \
            for species in self.species}
        
        '''
        # Test numpy ternary expression instead, maybe it can be pretty fast??
        
        ghkcurrents = {species: np.where( V_m*species.z<0, F*V_m*species.z/phi*(self.inside.value(species,system_state)*exp(V_m*species.z/phi)-self.outside.value(species,system_state))/(exp(V_m*species.z/phi)-1.0), 
            F*V_m*species.z/phi*(self.inside.value(species,system_state)-self.outside.value(species,system_state)*exp(-V_m*species.z/phi))/(1.0-exp(-V_m*species.z/phi))) \
            for species in self.species}
        
        channelcurrents = []
        for channel in self.channels:
            if issubclass(type(channel),GHKChannel):
                channelcurrents.append( dictmult(channel.conductance(system_state), scalar_mult_dict(ghkcurrents,self.channeldensity[channel])))
            else:
                channelcurrents.append(scalar_mult_dict(channel.current( system_state),self.channeldensity[channel]))
        # GHK
        '''
        # Slower than the for loop
        channelcurrents = [dictmult(channel.conductance(system_state), scalar_mult_dict(ghkcurrents,self.channeldensity[channel])) \
            if issubclass(type(channel),GHKChannel) else scalar_mult_dict(channel.current( system_state),self.channeldensity[channel]) \
            for channel in self.channels]
        '''
        
        
        counter = collections.Counter()
        
        for d in channelcurrents:
            counter.update(d)
        return counter

    def fluxes(self, system_state = None):
        """ Compute the instantaneous fluxes through the membrane
        # These fluxes are per total volume and not yet adjusted for volume fraction
        """
        V_m = self.phi(system_state)
        invalues = self.inside.get_val_dict(system_state)
        outvalues = self.outside.get_val_dict(system_state)
        
        currents = self.currents(system_state=system_state)
        fluxes = {key: current/F/key.z*self.inside.density for key, current in currents.items() }
        return fluxes

    def waterFlow(self,system_state = None):
        """
        Compute rate of water flow through the channel in units L/s
        """

        inside_t = self.inside.tonicity(system_state)
        outside_t = self.outside.tonicity(system_state)

        totalpermeability = sum([channel.water_permeability(system_state)*self.channeldensity[channel] for channel in self.channels])
        return totalpermeability*(inside_t-outside_t)  # flow from in to out

    def currents_and_fluxes(self, system_state = None):
        """
        call this to compute the source terms and phidot
        invalues and outvalues are dicts
        """

        if system_state is not None:
            V_m = self.phi(system_state)
            invalues = self.inside.get_val_dict(system_state)
            outvalues = self.outside.get_val_dict(system_state)

        currents = self.currents(V_m=V_m, system_state=system_state)
        fluxes = {key: current/F/key.z for key, current in currents.items() }
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

    def set_phi(self,phi_m): self.phi_m = phi_m

    def phi_ion(self, species, system_state=None):
        Ce = self.outside.value(species,system_state)
        Ci = self.inside.value(species,system_state)
        return phi/species.z*(np.log(Ce)-np.log(Ci))
