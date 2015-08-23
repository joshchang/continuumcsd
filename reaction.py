from channels import *
from compartment import *
import collections
import operator
import numpy as np

class Reaction(object):
    """
    Define a reaction. A reaction can have any number of
    internal state variables. Reactions are associated with
    compartments, and species within compartments.
    get_dot_InternalVars() is integrated along with the rest of the system


    Reaction rate parameters are per-cell.
    """
    species = ()
    system_state_offset = 0
    N = 1
    def __init__(self,name):
        pass
    def flux(self,system_state = None):
        pass

class MembraneReaction(Reaction):
    """
    Membrane reactions can modify concentrations on both sides of the membrane
    These do not contributed to the charging of the membrane. The fluxes are conservative, meaning
    that a change on one side of the membrane is balanced by an equal and opposite change on the other side
    of a membrane. Compartment reactions on the other hand need-not be conservative. The flux computed
    is per unit cell.
    """
    membrane = None
    def __init__(self,name,membrane):
        self.membrane = membrane
        self.name = name
    def flux(self,system_state=None, currents = None):
        pass

class CompartmentReaction(Reaction):
    """
    These are reactions that occur within a single compartment. They need not be conservative. The reactions take
    the form
    d(v*X/dt) = J, to account for volume variations in the reactions
    The flux computed is per unit cell.
    """
    compartment = None
    def __init__(self,name,membrane):
        self.membrane = membrane
        self.name = name
    def flux(self,system_state=None, invalues = None):
        pass

class Buffer(Reaction):
    """
    Buffer reactions species + free buffer <--> buffered species
    Internal variable is the amount of free buffer
    """
    species = [Ca]
    def __init__(self,species, capacity, kon, koff):
        self.capacity = capacity # concentration capacity (per cell)
        self.kon = kon
        self.koff = koff
        self.species.add(species)
        self.compartment = None

    def equilibriate(self):
        self.state = self.capacity-self.kon/self.koff*self.compartment.value(self.species)

    def flux(self,system_state):
        return self.kon*self.compartment.value(self.species)-self.koff*(self.capacity-self.state)

    def getInternalVars(self,system_state=None):
        if system_state is None:
            return self.state
        return system_state[self.system_state_offset:self.system_state_offset+self.N]

    def setInternalVars(self,system_state):
        self.state = system_state[self.system_state_offset:self.system_state_offset+self.N]

    def get_dot_InternalVars(self,system_state):
        values = self.getInternalVars(system_state)
        return self.kon*self.compartment.value(self.species)-self.koff*(self.capacity-values)

class CaBuffer(Reaction):
    species = ()
    def __init__(self,capacity,kon,koff):
        self.kon = kon
        self.koff = koff
        self.capacity = capacity
        self.species.add(Ca)
