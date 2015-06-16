from channels import *
from compartment import *
import collections
import operator
import numpy as np

class Reaction(object):
    """
    Definte a reaction. A reaction can have any number of
    internal state variables. Reactions are associated with
    compartments, and species within compartments.

    Reaction rate parameters are per-cell.
    """
    species = ()
    innervars = 0
    state = None
    def __init__(self,name):
        pass
    def flux(self):
        pass

class Buffer(Reaction):
    """
    Buffer reactions species + free buffer <--> buffered species
    Internal variable is the amount of free buffer
    """
    species = set()
    def __init__(self,species, capacity, kon, koff):
        self.capacity = capacity # concentration capacity
        self.kon = kon
        self.koff = koff
        self.species.add(species)

    def equilibriate(self):
        self.state = self.capacity-self.kon/self.koff*self.compartment.value(self.species)

    def flux(self):
        return self.kon*self.compartment.value(self.species)-self.koff*(self.capacity-self.state)

    def getInternalVars(self):
        return self.state

    def setInternalVars(self,system_state):
        self.state = system_state

    def get_dot_InternalVars(self,values):
        return self.kon*self.compartment.value(self.species)-self.koff*(self.capacity-values)

class CaBuffer(Reaction):
    species = ()
    def __init__(self,capacity,kon,koff):
        self.kon = kon
        self.koff = koff
        self.capacity = capacity
        self.species.add(Ca)
