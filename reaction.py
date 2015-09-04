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

    def vectorizevalues(self):
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

    def flux(self, system_state, volfractions=None, invalues=None, outvalues=None):
        """
        """
        pass

class CompartmentReaction(Reaction):
    """
    These are reactions that occur within a single compartment. They need not be conservative. The reactions take
    the form
    d(v*X/dt) = J, to account for volume variations in the reactions
    The flux computed is per unit cell.
    """
    compartment = None
    conservative = True  # if conservative, rates and concentrations vary with volume changes in the compartment

    def __init__(self, name, compartment):
        self.compartment = compartment
        self.name = name

    def flux(self, system_state, volfraction=None, dotvolfraction=None, invalues=None):
        """

        :param system_state:
        :param volfraction:  numpy array
        :param dotvolfraction:
        :param invalues:
        :return:
        """
        pass

    def get_freeCapacity(self, system_state):
        pass

    def get_InternalVars(self, system_state):
        pass

    def get_dot_InternalVars(self,system_state):
        pass
