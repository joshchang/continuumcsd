'''
Glutamate exocytosis dependent on docking pool/reserve pool/intracellular calcium concentration
This is a membrane "channel"
'''

from species import *
from channels import *
from reaction import *
from compartment import *
from membrane import *
import numpy as np

class GlutmateExocytosis(MembraneReaction):
    species = [Glu]
    a1 = 5e-5
    a2 = 5e-6
    a3 = 850 # s^{-1}
    P_rel0 = 0.0
    membrane = None

    def __init__(self,name,membrane,Nrel):
        super(self.__class__, self).__init__(name, membrane)
        self.Nrel = Nrel

    def flux(self, system_state, volfrac=None, dotvolfrac=None, invalues=None, outvalues=None):
        """
        Flux out per cell
        """
        Cai = invalues[Ca] if invalues is not None else self.membrane.inside.value(Ca,system_state)
        glu = invalues[Glu] if invalues is not None else self.membrane.inside.value(Glu,system_state)
        Prel = 1.0 / (1.0 + np.power(20e-6 / Cai, 4.0))  # - self.P_rel0
        return {Glu: Prel * glu * self.Nrel}

    def equilibriate(self):
        Cai = self.membrane.inside.value(Ca)
        self.P_rel0 = 1.0 / (1.0 + np.power(20e-6 / Cai, 4.0))

    def getInternalVars(self):
        return None


class GlutamatePackaging(CompartmentReaction):
    """
    Generate glutamate for packaging into vesicles
    """
    species = [Glu]

class GlutamateDecay(CompartmentReaction):
    species = [Glu]

    def __int__(self, name, compartment):
        self.name = name
        self.compartment = compartment

    def flux(self, system_state, volfrac=None, dotvolfrac=None, invalues=None):
        glu = self.compartment.value(Glu, system_state) if invalues is None else invalues[Glu]
        return {Glu: -glu * 10.0}

    def equilibriate(self):
        pass


class GlutamateReuptake(MembraneReaction):
    species = [Glu]

    def __init__(self, name, membrane, Nrel):
        super(self.__class__, self).__init__(name, membrane)
        self.Nrel = Nrel

    def get_InternalVars(self, system_state):
        return None

    def flux(self, system_state, volfrac=None, dotvolfrac=None, invalues=None):
        pass

    def equilibriate(self):
        return