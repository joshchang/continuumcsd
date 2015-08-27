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
    membrane = None

    def __init__(self,name,membrane,Nrel):
        super(GlutmateExocytosis, self).__init__(name, membrane)
        self.Nrel = Nrel

    def flux(self, V_m=None, system_state=None, invalues=None, outvalues=None):
        Cai = invalues[Ca] if invalues is not None else self.membrane.inside.value(Ca,system_state)
        glu = invalues[Glu] if invalues is not None else self.membrane.inside.value(Glu,system_state)
        CaiHill = np.power(Cai,4.0)
        KnHIll = power(20e-6,4)
        Prel = CaiHill/(CaiHill+KnHIll)
        return {Glu: Prel * glu * self.Nrel}

class GlutamatePackaging(Reaction):
    """
    Generate glutamate for packaging into vesicles
    """
    species = [Glu]