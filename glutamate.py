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

class GlutmateExocytosis(Reaction):
    species = [Glu]
    a1 = 5e-5
    a2 = 5e-6
    a3 = 850 # s^{-1}
    membrane = None

    def get_Internal_vars(self):
        pass

    def get_dot_Interval_vars(self):

        pass

    def flux(self,V_m=None,system_state = None, ICa = None, invalues = None, outvalues = None):
        Cai = invalues[Ca] if invalues is not None else self.membrane.inside.value(Ca,system_state)
        glu = invalues[Glu] if invalues is not None else self.membrane.inside.value(Glu,system_state)
        if ICa is None:
            # need to compute the calcium current. This kind of sucks!!
            pass
        CaiHill = np.power(Cai,4.0)
        KnHIll = power(20e-6,4)
        Prel = CaiHill/(CaiHill+KnHIll)
        return {Ca: Prel*glu*ICa}

class GlutamatePackaging(Reaction):
    species = [Glu]