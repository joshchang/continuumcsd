'''
Glutamate exocytosis dependent on docking pool/reserve pool/intracellular calcium concentration
This is a membrane reaction
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


