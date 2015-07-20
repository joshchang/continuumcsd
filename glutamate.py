'''
Glutamate exocytosis dependent on docking pool/reserve pool/intracellular calcium concentration
'''

from species import *
from channels import *
from reaction import *
from compartment import *
from membrane import *


class GlutmateExocytosis(Channel):
    species = [Glu]
    a1 = 5e-5
    a2 = 5e-6
    a3 = 850 # s^{-1}
