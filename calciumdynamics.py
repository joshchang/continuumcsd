from species import *
from reaction import *

"""
Calcium buffering is a compartment reaction
"""


class CaMbuffer(CompartmentReaction):
    compartment = None
    capacity = 1.0

    def flux(self, system_state, invalues=None):
        if invalues is None: invalues = self.compartment.get_val_dict(system_state)

    def __init__(self, name, compartment, capacity):
        super(CaMbuffer).__init__(name, compartment)
        self.capacity = capacity
        pass
