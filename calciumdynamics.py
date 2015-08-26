from species import *
from reaction import *

"""
Calcium buffering is a compartment reaction

\dot( vB_free ) = -k_on v Ca B_free + k_off v (B_tot - B_free)
 \dot{B_free} = -k_on Ca B_free + k_off(B_tot-B_free) - B_free dot(v)
"""


class CaMbuffer(CompartmentReaction):
    compartment = None
    capacity = 1.0
    free = capacity

    def flux(self, system_state, invalues=None, volfrac=None, dotvolfrac=None):
        if invalues is None: invalues = self.compartment.get_val_dict(system_state)

    def get_dot_InternalVars(self, system_state=None, invalues=None, volfrac=None, dotvolfrac=None):
        if system_state is not None:
            free = system_state[self.system_state_offset:self.system_state_offset + self.N]
        else:
            free = self.free

        return -self.k_on * invalues[Ca] * volfrac * free + self.k_off * volfrac * (self.capacity - free)

    def get_InternalVars(self, system_state):
        if system_state is not None:
            return system_state[self.system_state_offset:self.system_state_offset + self.N]
        return self.free

    def __init__(self, name, compartment, capacity):
        super(CaMbuffer).__init__(name, compartment)
        self.capacity = capacity
