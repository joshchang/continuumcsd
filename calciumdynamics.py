from species import *
from reaction import *

"""
Calcium buffering is a compartment reaction

\dot( vB_free ) = -k_on v Ca B_free + k_off v (B_tot - B_free)
 \dot{B_free} = -k_on Ca B_free + k_off(B_tot-B_free) - B_free dot(v)

"""


class CaMbuffer(CompartmentReaction):
    compartment = None
    capacity = 1.0  # rest concentration of the total buffer (bound and unbound) at the original volume
    free = capacity  # free buffer concentration
    k_off = 1.0
    k_on = 1.0
    species = [Ca]

    def flux(self, system_state, invalues=None, volfrac=None, dotvolfrac=None):
        if invalues is None: invalues = self.compartment.get_val_dict(system_state)
        free = self.get_InternalVars(system_state)
        return volfrac * (invalues[Ca] * self.k_on - self.k_off * (self.capacity - free))

    def get_dot_InternalVars(self, system_state=None, invalues=None, volfrac=None, dotvolfrac=None):
        if system_state is not None:
            free = system_state[self.system_state_offset:self.system_state_offset + self.N]
        else:
            free = self.free

        return -self.k_on * invalues[Ca] * volfrac * free + self.k_off * volfrac * (
            self.capacity - free) - free * dotvolfrac, {
                   Ca: volfrac * (invalues[Ca] * self.k_on - self.k_off * (self.capacity - free))}

    def getInternalVars(self, system_state=None):
        if system_state is not None:
            return system_state[self.system_state_offset:self.system_state_offset + self.N]
        return self.free

    def equilibriate(self, system_state=None, invalues=None, volfrac=None, dotvolfrac=None):
        if invalues is None: invalues = self.compartment.get_val_dict(system_state)
        self.free = self.k_off * self.capacity / (self.k_off + self.k_on * invalues[Ca])
        pass

    def __init__(self, name, compartment, capacity):
        super(self.__class__, self).__init__(name, compartment)
        self.capacity = capacity

    def vectorizevalues(self):
        if ~hasattr(self.free, '__iter__'):
            self.free = np.ones(self.N) * self.free


class LavrentovichHemkin(CompartmentReaction):
    """
    Lavrentovich and Hemkin dynamics for intracellular calcium
    state space: X (Ca) Y (CaER) Z (IP3)
    vserca=vm2*(x^2/(x^2+k2^2))
    vplc=vp*(x^2/(x^2+kp^2))
    vcicr=4*vm3*((kcaa^n)*(x^n)/((x^n+kcaa^n)*(x^n+kcat^n)))*(z^m/(z^m+kip3^m))*(y-x)

    dx/dt=vin-kout*x+vcicr-vserca+kf*(y-x)
    dy/dt=vserca-vcicr-kf*(y-x)
    dz/dt=vplc-kdeg*z

    Keep IP3 as internal variable to track for now
    Internal vars: CaER, IP3
    External vars: internal Ca
    """

    CaER = 1.5e-6
    IP3 = 0.1e-6
    vm2 = 15e-6
    vm3 = 40e-6
    vin = 0.05e-6
    vp = 0.05e-6
    k2 = 0.1e-6
    kcaa = 0.15e-6
    kcat = 0.15e-6
    kip3 = 0.1e-6
    kp = 0.3e-6
    kdeg = 0.08  # /s
    kout = 0.5  # /s
    kf = 0.5  # /s
    n = 2.02
    m = 2.2
    species = [Ca]

    def getInternalVars(self, system_state=None):
        if system_state is not None:
            return system_state[self.system_state_offset:self.system_state_offset + 2 * self.N]
        return None

    def getCaER(self, system_state=None):
        if system_state is not None:
            return system_state[self.system_state_offset:self.system_state_offset + self.N]
        return self.CaER

    def getIP3(self, system_state=None):
        if system_state is not None:
            return system_state[self.system_state_offset + N:self.system_state_offset + 2 * self.N]
        return IP3

    def get_dot_InternalVars(self, system_state, invalues=None, volfrac=None, dotvolfrac=None):

        #     dx/dt=vin-kout*x+vcicr-vserca+kf*(y-x)
        # dy/dt=vserca-vcicr-kf*(y-x)
        # dz/dt=vplc-kdeg*z
        IP3 = self.getIP3(system_state)
        CaER = self.getCaER(system_state)
        Cai = invalues[Ca] if invalues is not None else self.compartment.get(Ca, system_state)

        vserca = self.vm2 * (1.0 / (1.0 + np.power(self.k2 / Cai, 2)))
        vplc = self.vp * (1.0 / (1.0 + power(self.kp / Cai)))
        vcicr = 4.0 * self.vm3 * (CaER - Cai) / (1.0 + power(self.kip3 / IP3, self.m)) / (
        1.0 + power(self.kcat / Cai, self.n)) / (1.0 + power(Cai / self.kcaa, self.n))

        return np.flatten([-self.kout * Cai + vcicr - vserca + self.kf * (CaER - Cai), vplc - self.kdeg * IP3]), \
               {Ca: -self.kout * Cai * vcicr - vserca + self.kf * (CaER - Cai)}

    def vectorizevalues(self):
        if ~hasattr(self.IP3, '__iter__'): self.IP3 = self.IP3 * np.ones(self.N)
        if ~hasattr(self.CaER, '__iter__'): self.CaER = self.CaER * np.ones(self.N)

    def flux(self, system_state, invalues=None, volfrac=None, dotvolfrac=None):
        IP3 = self.getIP3(system_state)
        CaER = self.getCaER(system_state)
        Cai = invalues[Ca] if invalues is not None else self.compartment.get(Ca, system_state)

        vserca = self.vm2 * (1.0 / (1.0 + np.power(self.k2 / Cai, 2)))
        vplc = self.vp * (1.0 / (1.0 + power(self.kp / Cai)))
        vcicr = 4.0 * self.vm3 * (CaER - Cai) / (1.0 + power(self.kip3 / IP3, self.m)) / (
            1.0 + power(self.kcat / Cai, self.n)) / (1.0 + power(Cai / self.kcaa, self.n))

        return {Ca: 0, IP3: 0}

    def equilibriate(self, system_state):
        IP3 = self.getIP3(system_state)
        CaER = self.getCaER(system_state)
