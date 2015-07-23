#!/usr/bin/env python

"""
author: Joshua C Chang
email: joshchang@ucla.edu

"""


from numpy import power, exp, log, sqrt, sum
from params import *
from species import *
import numpy as np
import math

def scalar_mult_dict(dictionary, scalar):
    return { key: scalar*value for key,value in dictionary.items()}

class Channel(object):
    system_state_offset = 0
    N = 1
    """Generic ion channel class (also emcompasses pumps)
        Args:
            name (String): name of channel
            species ([Species]): species that are possibly permeable
            gmax ([float]): maximum conductance for each species (for single channel)
    """
    def current(self, system_state=None):
        """
            Return the current as a dict with ion: current pairs
            If V_m is defined, use it as the potential. Otherwise, use
            V_m()
        """
        return

    def water_permeability(self, system_state=None):
        """
        Water permeability of channel as a function of V_m
        Units of permeability are L/Molarity/s for each single channel
        """
        if self.N>1:
            return np.zeros(self.N)
        return 0.0

    def current_infty(self,system_state = None):
        """ This is the current if the opening/closing is able to immediately adjust to the environment
        Override this method for GHK channels

        """
        return self.current(system_state = system_state)

    def equilibriate(self):
        """
        Equilibriate the internal state
        """
        return

    def getInternalVars(self): return None

    def setInternalVars(self,system_state): return None

    def vectorizevalues(self):
        pass


class GHKChannel(Channel):
    m = 0.0
    h = 1.0
    def current(self, system_state = None, invalues=None, outvalues=None):
        V_m = self.membrane.phi(system_state)
        h = self.get_h(system_state)
        m = self.get_m(system_state)
        if invalues is None: invalues = self.membrane.inside.get_val_dict(system_state)
        if outvalues is None: outvalues = self.membrane.outside.get_val_dict(system_state)
        print("calling GHKcurrent, want GHK conductance call instead!!")
        gating = power(m,self.p)*power(h,self.q)*F*species.z*V_m/phi

        #For safety, but SLOW!!!
        if not hasattr(V_m, '__iter__'):
            # V_m is not iterable, use ternary operator


            I = [ gmax* gating\
                *( (invalues[species] - exp(-V_m*species.z/phi)*outvalues[species])/(1.0-exp(-V_m*species.z/phi)) \
                if V_m*species.z >0 else \
                (invalues[species]*exp(V_m*species.z/phi) -outvalues[species])/(exp(V_m*species.z/phi)-1.0 ) )\
                for gmax,species in zip(self.gmax,self.species)]
        else:
            """
            I = [ gmax*gating \
                *np.fromiter( [ (invalues[species]-exp(-vm*species.z/phi)*outvalues[species])/(1.0-exp(-vm*species.z/phi)) \
                    if vm*species.z>0 else  \
                    (invalues[species]*exp(vm*species.z/phi)-outvalues[species])/(exp(vm*species.z/phi)-1.0)
                    for vm in V_m], np.float64)
                for gmax,species in zip(self.gmax,self.species)]
            """
            insidestuff = vm*species.z/phi
            condition = (insidestuff>0)
            I = [
                gmax*gating*((invalues[species]-exp(-vm*species.z/phi)*outvalues[species])/(1.0-exp(-vm*species.z/phi)) if species.z<0 else \
                    (invalues[species]*exp(vm*species.z/phi)-outvalues[species])/(exp(vm*species.z/phi)-1.0)) \
                    for gmax, species in zip(self.gmax, self.species)
            ]



        return {ion:current for ion,current in zip(self.species,I)}

    def get_h(self, system_state=None):
        """
        m^p h^q
        :param system_state: Vector of the system state
        :return: numpy array of h value or ones
        """
        if self.q == 0:
            return np.ones(self.N)
        if system_state is None:
            V_m = self.membrane.phi(system_state)
            return self.hinfty(V_m)
        if self.p == 0:
            return system_state[self.system_state_offset:self.system_state_offset+self.N]
        else:
            return system_state[self.system_state_offset+self.N:self.system_state_offset+2*self.N]

    def get_m(self, system_state = None):
        """

        :param system_state: numpy array of the system state
        :return:
        """
        if self.p == 0:
            return np.ones(self.N)
        if system_state is None:
            V_m = self.membrane.phi(system_state)
            return self.minfty(V_m)
        return system_state[self.system_state_offset:self.system_state_offset+self.N]

    def conductance(self, system_state = None, invalues=None, outvalues=None):
        V_m = self.membrane.phi(system_state)
        h = self.get_h(system_state)
        m = self.get_m(system_state)
        #if invalues is None: invalues = self.membrane.inside.get_val_dict(system_state)
        #if outvlaues is None: outvalues = self.membrane.outside.get_val_dict(system_state)
        gate = power(m,self.p)*power(h,self.q)
        return {ion: gmax*gate for gmax,ion in zip(self.gmax,self.species)}

    def dIdV(self, system_state=None):
        """
        For constructing a Jacobian. This is the change in the total current
        through the membrane
        """
        V_m = self.membrane.phi(system_state)
        h = self.h(system_state)
        m = self.m(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)

        dIdV = [ gmax*species.z*(V_m*outvalues[species]*species.z*(exp(V_m*species.z/phi) - 1.0) + \
            V_m*species.z*(outvalues[species] - invalues[species]*exp(V_m*species.z/phi)) - \
            phi*(outvalues[species] - invalues[species]*exp(V_m*species.z/phi))*(exp(V_m*species.z/phi) - 1.0)) \
            /(phi**2*power(exp(V_m*species.z/phi) - 1.0,2))
            for gmax, species in zip(self.gmax,self.species)]
        return power(m,p)*power(h,q)*F*sum(dIdV)

    def dIdm(self, system_state = None):
        if self.p == 0: return 0.0

    def dIdh(self, system_state = None):
        if self.q == 0: return 0.0

    def dmdV(self, system_state = None):
        return self.dalphadV(m,system_state)

    def dhdV(self, system_state = None):
        pass

    def conductance_infty(self, system_state = None):
        V_m = self.membrane.phi(system_state)

        alpham = self.alpham(V_m)
        betam = self.betam(V_m)
        alphah = self.alphah(V_m)
        betah = self.betah(V_m)
        m_infty = alpham/(alpham+betam)
        h_infty = alphah/(alphah+betah)
        gate = power(m_infty,self.p)*power(h_infty,self.q)
        return {ion: gmax*gate for gmax,ion in zip(self.gmax,self.species) }

    def current_infty(self, system_state = None):
        """
        This is the current when the gates have equilibriated to V_m, and the
        given intra- and extra- cellular concentrations
        invalues and outvalues are dicts taken species are arguments
        """
        V_m = self.membrane.phi(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)

        alpham = self.alpham(V_m)
        betam = self.betam(V_m)
        alphah = self.alphah(V_m)
        betah = self.betah(V_m)
        m_infty = alpham/(alpham+betam)
        h_infty = alphah/(alphah+betah)

        # Maybe let's not compute this multiple times, once for each channel!
        if not hasattr(V_m, '__iter__'):
            # V_m is not iterable, use ternary operator
            I = [ gmax*power(m_infty,self.p)*power(h_infty,self.q)*F*species.z*V_m/phi \
                *( (self.membrane.inside.value(species) - exp(-V_m*species.z/phi)*self.membrane.outside.value(species))/(1.0-exp(-V_m*species.z/phi)) \
                if V_m*species.z >0 else \
                (self.membrane.inside.value(species)*exp(V_m*species.z/phi) -self.membrane.outside.value(species))/(exp(V_m*species.z/phi)-1.0 ) )\
                for gmax,species in zip(self.gmax,self.species)]
        else:
            c_in = self.membrane.inside.value(species)
            c_out = self.membrane.outside.value(species)
            I = [ gmax*power(m_infty,self.p)*power(h_infty,self.q)*F*species.z*V_m/phi \
                *np.fromiter( [ (cin-exp(-vm*species.z/phi)*cout)/(1.0-exp(-vm*species.z/phi)) \
                    if vm*species.z>0 else  \
                    (cin*exp(vm*species.z/phi)-cout)/(exp(vm*species.z/phi)-1.0)
                    for (vm,cin,cout) in zip(V_m,c_in,c_out) ], np.float64)
                for gmax,species in zip(self.gmax,self.species)]
        return {ion:current for ion,current in zip(self.species,I)}

    def mdot(self, V_m=None, m=None ):
        """ Compute dm/dt
        """
        if V_m is None: V_m = self.membrane.phi()
        if m is None: m = self.m
        return self.alpham(V_m)*(1.0-m)-self.betam(V_m)*m

    def hdot(self, V_m=None, h=None):
        """ Compute dh/dt
        """
        if V_m is None: V_m = self.membrane.phi()
        if h is None: h = self.h

        return self.alphah(V_m)*(1.0-h)-self.betah(V_m)*h

    def minfty(self, V_m=None):
        if V_m is None: V_m = self.membrane.phi()
        am = self.alpham(V_m)
        bm = self.betam(V_m)
        return am/(am+bm)

    def hinfty(self, V_m=None):
        if V_m is None: V_m = self.membrane.phi()
        ah = self.alphah(V_m)
        bh = self.betah(V_m)
        return ah/(ah+bh)

    def equilibriate(self,V_m=None):
        if V_m is None: V_m = self.membrane.phi()
        if not self.p == 0:
            self.m = self.minfty(V_m)
        else:
            self.m = 1.0
        if not self.q ==0:
            self.h = self.hinfty(V_m)
        else:
            self.h = 1.0

    def alpham(self, system_state = None):
        """
        Over-ride this method
        """
        return 1.0

    def betam(self, system_state = None):
        """
        Over-ride this method
        """
        return 1.0

    def alphah(self, system_state = None):
        return 1.0

    def betah(self, system_state = None):
        return 1.0

    def equilibriate_gates(self, system_state = None):
        V_m = self.membrane.phi(system_state)
        self.m = self.minfy(V_m)
        self.h = self.hinfty(V_m)

    def vectorizevalues(self):
        if not hasattr(self.m,'__iter__') and self.m is not None:
            self.m = np.ones(self.N)*self.m
        if not hasattr(self.h,'__iter__') and self.h is not None:
            self.h = np.ones(self.N)*self.h

    # Internal vars for the standard GHK equation are the gating variables
    # [p, q]
    # expect that system_state offset is set??
    def getInternalVars(self,system_state = None):
        if self.p == 0 and self.q==0: return None
        elif self.q ==0: return self.get_m(system_state)
        elif self.q == 0: return self.get_h(system_state)
        return np.array([self.get_m(system_state),self.get_h(system_state)]).flatten()

    def setInternalVars(self,system_state):
        if self.p == 0: self.h = self.get_h(system_state)
        elif self.q ==0: self.m = self.get_m(system_state)
        else:
            self.m = self.get_m(system_state)
            self.h = self.get_h(system_state)
        pass

    def get_dot_InternalVars(self,system_state,t):
        if self.p ==0 and self.q==0: return None
        elif self.p == 0:
            temp = self.hdot(self.membrane.phi(system_state),self.get_h(system_state))
        elif self.q == 0:
            temp = self.mdot(self.membrane.phi(system_state),self.get_m(system_state))
        else:
            temp = np.zeros(2*self.N)
            temp[:self.membrane.N] = self.mdot(self.membrane.phi(system_state),self.get_m(system_state))
            temp[self.membrane.N:] = self.hdot(self.membrane.phi(system_state),self.get_h(system_state))
        return temp



class HHChannel(Channel):
    gmax = []
    def current(self,system_state = None):
        V_m = self.membrane.phi(system_state)
        h = self.h(system_state)
        m = self.m(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        Eion = [ phi/species.z*(np.log(outvalues[species])-np.log(invalues[species])) for species in self.species]
        return self.gmax*(V_m-Eion)

    def species_current(self,species,system_state = None):
        E =  [ phi/species.z*(np.log(outvalues[species])-np.log(invalues[species])) for species in self.species]
        return self.gmax[self.species.index(species)]*(V_m-E)


class LeakChannel(HHChannel):
    def __init__(self,species):
        self.gmax = 0 # reset this to balance the specific ion current
        self.species = species # only a single species

    def current(self,system_state = None, invalues = None, outvalues = None):
        V_m = self.membrane.phi(system_state)
        Ce = self.membrane.outside.value(self.species,system_state) if invalues is None else invalues[self.species]
        Ci = self.membrane.inside.value(self.species,system_state) if outvalues is None else outvalues[self.species]

        return {self.species: self.gmax*(V_m-phi/self.species.z*(np.log(Ce)-np.log(Ci)))}

    def set_gmax(self,gmax):
        self.gmax = gmax