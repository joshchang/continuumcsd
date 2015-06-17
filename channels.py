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


class GHKChannel(Channel):
    m = 0.0
    h = 1.0
    def current(self, system_state = None):
        V_m = self.membrane.phi(system_state)
        h = self.get_h(system_state)
        m = self.get_m(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        print "calling GHKcurrent, want GHK conductance call instead!!"
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
                gmax*gating*(condition*(invalues[species]-exp(-insidestuff)*outvalues[species])/(1.0-exp(-insidestuff)) +\
                (condition-1)*(invalues[species]*exp(insidestuff)-outvalues[species])/(exp(insidestuff)-1.0)) \
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

    def conductance(self, system_state = None):
        V_m = self.membrane.phi(system_state)
        h = self.get_h(system_state)
        m = self.get_m(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
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
        Eion = [ self.membrane.phi_ion(species,system_state) for species in self.species]
        return self.gmax*(V_m-Eion)

    def species_current(self,species,system_state = None):
        E =  [ self.membrane.phi_ion(species) for species in self.species]
        return self.gmax[self.species.index(species)]*(V_m-E)


class LeakChannel(HHChannel):
    def __init__(self,species):
        self.gmax = 0 # reset this to balance the specific ion current
        self.species = species # only a single species

    def current(self,system_state = None):
        V_m = self.membrane.phi(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)

        return {self.species: self.gmax*(V_m-self.membrane.phi_ion(self.species,system_state))}

    def set_gmax(self,gmax):
        self.gmax = gmax

class Pump(Channel):
    def current(self,system_state = None):
        pass

class NaKATPasePump(Pump):
    species = [K,Na]
    gmax = np.array([-2.0,3.0])*2e-6  #2K per 3Na
    def current(self,system_state = None):
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        I=power(1.0+3.5e-3/outvalues[K] , -2)*power(1.0+0.014/outvalues[Na],-3)
        return {ion:I*gmax for ion,gmax in zip(self.species,self.gmax)}

class NaCaExchangePump(Pump):
    # taken from Bennett et al.
    species = [Na,Ca]
    gmax = np.array([-3,1])*2e-7
    def current(self,system_state=None):
        V_m = self.membrane.phi(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        Cae = outvalues[Ca]
        Cai = invalues[Ca]
        Nae = outvalues[Na]
        Nai = invalues[Na]
        '''
        # We want to do this, but it is SUPER SLOW!!!
        if hasattr(Nai,'__iter__'):
            # Prevent overflows
            I = np.fromiter([(power(nai/nae,3)*(cae/cai)*exp(vm/phi)-2.5)/(1+power(.0875/nae,3)) \
                /(cae/cai+1.38e-3/cai)/(exp(0.65*vm/phi)+0.1) \
                if vm<0 else \
                (power(nai/nae,3)*(cae/cai)-2.5*exp(-vm/phi))/(1+power(.0875/nae,3)) \
                    /(cae/cai+1.38e-3/cai)/(exp(-0.35*vm/phi) + 0.1*exp(-vm/phi)  ) \
                for (nai,nae,cai,cae,vm) in zip(Nai,Nae,Cai,Cae,V_m)],np.float64)
            pass
        else:
            gamma = 0.35
            I = (power(Nai/Nae,3)*(Cae/Cai)*exp(gamma*V_m/phi)-2.5*exp((gamma-1.0)*V_m/phi))/(1+power(.0875/Nae,3)) \
                /(Cae/Cai+1.38e-3/Cai)/(0.1*exp((gamma-1)*V_m/phi)+1)
        '''

        # Maybe this is faster??

        I = (V_m<0)*(power(Nai/Nae,3)*(Cae/Cai)*exp(V_m/phi)-2.5)/(1+power(.0875/Nae,3))/(Cae/Cai+1.38e-3/Cai)/(exp(0.65*V_m/phi)+0.1)+\
            (V_m>=0)*(power(Nai/Nae,3)*(Cae/Cai)-2.5*exp(-V_m/phi))/(1+power(.0875/Nae,3))/(Cae/Cai+1.38e-3/Cai)/(exp(-0.35*V_m/phi) + 0.1*exp(-V_m/phi) )

        """
        I = np.where(V_m<0, (power(Nai/Nae,3)*(Cae/Cai)*exp(V_m/phi)-2.5)/(1+power(.0875/Nae,3))/(Cae/Cai+1.38e-3/Cai)/(exp(0.65*V_m/phi)+0.1), # First line runs if <0
            (power(Nai/Nae,3)*(Cae/Cai)-2.5*exp(-V_m/phi))/(1+power(.0875/Nae,3))/(Cae/Cai+1.38e-3/Cai)/(exp(-0.35*V_m/phi) + 0.1*exp(-V_m/phi) )) # Second line if >0
        """
        return {ion:gmax*I for ion,gmax in zip(self.species,self.gmax)}

class PMCAPump(Pump):
    species = [Ca]
    gmax = 1e-5

    def current(self,V_m = None, invalues = None, outvalues = None):
        h = 1.0
        KPMCA = 1e-6
        if invalues is None: invalues = self.membrane.inside.values
        if outvalues is None: outvalues = self.membrane.outside.values
        Cai = invalues[Ca]
        return {Ca: self.gmax/(1+power(KPMCA/Cai,h))}

class NaPChannel(GHKChannel):
    species = [Na]
    p = 2
    q = 1
    gmax = 2e-9
    def alpham(self, V_m):
        return pow(6.0*(1+exp(-(143*V_m+5.67))),-1)

    def alphah(self, V_m):
        return 5.12e-8*exp(-(56*V_m+2.94))

    def betam(self, V_m):
        return self.alpham(V_m)*exp(-(143*V_m+5.67));

    def betah(self, V_m):
        return 1.6e-6*pow(1+exp(-(200*V_m+8.0)),-1)

class NaTChannel(GHKChannel):
    species = [Na]
    p = 3
    q = 1
    gmax = np.array([2e-9])
    def alpham(self, V_m):
        return 0.32*(V_m+51.9e-3)/(1.0-exp(-(250*V_m+12.975)))

    def betam(self, V_m):
        return 0.28*(V_m+24.89e-3)/(exp(-(200*V_m+4.978))-1.0)

    def alphah(self, V_m):
        return 0.128*exp(-56*V_m+2.94)

    def betah(self, V_m):
        return 1.6e-6/(1+exp(-200*V_m+6.0))

class KDRChannel(GHKChannel):
    species = [K]
    p = 2
    q = 0
    gmax = np.array([1.75e-9])
    def alpham(self, V_m):
        return 16.0*(V_m+34.9e-3)/(1.0-exp(-(200*V_m+6.98)))
    def betam(self, V_m):
        return 0.25*exp(-(250.0*V_m+1.25))


class NonSpecificChlorideChannel(Channel):
    gmax = 2e-9
    species = [Cl]
    V_m = -0.07
    def __init__(self,V_m=None):
        if V_m is not None:
            self.V_m = V_m

    def current(self,system_state=None):
        V_m = self.membrane.phi(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        return {Cl: self.gmax*(V_m-self.membrane.phi_ion(Cl,system_state)) }

    def current_infty(self,V_m):
        return self.current(V_m)

class KIRChannel(Channel):
    species = [K]
    gmax = 2.0e-9
    def current(self,system_state=None):
        V_m = self.membrane.phi(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        return {K:1.0*self.gmax* (V_m-self.membrane.phi_ion(K,system_state)) \
            /(sqrt(outvalues[K])*(1.0+exp(V_m+0.01+0.08 )))  }

    def current_infty(self,V_m):
        return self.current(V_m)


class KAChannel(GHKChannel):
    species = [K]
    gmax = np.array([2.5e-9]) # Siemens
    p =2
    q =1
    def alpham(self,V_m):
        return 20.0*(V_m+56.9e-3)/(1-exp(-(100*V_m+5.69)))

    def betam(self, V_m):
        return 17.5*(V_m+29.9e-3)/(exp(-(100*V_m+2.99))-1)

    def alphah(self, V_m):
        return 16.0*exp(-(56*V_m+4.61))

    def betah(self, V_m):
        return 0.5/(1+exp(-(200*V_m+11.98)))


class NMDAChannel(GHKChannel):
    p = 1
    q = 1
    r = 1
    species = [Na, K, Ca]
    gmax = np.array([2,2,20])*1e-9

    def alpham(self,V_m):
        return 0.5/(1+exp( (13.5e-3-self.membrane.outside.value(K))/1.42e-3))

    def betam(self, V_m):
        return 0.5 - self.alpham(V_m)

    def alphah(self, V_m):
        return power(2.0e3*(1.0+exp((self.membrane.outside.value(K)-6.75e-3)/0.71e-3)),-1)

    def betah(self, V_m):
        return 5e-4 - self.alphah(V_m)

class KDRglialChannel(GHKChannel):
    p = 4
    q = 0
    species = [K]
    gmax = [2e-9]

    def alpham(self,V_m):
        scaletaun = 1.5
        shiftn = 0.05
        return scaletaun*16.0*(0.0351-V_m-shiftn-0.07)/(exp((35.1e-3-V_m-shiftn-.07)/0.05)-1.0)

    def betam(self,V_m):
        scaletaun = 1.5
        shiftn = 0.05
        return scaletaun*0.25*exp((.02-V_m-0.07)/0.04)

class HoleChannel(Channel):
    def current(self,V_m=None,system_state=None):
        V_m = self.membrane.phi(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        return {species:self.gmax*(V_m-self.membrane.phi_ion(species)) for species in self.species }

    def __init__(self,species,gmax):
        self.species = species
        self.gmax = gmax

class AquaPorin(Channel):
    species = []
    gmax = []
    def water_permeability(self, system_state=None):
        return 1e-5