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
from channel import *
import collections



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
        if isinstance(Nai, collections.Sequence):
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
    gmax = np.array([2e-9])
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
        return {Cl: self.gmax*(V_m-phi/Cl.z*(np.log(outvalues[Cl])-np.log(invalues[Cl]))) }

    def current_infty(self,V_m):
        return self.current(V_m)

class KIRChannel(Channel):
    species = [K]
    gmax = 2.0e-9
    def current(self,system_state=None):
        V_m = self.membrane.phi(system_state)
        invalues = self.membrane.inside.get_val_dict(system_state)
        outvalues = self.membrane.outside.get_val_dict(system_state)
        Ek = phi/K.z*(np.log(outvalues[K])-np.log(invalues[K]))
        return {K:1.0*self.gmax* (V_m-Ek) \
            /(sqrt(outvalues[K]*1e3)*(1.0+exp(1000*(V_m-Ek) )))  }

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
    #name = 'GLutamate-independent NMDA channel'
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


class gNMDAChannel(NMDAChannel):
    """
    Glutamate-dependent NMDA Channel
    """
    r1 = 0.072
    r2 = 6.6
    Popen = 0
    #name = 'Glutamate-dependent NMDA Channel'

    def get_dot_InternalVars(self,system_state,t):
        # Glutamate gating variable
        old = super(NMDAChannel, self).get_dot_InternalVars(system_state,t)
        Popen = self.get_Popen(system_state)
        g = self.membrane.outside.value(Glu,system_state)
        dotPopen = self.r1*g*(1.0-Popen) - self.r2*Popen
        return np.concatenate([old,dotPopen])

    def get_Popen(self,system_state=None):
        if system_state is None:
            return self.Popen
        return system_state[self.system_state_offset+2*self.N:self.system_state_offset+3*self.N]

    def getInternalVars(self):
        old = super(NMDAChannel,self).getInternalVars()
        Popen = self.get_Popen()
        return np.concatenate([old,Popen])

    def setInternalVars(self,system_state):
        super(NMDAChannel,self).setInternalVars(system_state)
        self.Popen = self.get_Popen(system_state)

    def current(self,V_m=None,system_state=None):
        old = super(NMDAChannel,self).current(system_state) # this is a dict
        return scalar_mult_dict(old,self.get_Popen(system_state))

    def vectorizevalues(self):
        super(NMDAChannel,self).vectorizevalues()
        if self.Popen is not None and not isinstance(self.Popen, collections.Sequence):
            self.Popen = np.ones(self.N)*self.Popen

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
        return {species:self.gmax*(V_m-phi/species.z*(np.log(outvalues[species])-np.log(invalues[species]))) for species in self.species }

    def __init__(self,species,gmax):
        self.species = species
        self.gmax = gmax

class AquaPorin(Channel):
    species = []
    gmax = []
    def water_permeability(self, system_state=None):
        return 1e-5