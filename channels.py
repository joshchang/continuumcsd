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
from collections import defaultdict

class NaTChannel(GHKChannel):
    species = [Na]
    p = 3
    q = 1
    max_permeability = np.array([3e-12])

    def alpham(self, V_m):
        return -320.0*(V_m*1000+51.9)/(exp(-.25*(V_m*1000+51.9))-1.0+1e-17)

    def betam(self, V_m):
        return 280.0*(V_m*1000+24.89)/(exp(.200*(V_m*1000+24.89))-1.0+1e-17)

    def alphah(self, V_m):
        return 128.0*exp(-(56*V_m+2.94))

    def betah(self, V_m):
        return 4000./(exp(-(200*V_m+6.0))+1.0)

class NaPChannel(GHKChannel):
    species = [Na]
    p = 2
    q = 1
    max_permeability = np.array([3e-12])

    def alpham(self, V_m):
        '''

        :param V_m: Membrane potential in Volts
        :return: 1/s
        '''
        return 1000*pow(6.0 * (1 + exp(-(143 * V_m + 5.67))), -1)

    def betam(self, V_m):
        return 1000.0/6.0*(1.0-1./(1+exp(-(143*V_m+5.67))))

    def alphah(self, V_m):
        return 5.12e-8 * 1000 *exp(-(56*V_m+2.94))


    def betah(self, V_m):
        return 1.6e-6 * 1000 * pow(1 + exp(-(200 * V_m + 8.0)), -1)

class KDRChannel(GHKChannel):
    # https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=113446&file=%5cNEURON-2008b%5ckdr.mod
    species = [K]
    p = 2
    q = 0
    max_permeability = np.array([1.75e-12])

    def alpham(self, V_m):
        return -16.0 *(V_m*1000+34.9)/(exp(-0.2*(V_m*1000+34.9))-1.0)

    def betam(self, V_m):
        return 250 * exp(-(250.0 * V_m + 1.25))

class KAChannel(GHKChannel):
    species = [K]
    max_permeability = np.array([2.5e-12])  # permeability
    p = 2
    q = 1

    def alpham(self, V_m):
        return -20*(V_m*1000+56.9)/(exp(-0.1*(V_m*1000+56.9))-1.0+1e-20)

    def betam(self, V_m):
        return 17.5*(V_m*1000+29.9)/(exp(0.1*(V_m*1000+29.9))-1.0+1e-20)

    def alphah(self, V_m):
        return 16.0*exp(-(56*V_m+4.61))

    def betah(self, V_m):
        return 500/(exp(-(200*V_m+11.98))+1)


class NMDAChannel(GHKChannel):
    p = 1
    q = 1
    # name = 'GLutamate-independent NMDA channel'
    species = [Na, K, Ca]
    max_permeability = np.array([2, 2, 20]) * 1e-12

    def alpham(self, V_m):
        return 0.5 / (1 + exp((13.5e-3 - self.membrane.outside.value(K)) / 1.42e-3))

    def betam(self, V_m):
        return 0.5 - self.alpham(V_m)

    def alphah(self, V_m):
        return power(2.0e3 * (1.0 + exp((self.membrane.outside.value(K) - 6.75e-3) / 0.71e-3)), -1)

    def betah(self, V_m):
        return 5e-4 - self.alphah(V_m)


class NaKATPasePump(Channel):
    species = [K, Na]
    gmax = np.array([-2.0, 3.0]) * 2e-4  # 2K per 3Na

    def current(self, system_state=None, V_m=None, invalues=None, outvalues=None):
        Ke = self.membrane.outside.value(K,system_state) if outvalues is None else outvalues[K]
        Nai = self.membrane.outside.value(Na,system_state) if invalues is None else invalues[Na]

        I = power(1.0 + 3.5e-3 / Ke, -2) * power(1.0 + 0.014 / Nai, -3)
        return {K: self.gmax[0]*I, Na: self.gmax[1]*I}


class NaCaExchangePump(Channel):
    # taken from Bennett et al.
    species = [Na, Ca]
    gmax = np.array([-3, 1]) * 2e-9

    def current(self, system_state=None, V_m = None, invalues=None, outvalues=None):
        if V_m is None: V_m = self.membrane.phi(system_state)
        if invalues is None:
            Cai = self.membrane.inside.value(Ca,system_state)
            Nai = self.membrane.inside.value(Na,system_state)
        else:
            Cai = invalues[Ca]
            Nai = invalues[Na]
        if outvalues is None:
            Cae = self.membrane.outside.value(Ca,system_state)
            Nae = self.membrane.outside.value(Na,system_state)
        else:
            Cae = outvalues[Ca]
            Nae = outvalues[Na]
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

        I = np.where(V_m<0, (power(Nai / Nae, 3) * (Cae / Cai) * exp(V_m / phi) - 2.5) / (1 + power(.0875 / Nae, 3)) / (
        Cae / Cai + 1.38e-3 / Cai) / (exp(0.65 * V_m / phi) + 0.1) ,  (power(Nai / Nae, 3) * (Cae / Cai) - 2.5 * exp(-V_m / phi)) / (1 + power(.0875 / Nae, 3)) / (
            Cae / Cai + 1.38e-3 / Cai) / (exp(-0.35 * V_m / phi) + 0.1 * exp(-V_m / phi)))

        """
        I = np.where(V_m<0, (power(Nai/Nae,3)*(Cae/Cai)*exp(V_m/phi)-2.5)/(1+power(.0875/Nae,3))/(Cae/Cai+1.38e-3/Cai)/(exp(0.65*V_m/phi)+0.1), # First line runs if <0
            (power(Nai/Nae,3)*(Cae/Cai)-2.5*exp(-V_m/phi))/(1+power(.0875/Nae,3))/(Cae/Cai+1.38e-3/Cai)/(exp(-0.35*V_m/phi) + 0.1*exp(-V_m/phi) )) # Second line if >0
        """
        return {Na: self.gmax[0]*I, Ca: self.gmax[1]*I}


class PMCAPump(Channel):
    species = [Ca]
    gmax = 2e-7 #@TODO Figure out this parameter!!

    def current(self, system_state = None, V_m=None, invalues=None, outvalues=None):
        h = 1.0
        KPMCA = 1e-6
        Cai = invalues[Ca] if invalues is not None else self.membrane.inside.value(Ca,system_state)
        return {Ca: self.gmax / (1 + power(KPMCA / Cai, h))}

class NonSpecificChlorideChannel(Channel):
    gmax = 2e-11
    species = [Cl]
    V_m = -0.07

    def __init__(self, V_m=None):
        if V_m is not None:
            self.V_m = V_m

    def current(self, system_state=None, V_m = None, invalues = None, outvalues = None):
        if V_m is None: V_m = self.membrane.phi(system_state)
        Cli = invalues[Cl] if invalues is not None else self.membrane.inside.value(Cl,system_state)
        Cle = outvalues[Cl] if outvalues is not None else self.membrane.outside.value(Cl,system_state)
        return {Cl: self.gmax * (V_m - phi / Cl.z * (np.log(Cle) - np.log(Cli)))}

    def current_infty(self, V_m):
        return self.current(V_m)


class KIRChannel(Channel):
    species = [K]
    gmax = 2.0e-15

    def current(self, system_state=None, V_m = None, invalues=None, outvalues=None):
        if V_m is None: V_m = self.membrane.phi(system_state)
        Ke = outvalues[K] if outvalues is not None else self.membrane.outside.value(K,system_state)
        Ki = invalues[K]  if invalues is not None else self.membrane.inside.value(K,system_state)
        E_K = phi / K.z * (np.log(Ke) - np.log(Ki))
        return {K: self.gmax*(V_m-E_K)/sqrt(Ke*1e3)/(1.0+exp(1000*(V_m-E_K))) }
        #return {K: self.gmax*(V_m-E_K)/sqrt(Ke*1e3)/np.where( V_m<E_K, (1.0+exp(1000*(V_m-E_K))), (1.0+exp(-1000*(V_m-E_K)))/ exp(-1000*(V_m-E_K)))}

    def current_infty(self, V_m):
        return self.current(V_m)


class gNMDAChannel(NMDAChannel):
    """
    Glutamate-dependent NMDA Channel
    """
    r1 = 0.072
    r2 = 6.6
    Popen = 0
    # name = 'Glutamate-dependent NMDA Channel'

    def get_dot_InternalVars(self, system_state, t):
        # Glutamate gating variable
        old = super(NMDAChannel, self).get_dot_InternalVars(system_state, t)
        Popen = self.get_Popen(system_state)
        g = self.membrane.outside.value(Glu, system_state)
        dotPopen = self.r1 * g * (1.0 - Popen) - self.r2 * Popen
        return np.concatenate([old, dotPopen])

    def get_Popen(self, system_state=None):
        if system_state is None:
            return self.Popen
        return system_state[self.system_state_offset + 2 * self.N:self.system_state_offset + 3 * self.N]

    def getInternalVars(self):
        old = super(NMDAChannel, self).getInternalVars()
        Popen = self.get_Popen()
        return np.concatenate([old, Popen])

    def setInternalVars(self, system_state):
        super(NMDAChannel, self).setInternalVars(system_state)
        self.Popen = self.get_Popen(system_state)

    def current(self, system_state=None, V_m=None, invalues = None, outvalues = None):
        old = super(NMDAChannel, self).current(system_state)  # this is a dict
        return scalar_mult_dict(old, self.get_Popen(system_state))

    def vectorizevalues(self):
        super(NMDAChannel, self).vectorizevalues()
        if self.Popen is not None and not isinstance(self.Popen, collections.Sequence):
            self.Popen = np.ones(self.N) * self.Popen


class KDRglialChannel(GHKChannel):
    p = 4
    q = 0
    species = [K]
    max_permeability = [1e-12]

    def alpham(self, V_m):
        scaletaun = 1.5
        shiftn = 0.05
        return scaletaun * 16000.0 * (0.0351 - V_m - shiftn - 0.07) / (exp((35.1e-3 - V_m - shiftn - .07) / 0.05) - 1.0)

    def betam(self, V_m):
        scaletaun = 1.5
        shiftn = 0.05
        return scaletaun * 250.0 * exp((.02 - V_m - 0.07) / 0.04)

class CaLChannel(GHKChannel):
    #Somjen channel: https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=113446&file=%5cNEURON-2008b%5ccal2.mod
    species = [Ca]
    max_permeability = np.array([1e-12])
    p = 2
    q = 0
    def alpham(self, V_m, system_state = None):
        return 333.333*15.69*(-1000*V_m-10+81.5)*exp((1000*V_m+10-81.5)/10.0)/(-exp((1000*V_m+10-81.5)/10.0)+1.0)
    def betam(self, V_m, system_state = None):
        return 333.333*0.29/exp((1000*V_m+10)/10.86)

class CaPChannel(GHKChannel):
    species = [Ca]
    max_permeability = np.array([4e-12])
    p = 1
    q = 1
    def alphah(self, V_m, system_state = None):
        return 001.5/(1+exp( (1000*V_m+29)/8))
    def betah(self, V_m, system_state = None):
        return 005.5*exp((1000*V_m+23)/8)/(1+exp((1000*V_m+23)/8))
    def alpham(self, V_m, system_state = None):
        return 8500.0*exp((1000*V_m-8)/12.5)/(1+exp((1000*V_m-8)/12.5))
    def betam(self, V_m, system_state=None):
        return 35000.0/(1+exp((1000*V_m+74)/14.5))

class KSKChannel(Channel):
    # https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=113446&file=%5cNEURON-2008b%5csk.mod
    gmax = 2e-11
    species = [K]
    def current(self, system_state=None,V_m=None, invalues = None, outvalues = None):
        Cai = self.membrane.inside.value(Ca,system_state) if invalues is None else invalues[Ca]
        return {K: self.gmax/(1+power(3e-7/Cai,4.7) ) }

class HoleChannel(Channel):
    def current(self, system_state=None, V_m=None, invalues = None, outvalues = None):
        if V_m is None: V_m = self.membrane.phi(system_state)
        if invalues is None: invalues = self.membrane.inside.get_val_dict(system_state)
        if outvalues is None: outvalues = self.membrane.outside.get_val_dict(system_state)
        return {species: self.gmax * (V_m - phi / species.z * (np.log(outvalues[species]) - np.log(invalues[species])))
                for species in self.species}

    def __init__(self, species, gmax):
        self.species = species
        self.gmax = gmax


class AquaPorin(Channel):
    species = []
    gmax = []

    def water_max_permeability(self, system_state=None):
        return 1.0

    def current(self,system_state=None, V_m=None, invalues = None, outvalues = None):
        return defaultdict(float)

