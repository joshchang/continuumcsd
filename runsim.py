from __future__ import print_function
from matplotlib import animation
import time
from channels import *
from compartment import *
from membrane import *
from params import *
from glutamate import *

from csdmodel1d import *

import numpy as np
np.seterr(all='ignore')
from scipy.integrate import ode

model = CSDModelInterval(N=32,dx=50e-6) # define the model, grid spacing is 100 microns, or approximately two cell widths

# Define the compartments, and the membranes
ecs = Compartment("ecs")
neuron = CellCompartment("neuron",density = 2e5) # 2e5 neurons per meter, 4e10 per sq meter
glia = CellCompartment("glia",density = 2e5) #2e5 glia per meter

neuronal_er = CellCompartment("neuron_er",density = 2e5)
glial_er = CellCompartment("glia_er",density = 2e5)

neuronal_mito = CellCompartment("neuron_mito",density = 2e5)
glial_mito = CellCompartment("glial_mito",density = 2e5)

neuron_mem = Membrane("neuronal",inside=neuron,outside=ecs,Cm=Cn,phi_m=phi0)
glial_mem = Membrane("glial",inside=glia,outside=ecs,Cm=Cg,phi_m=phig0)

neuronal_er_mem = Membrane("neuronal_er",inside=neuronal_er,outside=neuron,Cm=Cn,phi_m=0.0)
neuronal_er_mem = Membrane("glial_er",inside=glial_er,outside=glia,Cm=Cn,phi_m=0.0)


# Add the compartments to the model
model.addCompartment(ecs,fraction=0.2) # ECS take 20% of the total volume
model.addCompartment(neuron,fraction=0.4-0.04) # Neurons take up 40% of the total volume
model.addCompartment(glia,fraction=0.4-0.04) # Neurons take up 40% of the total volume
model.addCompartment(neuronal_er,fraction = 0.02)
model.addCompartment(glial_er,fraction = 0.02)
model.addCompartment(neuronal_mito,fraction = 0.02)
model.addCompartment(glial_mito,fraction = 0.02)

# Add ion species
ecs.addSpecies(K,Ke0,name='K_e')
ecs.addSpecies(Cl,Cle0,name='Cl_e')
ecs.addSpecies(Na,Nae0,name='Na_e')
ecs.addSpecies(Ca,Cae0,name='Ca_e')
ecs.addSpecies(Glu,ge0,name = "g_e") # 4 micromolar in ecs

neuron.addSpecies(K,Ki0,0,'K_n')
neuron.addSpecies(Na,Nai0,0,'Na_n')
neuron.addSpecies(Cl,Cli0,0,'Cl_n')
neuron.addSpecies(Ca,Cai0,0,'Ca_n')
neuron.addSpecies(Glu,1e-6,name = "g_n")


glia.addSpecies(K,Ki0,name='K_g')
glia.addSpecies(Na,Nai0,name='Na_g')
glia.addSpecies(Cl,Clg0,name='Cl_g')
glia.addSpecies(Ca,Cai0,0,'Ca_g')

# add channels
neuron_mem.addChannel(NaTChannel(),10000.) # 10000 per neuron?
neuron_mem.addChannel(NaPChannel(),100.) # 100 per neuron
neuron_mem.addChannel(KDRChannel(),10000.) # number of channels per neuron
neuron_mem.addChannel(KAChannel(),10000.) # number of channels per neuron
neuron_mem.addChannel(CaPChannel(),10000.) # number of channels per neuron
neuron_mem.addChannel(CaLChannel(),10000.) # number of channels per neuron
neuron_mem.addChannel(CaNChannel(),10000.) # number of channels per neuron
neuron_mem.addChannel(gNMDAChannel(),10000.)

neuron_mem.addChannel(PMCAPump(),10000) # PMCA pump
neuron_mem.addChannel(NaCaExchangePump(),1000) # sodium-calcium exchanger
neuron_mem.addChannel(NaKATPasePump(),4e5) # 5000 ATPase per neuron
neuron_mem.addChannel(NonSpecificChlorideChannel(phi0),100000)
neuron_mem.addChannel(AquaPorin(),1e-6) # Add water exchange

glial_mem.addChannel(KIRChannel(),50000) # KIR Channel
glial_mem.addChannel(NaKATPasePump(),4e5) # 10000000 ATPase per glia
glial_mem.addChannel(KDRglialChannel(),175000)
glial_mem.addChannel(PMCAPump(),10000)
glial_mem.addChannel(NaCaExchangePump(),1000) # sodium-calcium exchanger
glial_mem.addChannel(NonSpecificChlorideChannel(-0.085),100000)
glial_mem.addChannel(AquaPorin(),1e-6) # Add water exchange

glial_mem.addChannel(CaPChannel(),10000.) # number of channels per neuron
glial_mem.addChannel(CaLChannel(),10000.) # number of channels per neuron
glial_mem.addChannel(CaNChannel(),10000.) # number of channels per neuron

model.addMembrane(neuron_mem)
model.addMembrane(glial_mem)

neuron_mem.addLeakChannels()
neuron.balanceWith(ecs)
glial_mem.addLeakChannels()
glia.balanceWith(ecs)

model.assembleSystem()

system_state = model.getInternalVars()

y = model.getInternalVars()
model.odesolver.set_initial_value(y,0)

stim_duration = 4.0 # poke holes for 1 seconds
model.odesolver.t = -stim_duration
system_states = []
t = []
t.append(-stim_duration)
start_time = time.time()

if __name__== '__main__':
    print('{:<7} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format('time', 'V_n','V_g', 'K_e', 'K_n','K_g','Cl_e','Ca_n','g_e') )
    print('{:<7.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:10.6f} {:10.6f}'.format(model.odesolver.t, 1e3*neuron_mem.phi()[0], 1e3*glial_mem.phi()[0], 1e3*ecs.value(K)[0], 1e3*neuron.value(K)[0], 1e3*glia.value(K)[0], 1e3*ecs.value(Cl)[0], 1e3*neuron.value(Ca)[0], 1e3*ecs.value(Glu)[0]))

    try:
        y = model.odesolver.integrate(model.odesolver.t+1e-6)
        system_states.append(y)
        t.append(model.odesolver.t)
        while model.odesolver.successful() and model.odesolver.t < 0.0:
            y = model.odesolver.integrate(model.odesolver.t+1e-3)
            if sum(np.isnan(y))>0: break
            print('{:<7.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:10.6f}'.format(model.odesolver.t, 1e3*neuron_mem.phi(y)[0], 1e3*glial_mem.phi(y)[0], 1e3*ecs.value(K,y)[0], 1e3*neuron.value(K,y)[0], 1e3*glia.value(K,y)[0], 1e3*ecs.value(Cl,y)[0], 1e3*neuron.value(Ca,y)[0], 1e3*ecs.value(Glu,y)[0]))
            system_states.append(y)
            t.append(model.odesolver.t)

        while model.odesolver.successful() and model.odesolver.t < 120.0:
            y = model.odesolver.integrate(model.odesolver.t+1e-3)
            if sum(np.isnan(y))>0: break
            print('{:<7.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:10.6f}'.format(model.odesolver.t, 1e3*neuron_mem.phi(y)[0], 1e3*glial_mem.phi(y)[0], 1e3*ecs.value(K,y)[0], 1e3*neuron.value(K,y)[0], 1e3*glia.value(K,y)[0], 1e3*ecs.value(Cl,y)[0], 1e3*neuron.value(Ca,y)[0], 1e3*ecs.value(Glu,y)[0]))
            system_states.append(y)
            t.append(model.odesolver.t)
    except KeyboardInterrupt:
        print('{:<7.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:10.6f}'.format(model.odesolver.t, 1e3*neuron_mem.phi(y)[0], 1e3*glial_mem.phi(y)[0], 1e3*ecs.value(K,y)[0], 1e3*neuron.value(K,y)[0], 1e3*glia.value(K,y)[0], 1e3*ecs.value(Cl,y)[0], 1e3*neuron.value(Ca,y)[0], 1e3*ecs.value(Glu,y)[0]))

    print('{:<7.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:10.6f}'.format(model.odesolver.t, 1e3*neuron_mem.phi(y)[0], 1e3*glial_mem.phi(y)[0], 1e3*ecs.value(K,y)[0], 1e3*neuron.value(K,y)[0], 1e3*glia.value(K,y)[0], 1e3*ecs.value(Cl,y)[0],1e3*neuron.value(Ca,y)[0], 1e3*ecs.value(Glu,y)[0]))
    elapsed_time = time.time() - start_time
    print('{:<7.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:10.6f}'.format(model.odesolver.t, 1e3*neuron_mem.phi(y)[0], 1e3*glial_mem.phi(y)[0], 1e3*ecs.value(K,y)[0], 1e3*neuron.value(K,y)[0], 1e3*glia.value(K,y)[0], 1e3*ecs.value(Cl,y)[0], 1e3*neuron.value(Ca,y)[0], 1e3*ecs.value(Glu,y)[0]))
    print("\nElapsed time %.3f" % (elapsed_time) )

