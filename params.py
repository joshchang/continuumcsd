from numpy import exp

# SI UNITS!!
R = 8.314   # V C/ mol K
F = 96485 # C/mol
T = 310 # K
phi = R*T/F

# These are steady-state diffusivities
DK = 1.96e-9 #m^2/s
DNa = 1.33e-9
DCl = 2.03e-9
DO2 = 5e-10
DGlu = 4e-10
DATP = 8e-10
DCa = 2e-9 # fix this!!
DIP3 = 2.8e-10

# Per neuron
Cn = 1.70e-10    # Total cell capacitance of a neuron in Farads
An_cm = Cn*1e6   # Surface area in centimeters
An = An_cm*1e-4  # Surface area of a neuron in square centimeters

Cg = 1.345e-11  # http://jn.physiology.org/content/82/5/2731
Ag_cm = Cg*1e6  #square cm
Ag = Ag_cm*1e-4 # squared meters

# 1e-6 F/cm^2

# Initial values
Ke0 = 3.5e-3
Nae0 = 140e-3
Nai0 = 10e-3 # M
Nag0 = 30e-3
Ki0 = 133.5e-3 #M
Kg0 = 113.5e-3 # Kager 2009
phi0 = -70e-3 #V
phig0 = -85e-3
Cae0 = 1e-3
Cle0 = Ke0+Nae0+2*Cae0
ge0 = 4e-8  # M http://www.jneurosci.org/content/27/36/9736.full
Glun0 = 5e-4  # who knows?? Tune this until we have exhaustion on the right kinetics

Cli0 = exp(phi0/phi)*Cle0
Clg0 = exp(phig0/phi)*Cle0
A0 = Ki0+Nai0-Cli0 # Electroneutrality and isotonicity

Cai0 = 50e-9
Cag0 = 50e-9