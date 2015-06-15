from numpy import exp

# SI UNITS!!
R = 8.31e-3   # V C/ mol K
F = 96.485 # C/mol
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
Cn = 170e-10
Cg = 13.45e-10 #http://jn.physiology.org/content/82/5/2731

# Initial values
Ke0 = 3.5e-3
Nae0 = 140e-3
Nai0 = 10e-3 # M
Ki0 = 133.5e-3 #M
phi0 = -70e-3 #V
Cae0 = 1e-3
Cle0 = Ke0+Nae0+2*Cae0

Cli0 = exp(phi0/phi)*Cle0
A0 = Ki0+Nai0-Cli0 # Electroneutrality and isotonicity

Cai0 = 1e-9
