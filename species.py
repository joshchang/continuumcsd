from params import *

class Species(object):
    def __init__(self,name,z=0,tonicity=1,diffusivity=0):
        """Define an ion or other species to track

        SI Units
        """
        self.name = name
        self.z = z
        self.tonicity = tonicity # molar
        self.diffusivity = diffusivity  #m^2/s

    def __str__(self):
        return self.name


K = Species("K+",1,1,DK)
Na = Species("Na+",1,1,DNa)
Cl = Species("Cl-",-1,1,DCl)
Ca = Species("Ca2+",2,1,DCa)
O2 = Species("O2",0,0,DO2)
ATP = Species("ATP",-2,1,DATP)
IP3 = Species("IP3",-3,1,DIP3)
Glu = Species("Glutamate",3,1,DGlu)
Anion = Species("Anion",-1,1,0)   # nonspecific cation
Cation = Species("Cation",1,1,0)  # nonspecific anion
