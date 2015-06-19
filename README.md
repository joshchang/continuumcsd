# CSDFEM

An object-oriented Pythonic implementation of multi-compartment continuum cortical spreading depression wave models.
The models consist of sets of Nernst-Planck equations coupled to ODEs describing source terms. The source terms
consist of the biochemistry of the neurons and other relevant cells in the brain. GHK is used as the formulation
for ion currents.

 For simulations over a 1-dimensional interval domain, the method of lines is used. Reflecting boundary
  conditions are assumed. For higher-dimension simulations, the FENICS library is used to implement finite
  elements. Lie splitting is used in this case. This code is still under heavy development, and may not be completely
  workable at this point. Check out the ipython notebook for useage.
