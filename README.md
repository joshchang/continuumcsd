# CSDFEM

An object-oriented Pythonic implementation of multi-compartment cortical spreading depression wave models.
 For simulations over a 1-dimensional interval domain, the method of lines is used. Reflecting boundary
  conditions are assumed. For higher-dimension simulations, the FENICS library is used to implement finite
  elements. Lie splitting is used in this case. This code is still under heavy development, and may not be completely
  workable at this point. Check out the ipython notebook for useage.