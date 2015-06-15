import ufl
import dolfin
from dolfin import inner, nabla_grad, dx
from csdmodel import *


class CSDModelDolfin(CSDModel):
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        self.v = dolfin.FunctionSpace(self.mesh,"Lagrange",2)

        # Each compartment is associated with a volume fraction
        self.compartments = []
        self.volfrac = {} # \sum v_j = 1, for v_j <=1
        """
         Store the volume fractions of each of the compartments
        """

        # membranes
        self.membranes = []

        self.numdiffusers = 0
        self.numpoisson = 0
        self.num_membrane_potentials = 0
        self.isAssembled = False

    def assembleSystem(self):
        """Assemble the FEM system. This is only run a single time before time-stepping. The values of the coefficient
        fields need to be updated between time-steps
        """
        # Loop through the entire model and composite the system of equations
        self.diffusors = []  # [[compartment, species, diffusivity of species],[ ...],[...]]
        """
           Diffusors have source terms
        """
        self.electrostatic_compartments = [] # (compartment) where electrostatic equations reside
        """
           Has source term
        """
        self.potentials = [] # (membrane)
        """
            No spatial derivatives, just construct ODE
        """
        self.channelvars = [] #(membrane, channel, ...)

        for compartment in self.compartments:
            s = 0
            for species in compartment.species:
                if compartment.diffusivities[species] < 1e-10: continue
                self.diffusors.extend([ [compartment,species,compartment.diffusivities[species]] ])
                s+=compartment.diffusivities[species]*abs(species.z)
            if s>0:
                self.electrostatic_compartments.extend([compartment])
                # Otherwise, there are no mobile charges in the compartment


        # the number of potentials is the number of spatial potentials + number of membrane potentials
        self.numdiffusers = len(self.diffusors)
        self.numpoisson = len(self.electrostatic_compartments)

        # Functions
        # Reaction-diffusion type
        #   Diffusers    :numdiffusers
        #
        # Coefficient
        #   Diffusivities
        self.V_np = dolfin.MixedFunctionSpace([self.v]*(self.numdiffusers))
        self.V_poisson = dolfin.MixedFunctionSpace([self.v]*self.numpoisson)
        #self.V = self.V_diff*self.V_electro
        self.V = dolfin.MixedFunctionSpace([self.v]*(self.numdiffusers+self.numpoisson))

        self.dofs_is = [self.V.sub(j).dofmap().dofs() for j in range(self.numdiffusers+self.numpoisson)]
        self.N = len(self.dofs_is[0])

        self.diffusivities = [dolfin.Function(self.v) for j in range(self.numdiffusers)]

        self.trialfunctions = dolfin.TrialFunctions(self.V)  # Trial function
        self.testfunctions = dolfin.TestFunctions(self.V) # test functions, one for each field

        self.sourcefunctions = [dolfin.Function(self.v) for j in range(self.numpoisson+self.numdiffusers)]

        self.permitivities = [dolfin.Function(self.v) for j in range(self.numpoisson)]

        # index the compartments!

        self.functions__ = dolfin.Function(self.V)

        self.np_assigner = dolfin.FunctionAssigner(self.V_np,[self.v]*self.numdiffusers)
        self.poisson_assigner = dolfin.FunctionAssigner(self.V_poisson,[self.v]*self.numpoisson)
        self.full_assigner = dolfin.FunctionAssigner(self.V,[self.v]*(self.numdiffusers+self.numpoisson))

        self.prev_value__ = dolfin.Function(self.V)
        self.prev_value_ = dolfin.split(self.prev_value__)
        self.prev_value = [dolfin.Function(self.v) for j in range(self.numdiffusers+self.numpoisson)]

        self.vfractionfunctions = [dolfin.Function(self.v) for j in range(len(self.compartments))]
        # Each reaction diffusion eqn should be indexed to a single volume fraction function
        # Each reaction diffusion eqn should be indexed to a single potential function
        # Each reaction diffusion eqn should be indexed to a single valence
        self.dt = dolfin.Constant(0.1)
        self.eqs = []
        self.phi = dolfin.Constant(phi)

        for membrane in self.membranes:
            self.potentials.extend([membrane])
            membrane.phi_m = np.ones(self.N)*membrane.phi_m
        self.num_membrane_potentials = len(self.potentials) # improve this

        """
            Assemble the equations for the system
            Order of equations:
            for compartment in compartments:
                for species in compartment.species
                    diffusion (in order)
                volume
                potential for electrodiffusion
            Membrane potentials


        """
        for j,compartment in enumerate(self.compartments):
            self.vfractionfunctions[j].vector()[:] = self.volfrac[compartment]

        # Set the reaction-diffusion equations
        for j,(trial,old,test,source,D,diffusor) in enumerate(zip(self.trialfunctions[:self.numdiffusers],self.prev_value_[:self.numdiffusers] \
                ,self.testfunctions[:self.numdiffusers], self.sourcefunctions[:self.numdiffusers]\
                ,self.diffusivities,self.diffusors)):
            """
            This instance of the loop corresponds to a diffusion species in self.diffusors
            """
            compartment_index = self.compartments.index(diffusor[0]) # we are in this compartment

            try:
                phi_index = self.electrostatic_compartments.index(diffusor[0]) + self.numdiffusers
                self.eqs.extend([ trial*test*dx-test*old*dx+ \
                        self.dt*(inner(D*nabla_grad(trial),nabla_grad(test)) + \
                        dolfin.Constant(diffusor[1].z/phi)*inner(D*trial*nabla_grad(self.prev_value[phi_index]),nabla_grad(test)))*dx   ])

                """
                self.eqs.extend([ trial*test*dx-test*old*dx+ \
                        self.dt*(inner(D*nabla_grad(trial),nabla_grad(test)) + \
                        dolfin.Constant(diffusor[1].z/phi)*inner(D*trial*nabla_grad(self.prev_value[phi_index]),nabla_grad(test)) - source*test)*dx   ])
                """
                # electrodiffusion here

            except ValueError:
                # No electrodiffusion for this species
                self.eqs.extend([trial*test*dx-old*test*dx+self.dt*(inner(D*nabla_grad(trial),nabla_grad(test))- source*test)*dx ])

            self.prev_value[j].vector()[:] = diffusor[0].value(diffusor[1])
            self.diffusivities[j].vector()[:] = diffusor[2]
            #diffusor[0].setValue(diffusor[1],self.prev_value[j].vector().array()) # do this below instead


        self.full_assigner.assign(self.prev_value__,self.prev_value)
        self.full_assigner.assign(self.functions__,self.prev_value)  # Initial guess for Newton

        """
        Vectorize the values that aren't already vectorized
        """
        for compartment in self.compartments:
            for j,(species, val) in enumerate(compartment.values.items()):
                try:
                    length = len(val)
                    compartment.internalVars.extend([(species,self.N,j*self.N)])
                    compartment.species_internal_lookup[species] = j*self.N
                except:
                    compartment.values[species]= np.ones(self.N)*val
                    #compartment.internalVars.extend([(species,self.N,j*self.N)])
                    compartment.species_internal_lookup[species] = j*self.N


        # Set the electrostatic eqns
        # Each equation is associated with a single compartment as defined in

        for j,(trial, test, source, eps, compartment) in enumerate(zip(self.trialfunctions[self.numdiffusers:],self.testfunctions[self.numdiffusers:], \
                self.sourcefunctions[self.numdiffusers:], self.permitivities, self.electrostatic_compartments)):
            # set the permitivity for this equation
            eps.vector()[:] = F**2*self.volfrac[compartment]/R/T \
                *sum([compartment.diffusivities[species]*species.z**2*compartment.value(species)  for species in compartment.species],axis=0)
            self.eqs.extend( [inner(eps*nabla_grad(trial),nabla_grad(test))*dx - source*test*dx] )

        compartmentfluxes = self.updateSources()

        #


        """
        Set indices for the "internal variables"
        List of tuples
        (compartment/membrane, num of variables)

        Each compartment or membrane has method
        getInternalVars()
        get_dot_InternalVars(t,values)
        get_jacobian_InternalVars(t,values)
        setInternalVars(values)

        The internal variables for each object are stored starting in
        y[obj.system_state_offset]
        """
        self.internalVars = []
        index = 0
        for membrane in self.membranes:
            index2 = 0
            for channel in membrane.channels:
                channeltmp = channel.getInternalVars()
                if channeltmp is not None:
                    self.internalVars.extend([ (channel, len(channeltmp),index2)])
                    channel.system_state_offset = index+index2
                    channel.internalLength = len(channeltmp)
                    index2+=len(channeltmp)
            tmp = membrane.getInternalVars()
            if tmp is not None:
                self.internalVars.extend( [(membrane,len(tmp),index)] )
                membrane.system_state_offset = index
                index += len(tmp)
                membrane.points = self.N
        """
        Compartments at the end, so we may reuse some computations
        """

        for compartment in self.compartments:
            index2 = 0
            compartment.system_state_offset = index  # offset for this object in the overall state
            for species, value in compartment.values.items():
                compartment.internalVars.extend([(species,len(compartment.value(species)),index2)])
                index2 += len(value)
            tmp = compartment.getInternalVars()
            self.internalVars.extend( [(compartment,len(tmp),index)] )
            index += len(tmp)
            compartment.points = self.N

        for key, val in self.volfrac.items():
            self.volfrac[key] = val*np.ones(self.N)

        """
        Solver setup below
        self.pdewolver is the FEM solver for the concentrations
        self.ode is the ODE solver for the membrane and volume fraction
        The ODE solve uses LSODA
        """
        # Define the problem and the solver
        self.equation = sum(self.eqs)
        self.equation_ = dolfin.action(self.equation, self.functions__)
        self.J = dolfin.derivative(self.equation_,self.functions__)
        ffc_options = {"optimize": True, \
            "eliminate_zeros": True, \
            "precompute_basis_const": True, \
            "precompute_ip_const": True, \
            "quadrature_degree": 2}
        self.problem = dolfin.NonlinearVariationalProblem(self.equation_, self.functions__, None, self.J, form_compiler_parameters=ffc_options)
        self.pdesolver  = dolfin.NonlinearVariationalSolver(self.problem)
        self.pdesolver.parameters['newton_solver']['absolute_tolerance'] = 1e-9
        self.pdesolver.parameters['newton_solver']['relative_tolerance'] = 1e-9

        """
        ODE integrator here. Add ability to customize the parameters in the future
        """
        self.t = 0.0
        self.odesolver = ode(self.ode_rhs) #
        self.odesolver.set_integrator('lsoda', nsteps=3000, first_step=1e-6, max_step=5e-3 )
        self.odesolver.set_initial_value(self.getInternalVars(),self.t)

        self.isAssembled = True

    def updateSources(self,system_state=None):
        """
         Update all of the functions first @TODO

        """

        # Compute the membrane fluxes for the reaction-diffusion equations

        #fluxes = {membrane: membrane.fluxes() for membrane in self.membranes}

        #compartmentfluxes = {compartment:collections.Counter() for compartment in self.compartments}

        if system_state is not None:
            self.setInternalVars(system_state)

        """
        compartmentfluxes represents the RHS of the reaction-diffusion eqs.
        """
        temp_poisson_source = [np.zeros(self.N) for j in range(self.numpoisson)]

        #for membrane in self.membranes:
        #    currents, fluxes = membrane.currents_and_fluxes()
        #    compartmentfluxes[membrane.outside].update(fluxes)
        #   compartmentfluxes[membrane.inside].update(scalar_mult_dict(fluxes,-1.0))
        #    # @TODO add reaction fluxes

        """
        #In our splitting scheme, source functions are zero for the purposes of solving
        # the spatial PDE. We still compute the source functions but we may want to comment
        # these lines out
        """
        #for j, (compartment, species, D) in enumerate(self.diffusors):
        #    self.sourcefunctions[j].vector()[:] = compartmentfluxes[compartment][species]

        # Do poisson. Solve the time dependent problem now. Just the elliptical equation here

        for j,(trial, test, source, eps,D , compartment) in enumerate(zip(self.trialfunctions[self.numdiffusers:],self.testfunctions[self.numdiffusers:], \
                self.sourcefunctions[self.numdiffusers:], self.permitivities, self.diffusivities, self.electrostatic_compartments)):
            # set the permitivity for this equation
            eps.vector()[:] = F**2*self.volfrac[compartment]/R/T \
                *sum([D.vector()[:]*species.z**2*compartment.value(species)  for species in compartment.species])
            self.sourcefunctions[self.numdiffusers+j].vector()[:] = temp_poisson_source[j] #@FIXME!!!

        #return compartmentfluxes