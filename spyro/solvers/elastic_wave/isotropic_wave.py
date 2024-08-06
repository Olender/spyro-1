import numpy as np

from firedrake import (Constant, DirichletBC, Function)

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from ...domains.space import FE_method
from ...utils.typing import override

class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''
    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)

        self.rho = None   # Density
        self.lmbda = None # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity

        self.u_n = None   # Current displacement field
        self.u_nm1 = None # Displacement field in previous iteration
        self.u_np1 = None # Displacement field in next iteration

        # Volumetric sourcers (defined through UFL)
        self.body_forces = None

        # Boundary conditions
        self.bcs = []
    
    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        def constant_wrapper(value):
            if np.isscalar(value):
                return Constant(value)
            else:
                return value
        
        def get_value(key, default=None):
            return constant_wrapper(synthetic_data_dict.get(key, default))
        
        self.rho = get_value("density")
        self.lmbda = get_value("lambda", default=get_value("lame_first"))
        self.mu = get_value("mu", get_value("lame_second"))
        self.c = get_value("p_wave_velocity")
        self.c_s = get_value("s_wave_velocity")
        
        # Check if {rho, lambda, mu} is set and {c, c_s} are not
        option_1 = bool(self.rho) and \
                   bool(self.lmbda) and \
                   bool(self.mu) and \
                   not bool(self.c) and \
                   not bool(self.c_s)
        # Check if {rho, c, c_s} is set and {lambda, mu} are not
        option_2 = bool(self.rho) and \
                   bool(self.c) and \
                   bool(self.c_s) and \
                   not bool(self.lmbda) and \
                   not bool(self.mu)

        if option_1:
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        elif option_2:
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
        else:
            raise Exception(f"Inconsistent selection of isotropic elastic wave parameters:\n" \
                            f"    Density        : {bool(self.rho)}\n"\
                            f"    Lame first     : {bool(self.lmbda)}\n"\
                            f"    Lame second    : {bool(self.mu)}\n"\
                            f"    P-wave velocity: {bool(self.c)}\n"\
                            f"    S-wave velocity: {bool(self.c_s)}\n"\
                            "The valid options are \{Density, Lame first, Lame second\} "\
                            "or (exclusive) \{Density, P-wave velocity, S-wave velocity\}")
    
    @override
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError
    
    @override
    def _create_function_space(self):
        return FE_method(self.mesh, self.method, self.degree,
                         dim=self.mesh.ufl_cell().geometric_dimension())

    @override
    def _set_vstate(self, vstate):
        self.u_n.assign(vstate)

    @override
    def _get_vstate(self):
        return self.u_n

    @override
    def _set_prev_vstate(self, vstate):
        self.u_nm1.assign(vstate)

    @override
    def _get_prev_vstate(self):
        return self.u_nm1

    @override
    def _set_next_vstate(self, vstate):
        self.u_np1.assign(vstate)

    @override
    def _get_next_vstate(self):
        return self.u_np1
    
    @override
    def get_receivers_output(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self):
        return self.u_n

    @override
    def get_function_name(self):
        return "Displacement"

    @override
    def matrix_building(self):
        self.current_time = 0.0

        self.u_n = Function(self.function_space,
                            name=self.get_function_name())
        self.u_nm1 = Function(self.function_space,
                              name=self.get_function_name())
        self.u_np1 = Function(self.function_space,
                              name=self.get_function_name())
        
        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()
        
        if self.abc_boundary_layer_type is None:
            isotropic_elastic_without_pml(self)
        elif self.abc_boundary_layer_type == "PML":
            isotropic_elastic_with_pml(self)
    
    @override
    def rhs_no_pml(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.B
    
    def parse_initial_conditions(self):
        time_dict = self.input_dictionary["time_axis"]
        initial_condition = time_dict.get("initial_condition", None)
        if initial_condition is not None:
            x_vec = self.get_spatial_coordinates()
            self.u_n.interpolate(initial_condition(x_vec, 0 - self.dt))
            self.u_nm1.interpolate(initial_condition(x_vec, 0 - 2*self.dt))
    
    def parse_boundary_conditions(self):
        bc_list = self.input_dictionary.get("boundary_conditions", [])
        for tag, id, value in bc_list:
            if tag == "u":
                subspace = self.function_space
            elif tag == "uz":
                subspace = self.function_space.sub(0)
            elif tag == "ux":
                subspace = self.function_space.sub(1)
            elif tag == "uy":
                subspace = self.function_space.sub(2)
            else:
                raise Exception(f"Unsupported boundary condition with tag: {tag}")
            self.bcs.append(DirichletBC(subspace, value, id))
    
    def parse_volumetric_forces(self):
        acquisition_dict = self.input_dictionary["acquisition"]
        body_forces_data = acquisition_dict.get("body_forces", None)
        if body_forces_data is not None:
            x_vec = self.get_spatial_coordinates()
            self.body_forces = body_forces_data(x_vec, self.time)