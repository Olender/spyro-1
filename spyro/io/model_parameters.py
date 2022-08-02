import warnings
import spyro

default_optimization_parameters = {
    "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
    "Step": {
        "Type": "Augmented Lagrangian",
        "Augmented Lagrangian": {
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 5.0,
        },
        "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
    },
    "Status Test": {
        "Gradient Tolerance": 1e-16,
        "Iteration Limit": None,
        "Step Tolerance": 1.0e-16,
    },
}

default_dictionary = {}
default_dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped', # lumped, equispaced or DG, default is lumped
    "method": "MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
default_dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
default_dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
}
default_dictionary["synthetic_data"] = {    #For use only if you are using a synthetic test model
    "real_mesh_file": None,
    "real_velocity_file": None,
}
default_dictionary["inversion"] = {
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": default_optimization_parameters,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
default_dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
default_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 0.9), 20
    ),
}

# Simulate for 2.0 seconds.
default_dictionary["time_axis"] = {
    "initial_time": 0.0,  #  Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}

class model_parameters:
    def __init__(self, dictionary=default_dictionary):
        self.input_dictionary = dictionary
        self.cell_type = None
        self.method = None
        self.variant = None
        self.__get_method()
        
        self.degree = model_parameters.degree
        self.dimension = model_parameters.dimension
        self.final_time = model_parameters.final_time
        self.dt = model_parameters.dt
        self.initial_velocity_model = model_parameters.get_initial_velocity_model()
        self.function_space = None
        self.foward_output_file = 'forward_output.pvd'
        self.current_time = 0.0
        self.solver_parameters = model_parameters.solver_parameters
        self.c = self.initial_velocity_model
                
    def __unify_method_input(self):
        unified_method = None
        method = self.method
        if method == 'KMV' or method == 'MLT' or method == 'mass_lumped_triangle' or method == 'mass_lumped_tetrahedra':
            unified_method = 'mass_lumped_triangle'
        elif method == 'spectral' or method == 'SEM' or method == 'spectral_quadrilateral':
            unified_method = 'spectral_quadrilateral'
        elif method == 'DG_triangle':
            unified_method = method
        elif method == 'DG_quadrilatral':
            unified_method = method
        elif method == 'CG':
            unified_method = method
        else:
            warnings.warn(f"Method of {method} not accepted.")
        self.method = unified_method

    def __unify_cell_type_input(self):
        unified_cell_type = None
        dimension = self.dimension
        cell_type = self.cell_type
        if cell_type == 'T' or cell_type == 'triangles' or cell_type == 'triangle' or cell_type == 'tetrahedron' or cell_type == 'tetrahedra':
            unified_cell_type = 'triangle'
        elif cell_type == 'Q' or cell_type == 'quadrilateral' or cell_type == 'quadrilaterals' or cell_type == 'hexahedron' or cell_type == 'hexahedra':
            unified_cell_type = 'quadrilateral'
        elif cell_type == None:
            unified_cell_type = None
        else:
            warnings.warn(f"Cell type of {cell_type} not accepted.")
        self.cell_type = unified_cell_type

    def __unify_variant_input(self):
        unified_variant = None
        variant = self.variant

        if variant == 'spectral' or variant == 'GLL' or variant == 'SEM' or variant == 'lumped' or variant == 'KMV' :
            unified_variant = 'lumped'
        elif variant == 'equispaced' or variant == 'equis':
            unified_variant = 'equispaced'
        elif variant == 'DG' or variant == 'discontinuous_galerkin':
            unified_variant = 'DG'
        else:
            warnings.warn(f"Variant of {variant} not accepted.")
        self.method = unified_variant

    def __get_method_from_cell_type(self):
        cell_type = self.cell_type
        variant = self.variant
        dimension = self.dimension
        method = None
        if cell_type == 'triangle':
            if   variant == 'lumped':
                method = 'mass_lumped_triangle'
            elif variant == 'equispaced':
                method = 'CG_triangle'
            elif variant == 'DG':
                method = 'DG_triangle'
        elif cell_type == 'quadrilateral':
            if   variant == 'lumped':
                method = 'spectral_quadrilateral'
            elif variant == 'equispaced':
                method = 'CG_quadrilateral'
            elif variant == 'DG':
                method = 'DG_quadrilateral'

    def __get_method(self):
        dictionary = self.dictionary
        # Checking if method/cell_type + variant specified twice:
        if "method" in dictionary["options"] and ("cell_type" in dictionary["options"]) and ("variant" in dictionary["options"]):
            warnings.warn("Both methods of specifying method and cell_type with variant used. Method specification taking priority.")
        if "method" in dictionary["options"]:
            if dictionary["options"]["method"] != None:
                self.method = dictionary["options"]["method"]
                self.__unify_method_input()
                # For backwards compatibility
                if "variant" in dictionary["options"]:
                    if dictionary["options"]["variant"] == 'spectral' or dictionary["options"]["variant"] == 'GLL' and self.method == 'CG':
                        self.method = 'spectral_quadrilateral'
                
        elif ("cell_type" in dictionary["options"]) and ("variant" in dictionary["options"]):
            self.cell_type = dictionary["options"]["cell_type"]
            self.__unify_cell_type_input()
            self.variant   = dictionary["options"]["variant"]
            self.__unify_variant_input()
            self.__get_method_from_cell_type()
        else:
            raise ValueError("Missing options inputs.")

        
        




