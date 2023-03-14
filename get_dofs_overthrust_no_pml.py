from firedrake import File
import numpy as np
import spyro


line1 = spyro.create_transect((0.25, 0.25), (0.25, 7.25), 4)
line2 = spyro.create_transect((2.0, 0.25), (2.0, 7.25), 4)
line3 = spyro.create_transect((3.75, 0.25), (3.75, 7.25), 4)
line4 = spyro.create_transect((5.5, 0.25), (5.25, 7.25), 4)
line5 = spyro.create_transect((7.25, 0.25), (7.25, 7.25), 4)
lines = np.concatenate((line1, line2, line3, line4, line5))

sources = spyro.insert_fixed_value(lines, -0.10, 0)

receivers = spyro.create_2d_grid(0.25, 7.25, 0.25, 7.25, 30)
receivers = spyro.insert_fixed_value(receivers, -0.15, 0)

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 3,  # p order
    "dimension": 3,  # dimension
}
model["parallelism"] = {"type": "automatic"}  # automatic",
model["mesh"] = {
    "Lz": 5.175,  # depth in km - always positive
    "Lx": 7.50,  # width in km - always positive
    "Ly": 7.50,  # thickness in km - always positive
    "meshfile": "meshes/overthrust_no_pml.msh",
    "initmodel": None,
    "truemodel": "velocity_models/overthrust_no_pml.hdf5",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 6.0,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.0,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": len(sources),
    "source_pos": sources,
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": len(receivers),
    "receiver_locations": receivers,
}
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 4.00,  # Final time for event
    "dt": 0.00075,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)
print(f"Dofs count = {V.dim()} without PML for overthrust")
