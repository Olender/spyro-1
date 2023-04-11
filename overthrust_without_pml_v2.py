from firedrake import File
from firedrake.petsc import PETSc
import time
import numpy as np
import spyro
import os
import psutil

def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2**20)
    return mem

parprint = PETSc.Sys.Print

line1 = spyro.create_transect((0.25, 0.25), (0.25, 5.25), 4)
line2 = spyro.create_transect((1.0, 0.25), (1.0, 5.25), 4)
line3 = spyro.create_transect((2.75, 0.25), (2.75, 5.25), 4)
line4 = spyro.create_transect((3.5, 0.25), (3.25, 5.25), 4)
line5 = spyro.create_transect((4.25, 0.25), (4.25, 5.25), 4)
lines = np.concatenate((line1, line2, line3, line4, line5))

sources = spyro.insert_fixed_value(lines, -0.10, 0)

receivers = spyro.create_2d_grid(0.25, 5.25, 0.25, 5.25, 30)
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
    "Lz": 4.140,  # depth in km - always positive
    "Lx": 6.0,  # width in km - always positive
    "Ly": 6.0,  # thickness in km - always positive
    "meshfile": "meshes/overthrust_3D_exact_model_reduced_no_pmlv1.msh",
    "initmodel": None,
    "truemodel": "velocity_models/overthrust_3D_exact_model_reduced_v5_no_pml.hdf5",
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
model["aut_dif"] = {
    "status": False
}
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 4.00,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
t0 = time.time()
mesh, V = spyro.io.read_mesh(model, comm)
vp = spyro.io.interpolate(model, mesh, V, guess=False)
if comm.ensemble_comm.rank == 0:
    File("true_velocity.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
p, p_r = spyro.solvers.forward(
    model, mesh, comm, vp, sources, wavelet, receivers, output=False
)
tf = time.time()
parprint(f"Total simulation time, for overthrust, without PML = {tf-t0}")
parprint(f"Memory usage = {get_memory_usage()}")
spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
