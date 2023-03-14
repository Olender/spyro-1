from firedrake import File
import firedrake as fire
import spyro

def saving_source_and_receiver_location_in_csv(model):
    file_name = 'sources.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for source in model['acquisition']['source_pos']:
        z, x = source
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    file_name = 'receivers.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for receiver in model['acquisition']['receiver_locations']:
        z, x = receiver
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    return None

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "automatic",
}
model["mesh"] = {
    "Lz": 4.0,  # depth in km - always positive
    "Lx": 16.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": None,
    "initmodel": None,
    "truemodel": None,
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 1.0,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 1.0,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "source_pos": spyro.create_transect((-0.1, 1.0), (-0.1, 15.0), 4),
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 500,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 500),
}
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 5.00,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
saving_source_and_receiver_location_in_csv(model) #Open txt in paraview and use tabletopoints filter
Lz = model["mesh"]["Lz"] + model["BCs"]["lz"]
Lx = model["mesh"]["Lx"] + 2*model["BCs"]["lx"]
mesh = fire.RectangleMesh(25*2,90*2,Lz,Lx, comm=comm.comm)
mesh.coordinates.dat.data[:,0] *= -1.0
mesh.coordinates.dat.data[:,1] -= model["BCs"]["lx"]
element = spyro.domains.space.FE_method(mesh, "KMV", 4)
V = fire.FunctionSpace(mesh, element)

z, x = fire.SpatialCoordinate(mesh)
Vpml = fire.FunctionSpace(mesh, "DG", 0)
vp = fire.Function(Vpml)
pml = fire.conditional(z < -model["mesh"]["Lz"], 3.0, 1.0)
pml = fire.conditional(x < 0.0, 3.0, pml)
pml = fire.conditional(x > model["mesh"]["Lx"], 3.0, pml)
vp.interpolate(pml)
if comm.ensemble_comm.rank == 0:
    File("pml.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output = True)
spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
spyro.io.save_shots(model, comm, p_r)
