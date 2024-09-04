import firedrake as fire
import firedrake.adjoint as fire_ad
import spyro


# --- Basid setup to run a forward simulation with AD --- #
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or mass_lumped_triangle
    "quadrature": "KMV",  # Equi or mass_lumped_triangle
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5,  # regularization parameter
}

model["parallelism"] = {
    # options:
    # `shots_parallelism`. Shots parallelism.
    # None - no shots parallelism.
    "type": "shots_parallelism",
    "num_spacial_cores": 1,  # Number of cores to use in the spatial
                             # parallelism.
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC
    "outer_bc": "non-reflective",  # none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "source_pos": spyro.create_transect((0.2, 0.15), (0.8, 0.15), 3),
    "frequency": 7.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((0.2, 0.2), (0.8, 0.2), 10),
}
model["aut_dif"] = {
    "status": True,
}

model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 0.6,  # Final time for event (for test 7)
    "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}


def make_c_camembert(c_guess=False, plot_c=False):
    """Acoustic velocity model"""
    x, z = fire.SpatialCoordinate(mesh)
    if c_guess:
        c = fire.Function(V).interpolate(1.5 + 0.0 * x)
    else:
        c = fire.Function(V).interpolate(
            2.5
            + 1 * fire.tanh(100 * (0.125 - fire.sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2)))
        )
    if plot_c:
        outfile = fire.VTKFile("acoustic_cp.pvd")
        outfile.write(c)
    return c


def forward(
        c, compute_functional=False, true_data_receivers=None, annotate=False
):
    if annotate:
        fire_ad.continue_annotation()
    if model["parallelism"]["type"] is None:
        outfile = fire.VTKFile("solution.pvd")
        receiver_data = []
        J = 0.0
        for sn in range(len(model["acquisition"]["source_pos"])):
            rec_data, J_val = forward_solver.execute_acoustic(c, sn, wavelet)
            receiver_data.append(rec_data)
            J += J_val
            sol = forward_solver.solution
            outfile.write(sol)

    else:
        # source_number based on the ensemble.ensemble_comm.rank
        source_number = my_ensemble.ensemble_comm.rank
        receiver_data, J = forward_solver.execute_acoustic(
            c, source_number, wavelet,
            compute_functional=compute_functional,
            true_data_receivers=true_data_receivers
        )
        sol = forward_solver.solution
        fire.VTKFile(
            "solution_" + str(source_number) + ".pvd", comm=my_ensemble.comm
            ).write(sol)

    return receiver_data, J


# Use emsemble parallelism.
M = model["parallelism"]["num_spacial_cores"]
my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
mesh = fire.UnitSquareMesh(50, 50, comm=my_ensemble.comm)
element = fire.FiniteElement(
    model["opts"]["method"], mesh.ufl_cell(), degree=model["opts"]["degree"],
    variant=model["opts"]["quadrature"]
    )
V = fire.FunctionSpace(mesh, element)


forward_solver = spyro.solvers.forward_ad.ForwardSolver(model, mesh, V)
c_true = make_c_camembert()
# Ricker wavelet
wavelet = spyro.full_ricker_wavelet(
    model["timeaxis"]["dt"], model["timeaxis"]["tf"],
    model["acquisition"]["frequency"],
)

true_rec, _ = forward(c_true)

# --- FWI with AD --- #
c_guess = make_c_camembert(c_guess=True)
guess_rec, J = forward(
    c_guess, compute_functional=True, true_data_receivers=true_rec,
    annotate=True
    )

# :class:`~.EnsembleReducedFunctional` is employed to recompute in
# parallel the functional and its gradient associated with the multiple sources
# (3 in this case).
J_hat = fire_ad.EnsembleReducedFunctional(
    J, fire_ad.Control(c_guess), my_ensemble)
fire_ad.taylor_test(J_hat, c_guess, fire.Function(V).assign(1.0))
c_optimised = fire_ad.minimize(J_hat, method="L-BFGS-B",
                               options={"disp": True, "maxiter": 10},
                               bounds=(1.5, 3.5),
                               derivative_options={"riesz_representation": 'l2'})