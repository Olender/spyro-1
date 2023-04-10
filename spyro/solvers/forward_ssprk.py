from firedrake import *
from firedrake.assemble import create_assembly_callable

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_forward
from . import helpers

# Note this turns off non-fatal warnings
set_log_level(ERROR)


@ensemble_forward
def forward(
    model,
    mesh,
    comm,
    c,
    excitations,
    wavelet,
    receivers,
    source_num=0,
    output=False,
):
    """Secord-order in time fully-explicit scheme
    with implementation of a Perfectly Matched Layer (PML) using
    CG FEM with or without higher order mass lumping (KMV type elements).

    Parameters
    ----------
    model: Python `dictionary`
        Contains model options and parameters
    mesh: Firedrake.mesh object
        The 2D/3D triangular mesh
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
       c: Firedrake.Function
        The velocity model interpolated onto the mesh.
    excitations: A list Firedrake.Functions
    wavelet: array-like
        Time series data that's injected at the source location.
    receivers: A :class:`spyro.Receivers` object.
        Contains the receiver locations and sparse interpolation methods.
    source_num: `int`, optional
        The source number you wish to simulate
    output: `boolean`, optional
        Whether or not to write results to pvd files.

    Returns
    -------
    usol: list of Firedrake.Functions
        The full field solution at `fspool` timesteps
    usol_recv: array-like
        The solution interpolated to the receivers at all timesteps

    """

    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    excitations.current_source = source_num

    nt = int(tf / dt)  # number of timesteps

    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif (
        method == "CG"
        and mesh.ufl_cell() != quadrilateral
        and mesh.ufl_cell() != hexahedron
    ):
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    elif method == "CG" and (
        mesh.ufl_cell() == quadrilateral or mesh.ufl_cell() == hexahedron
    ):
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    else:
        raise ValueError("method is not yet supported")


    element = space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    qr_x, _, _ = quadrature.quadrature_rules(V)

    W =  V* V
    dwdt_trial, dpdt_trial = TrialFunctions(W)
    q1, q2 = TestFunctions(W)

    w_n, p_n= Function(W).split()

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = Function(V)
    u_n = Function(V)
    u_np1 = Function(V)

    if output:
        outfile = helpers.create_output_file("forward.pvd", comm, source_num)

    t = 0.0

    # -------------------------------------------------------
    m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dx(rule=qr_x)
    a = c * c * dot(grad(u_n), grad(v)) * dx(rule=qr_x)  # explicit

    FF = m1 + a

    X = Function(V)
    B = Function(V)

    lhs_ = lhs(FF)
    rhs_ = rhs(FF)

    A = assemble(lhs_, mat_type="matfree")
    solver = LinearSolver(A, solver_parameters=params)

    usol = [Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
    usol_recv = []
    save_step = 0

    assembly_callable = create_assembly_callable(rhs_, tensor=B)

    rhs_forcing = Function(V)

    for step in range(nt):
        rhs_forcing.assign(0.0)
        assembly_callable()
        f = excitations.apply_source(rhs_forcing, wavelet[step])
        B0 = B.sub(0)
        B0 += f
        solver.solve(X, B)

        u_np1.assign(X)

        usol_recv.append(receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

        if step % fspool == 0:
            usol[save_step].assign(u_np1)
            save_step += 1

        if step % nspool == 0:
            assert (
                norm(u_n) < 1
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if output:
                outfile.write(u_n, time=t, name="Pressure")
            if t > 0:
                helpers.display_progress(comm, t)

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
    usol_recv = utils.communicate(usol_recv, comm)

    return usol, usol_recv
