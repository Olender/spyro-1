from __future__ import with_statement

import os
import pickle

import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import segyio
from .. import domains


def ensemble_save(func):
    """Decorator for read and write shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[1]
        custom_file_name = kwargs.get("file_name")
        for snum in range(num):
            if is_owner(_comm, snum) and _comm.comm.rank == 0:
                if custom_file_name is None:
                    func(
                        *args,
                        **dict(
                            kwargs,
                            file_name="shots/shot_record_"
                            + str(snum + 1)
                            + ".dat",
                        )
                    )
                else:
                    func(*args, **dict(kwargs, file_name=custom_file_name))

    return wrapper


def ensemble_load(func):
    """Decorator for read and write shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[1]
        custom_file_name = kwargs.get("file_name")
        for snum in range(num):
            if is_owner(_comm, snum):
                if custom_file_name is None:
                    values = func(
                        *args,
                        **dict(
                            kwargs,
                            file_name="shots/shot_record_"
                            + str(snum + 1)
                            + ".dat",
                        )
                    )
                else:
                    values = func(
                        *args, **dict(kwargs, file_name=custom_file_name)
                    )
                return values

    return wrapper


def ensemble_plot(func):
    """Decorator for `plot_shots` to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[1]
        for snum in range(num):
            if is_owner(_comm, snum) and _comm.comm.rank == 0:
                func(*args, **dict(kwargs, file_name=str(snum + 1)))

    return wrapper


def ensemble_forward(func):
    """Decorator for forward to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                u, u_r = func(*args, **dict(kwargs, source_num=snum))
                return u, u_r

    return wrapper


def ensemble_propagator(func):
    """Decorator for forward to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        num = args[0].number_of_sources
        _comm = args[0].comm
        for snum in range(num):
            if is_owner(_comm, snum):
                u, u_r = func(*args, **dict(kwargs, source_num=snum))
                return u, u_r

    return wrapper


def ensemble_forward_ad(func):
    """Decorator for forward to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        fwi = kwargs.get("fwi")
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                if fwi:
                    u_r, J = func(*args, **dict(kwargs, source_num=snum))
                    return u_r, J
                else:
                    u_r = func(*args, **dict(kwargs, source_num=snum))

    return wrapper


def ensemble_forward_elastic_waves(func):
    """Decorator for forward elastic waves to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                u, uz_r, ux_r, uy_r = func(
                    *args, **dict(kwargs, source_num=snum)
                )
                return u, uz_r, ux_r, uy_r

    return wrapper


def ensemble_gradient(func):
    """Decorator for gradient to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        save_adjoint = kwargs.get("save_adjoint")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                if save_adjoint:
                    grad, u_adj = func(*args, **kwargs)
                    return grad, u_adj
                else:
                    grad = func(*args, **kwargs)
                    return grad

    return wrapper


def ensemble_gradient_elastic_waves(func):
    """Decorator for gradient (elastic waves) to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        save_adjoint = kwargs.get("save_adjoint")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                if save_adjoint:
                    grad_lambda, grad_mu, u_adj = func(*args, **kwargs)
                    return grad_lambda, grad_mu, u_adj
                else:
                    grad_lambda, grad_mu = func(*args, **kwargs)
                    return grad_lambda, grad_mu

    return wrapper


def write_function_to_grid(function, V, grid_spacing):
    """Interpolate a Firedrake function to a structured grid

    Parameters
    ----------
    function : firedrake.Function
        Function to interpolate
    V : firedrake.FunctionSpace
        Function space of function
    grid_spacing : float
        Spacing of grid points in metres

    Returns
    -------
    xi : numpy.ndarray
        x coordinates of grid points
    yi : numpy.ndarray
        y coordinates of grid points
    zi : numpy.ndarray
        Interpolated values on grid points
    """
    # get DoF coordinates
    m = V.ufl_domain()
    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.interpolate(m.coordinates, W)
    x, y = coords.dat.data[:, 0], coords.dat.data[:, 1]

    # add buffer to avoid NaN when calling griddata
    min_x = np.amin(x) + 0.01
    max_x = np.amax(x) - 0.01
    min_y = np.amin(y) + 0.01
    max_y = np.amax(y) - 0.01

    z = function.dat.data[:] * 1000.0  # convert from km/s to m/s

    # target grid to interpolate to
    xi = np.arange(min_x, max_x, grid_spacing)
    yi = np.arange(min_y, max_y, grid_spacing)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method="linear")

    return xi, yi, zi


def create_segy(velocity, filename):
    """Write the velocity data into a segy file named filename

    Parameters
    ----------
    velocity:
        Firedrake function representing the values of the velocity model to save
    filename: str
        Name of the segy file to save

    Returns
    -------
    None
    """
    spec = segyio.spec()

    velocity = np.flipud(velocity.T)

    spec.sorting = 2  # not sure what this means
    spec.format = 1  # not sure what this means
    spec.samples = range(velocity.shape[0])
    spec.ilines = range(velocity.shape[1])
    spec.xlines = range(velocity.shape[0])

    assert np.sum(np.isnan(velocity[:])) == 0

    with segyio.create(filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = velocity[:, tr]


@ensemble_save
def save_shots(model, comm, array, file_name=None):
    """Save a `numpy.ndarray` to a `pickle`.

    Parameters
    ----------
    model:

    comm:

    array: `numpy.ndarray`
        The data to save a pickle (e.g., a shot)
    filename: str, optional by default shot_number_#.dat
        The filename to save the data as a `pickle`

    Returns
    -------
    None

    """
    with open(file_name, "wb") as f:
        pickle.dump(array, f)
    return None


@ensemble_load
def load_shots(model, comm, file_name=None):
    """Load a `pickle` to a `numpy.ndarray`.

    Parameters
    ----------
    filename: str, optional by default shot_number_#.dat
        The filename to save the data as a `pickle`

    Returns
    -------
    array: `numpy.ndarray`
        The data

    """

    with open(file_name, "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)
    return array


def is_owner(ens_comm, rank):
    """Distribute shots between processors in using a modulus operator

    Parameters
    ----------
    ens_comm: Firedrake.ensemble_communicator
        An ensemble communicator
    rank: int
        The rank of the core

    Returns
    -------
    boolean
        `True` if `rank` owns this shot

    """
    return ens_comm.ensemble_comm.rank == (rank % ens_comm.ensemble_comm.size)


def _check_units(c):
    if min(c.dat.data[:]) > 100.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def interpolate(Model, fname, V):
    """Read and interpolate a seismic velocity model stored
    in a HDF5 file onto the nodes of a finite element space.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    mesh: Firedrake.mesh object
        A mesh object read in by Firedrake.
    V: Firedrake.FunctionSpace object
        The space of the finite elements.
    guess: boolean, optinal
        Is it a guess model or a `exact` model?

    Returns
    -------
    c: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the finite elements.

    """
    sd = V.mesh().geometric_dimension()
    m = V.ufl_domain()
    if Model.abc_status:
        minz = -Model.length_z - Model.abc_lz
        maxz = 0.0
        minx = 0.0 - Model.abc_lx
        maxx = Model.length_x + Model.abc_lx
        miny = 0.0 - Model.abc_ly
        maxy = Model.length_y + Model.abc_ly
    else:
        minz = -Model.length_z
        maxz = 0.0
        minx = 0.0
        maxx = Model.length_x
        miny = 0.0
        maxy = Model.length_y

    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.interpolate(m.coordinates, W)
    # (z,x) or (z,x,y)
    if sd == 2:
        qp_z, qp_x = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif sd == 3:
        qp_z, qp_x, qp_y = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise NotImplementedError

    with h5py.File(fname, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])

        if sd == 2:
            nrow, ncol = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)

            # make sure no out-of-bounds
            qp_z2 = [
                minz if z < minz else maxz if z > maxz else z for z in qp_z
            ]
            qp_x2 = [
                minx if x < minx else maxx if x > maxx else x for x in qp_x
            ]

            interpolant = RegularGridInterpolator((z, x), Z)
            tmp = interpolant((qp_z2, qp_x2))
        elif sd == 3:
            nrow, ncol, ncol2 = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)
            y = np.linspace(miny, maxy, ncol2)

            # make sure no out-of-bounds
            qp_z2 = [
                minz if z < minz else maxz if z > maxz else z for z in qp_z
            ]
            qp_x2 = [
                minx if x < minx else maxx if x > maxx else x for x in qp_x
            ]
            qp_y2 = [
                miny if y < miny else maxy if y > maxy else y for y in qp_y
            ]

            interpolant = RegularGridInterpolator((z, x, y), Z)
            tmp = interpolant((qp_z2, qp_x2, qp_y2))

    c = fire.Function(V)
    c.dat.data[:] = tmp
    c = _check_units(c)
    return c


def read_mesh(model_parameters):
    """Reads in an external mesh and scatters it between cores.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    ens_comm: Firedrake.ensemble_communicator
        An ensemble communicator

    Returns
    -------
    mesh: Firedrake.Mesh object
        The distributed mesh across `ens_comm`.
    """

    method = model_parameters.method
    ens_comm = model_parameters.comm

    num_sources = model_parameters.number_of_sources
    mshname = model_parameters.mesh_file

    if method == "CG_triangle" or method == "mass_lumped_triangle":
        mesh = fire.Mesh(
            mshname,
            comm=ens_comm.comm,
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    else:
        mesh = fire.Mesh(mshname, comm=ens_comm.comm)
    if ens_comm.comm.rank == 0 and ens_comm.ensemble_comm.rank == 0:
        print(
            "INFO: Distributing %d shot(s) across %d core(s). Each shot is using %d cores"
            % (
                num_sources,
                fire.COMM_WORLD.size,
                fire.COMM_WORLD.size / ens_comm.ensemble_comm.size,
            ),
            flush=True,
        )
    print(
        "  rank %d on ensemble %d owns %d elements and can access %d vertices"
        % (
            mesh.comm.rank,
            ens_comm.ensemble_comm.rank,
            mesh.num_cells(),
            mesh.num_vertices(),
        ),
        flush=True,
    )

    return mesh


def parallel_print(string, comm):
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(string, flush=True)