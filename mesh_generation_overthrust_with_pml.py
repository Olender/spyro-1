import meshio
import numpy as np
from mpi4py import MPI

from SeismicMesh import *

comm = MPI.COMM_WORLD

bbox = (-5175.0, 0.0, 0.0, 7500.0, 0.0, 7500.0)
cube = Cube(bbox)
wl = 3.0
freq = 5.0
hmin = 1500.0 / (wl * freq)
print(hmin)

fname = "/home/olender/common_files/velocity_models/overthrust_3D_true_model.hdf5"

nz, nx, ny = 237, 360, 360

edge_length = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    hmax=300,
    wl=wl,
    freq=freq,
    grade=0.35,
    pad_style="edge",
    domain_pad=750.0,
    nz=nz,
    nx=nx,
    ny=ny,
    byte_order="little",
    axes_order=(2, 1, 0),
    dtype="int32",
)

# write_velocity_model(
#     fname,
#     ofname="overthrust_3D_exact_model_reduced_v5",
#     nz=nz,
#     nx=nx,
#     ny=ny,
#     domain_pad=750.0,
#     pad_style="edge",
#     bbox=bbox,
#     byte_order="little",
#     axes_order=(2, 1, 0),
#     dtype="int32",
# )

points, cells = generate_mesh(
    edge_length=edge_length,
    domain=cube,
    verbose=2,
    max_iter=250,
)

points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=12.5,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=15.0,
)
dh_angles = (geometry.calc_dihedral_angles(points, cells) * 180) / np.pi
print(f"Minimum dihedral angle is {np.amin(dh_angles)}")


if comm.rank == 0:
    meshio.write_points_cells(
        "overthrust_3D_exact_model_reduced_v5.msh",
        points / 1000.0,
        [("tetra", cells)],
        file_format="gmsh22",
        binary=False,
    )
       