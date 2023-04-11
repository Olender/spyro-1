import meshio
import numpy as np
from mpi4py import MPI

from SeismicMesh import *

comm = MPI.COMM_WORLD

bbox = (-4140.0, 0.0, 0.0, 6000.0, 0.0, 6000.0)
cube = Cube(bbox)
wl = 3.0
freq = 5.0
hmin = 1500.0 / (wl * freq)
print(hmin)

fname = "overthrust_3D_exact_model_reduced_v5.bin"

nz, nx, ny = 207, 300, 300

edge_length = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    hmax=300,
    wl=wl,
    freq=freq,
    grade=0.35,
    nz=nz,
    nx=nx,
    ny=ny,
    byte_order="little",
    axes_order=(2, 1, 0),
    dtype="int32",
)

write_velocity_model(
    fname,
    ofname="overthrust_3D_exact_model_reduced_v5_no_pml",
    nz=nz,
    nx=nx,
    ny=ny,
    bbox=bbox,
    byte_order="little",
    axes_order=(2, 1, 0),
    dtype="int32",
)

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
   min_dh_angle_bound=3.0,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=5.0,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=7.0,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=8.0,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=9.0,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=9.3,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=9.5,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=9.7,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=10.0,
   max_iter=100,
)
dh_angles = (geometry.calc_dihedral_angles(points, cells) * 180) / np.pi
print(f"Minimum dihedral angle is {np.amin(dh_angles)}")
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=10.3,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=10.5,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=10.7,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=10.8,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=10.9,
   max_iter=100,
)
points, cells = sliver_removal(
   points=points,
   edge_length=edge_length,
   domain=cube,
   verbose=2,
   min_dh_angle_bound=11.0,
   max_iter=100,
)
dh_angles = (geometry.calc_dihedral_angles(points, cells) * 180) / np.pi
print(f"Minimum dihedral angle is {np.amin(dh_angles)}")
# points, cells = sliver_removal(
#    points=points,
#    edge_length=edge_length,
#    domain=cube,
#    verbose=2,
#    min_dh_angle_bound=11.2,
#    max_iter=100,
# )
# points, cells = sliver_removal(
#    points=points,
#    edge_length=edge_length,
#    domain=cube,
#    verbose=2,
#    min_dh_angle_bound=11.4,
#    max_iter=100,
# )
# points, cells = sliver_removal(
#    points=points,
#    edge_length=edge_length,
#    domain=cube,
#    verbose=2,
#    min_dh_angle_bound=11.5,
#    max_iter=100,
# )
# points, cells = sliver_removal(
#    points=points,
#    edge_length=edge_length,
#    domain=cube,
#    verbose=2,
#    min_dh_angle_bound=11.7,
#    max_iter=100,
# )
# points, cells = sliver_removal(
#    points=points,
#    edge_length=edge_length,
#    domain=cube,
#    verbose=2,
#    min_dh_angle_bound=11.9,
#    max_iter=100,
# )
# points, cells = sliver_removal(
#    points=points,
#    edge_length=edge_length,
#    domain=cube,
#    verbose=2,
#    min_dh_angle_bound=12.0,
#    max_iter=100,
# )

dh_angles = (geometry.calc_dihedral_angles(points, cells) * 180) / np.pi
print(f"Minimum dihedral angle is {np.amin(dh_angles)}")


if comm.rank == 0:
    meshio.write_points_cells(
        "overthrust_3D_exact_model_reduced_no_pmlv1.msh",
        points / 1000.0,
        [("tetra", cells)],
        file_format="gmsh22",
        binary=False,
    )
       
