import spyro
from firedrake import RectangleMesh, conditional, UnitSquareMesh, Function, FunctionSpace, File
from spyro.habc import HABC
import firedrake as fire
import numpy as np

from spyro.io.model_parameters import Model_parameters

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "user_mesh": None,
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.3)],
    "frequency": 5.0,
    "delay": 1.5,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 0.9), 20
    ),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output" : True,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

Model = Model_parameters(dictionary=dictionary)
n = 80
# user_mesh = UnitSquareMesh(n, n)
h_min = 1/n
user_mesh = UnitSquareMesh(n, n, diagonal="crossed")
# h_min = 1/n*np.sqrt(2)/2
print(h_min)
user_mesh.coordinates.dat.data[:, 0] *= -1.0

Model.set_mesh(user_mesh=user_mesh)

Wave_no_habc = spyro.AcousticWave(model_parameters=Model)

x, y = Wave_no_habc.get_spatial_coordinates()

Wave_no_habc.set_initial_velocity_model(conditional=conditional(x < -0.5, 1.5, 3.0))
Wave_no_habc._get_initial_velocity_model()
Wave_no_habc.c = Wave_no_habc.initial_velocity_model

habc = HABC(Wave_no_habc, h_min=h_min)
mesh = habc.get_mesh_with_pad()

V = FunctionSpace(mesh,"CG",1)
u = Function(V)
File("mesh.pvd").write(u)


Model_new = Model_parameters(dictionary=dictionary)
Model_new.set_mesh(user_mesh=mesh)
Wave = spyro.AcousticWave(model_parameters=Model)

x, y = Wave.get_spatial_coordinates()
Wave.set_initial_velocity_model(conditional=conditional(x < -0.5, 1.5, 3.0))
Wave._get_initial_velocity_model()
Wave.c = Wave.initial_velocity_model
# Wave.forward_solve()


print("END")


velmat.append([0.00, 0.00, 0.35, 0.10, 2.9])
velmat.append([0.00, 0.10, 0.25, 0.30, 2.9])
velmat.append([0.00, 0.30, 0.25, 0.70, 2.0])
velmat.append([0.00, 0.70, 0.10, 1.00, 3.7])
velmat.append([0.10, 0.70, 0.30, 0.90, 3.7])
velmat.append([0.25, 0.10, 0.75, 0.30, 2.5])
velmat.append([0.25, 0.30, 0.40, 0.70, 2.5])
velmat.append([0.35, 0.00, 0.70, 0.10, 2.1])
velmat.append([0.70, 0.00, 0.90, 0.10, 3.4])
velmat.append([0.80, 0.10, 0.90, 0.35, 3.4])
velmat.append([0.90, 0.00, 1.00, 0.20, 3.4])
velmat.append([0.90, 0.20, 1.00, 0.65, 2.6])
velmat.append([0.75, 0.10, 0.80, 0.50, 4.0])
velmat.append([0.80, 0.35, 0.90, 0.80, 4.0])
#
step = 0.001  # 0.01
num = int((0.75 - 0.4) / step) + 1
x1 = np.linspace(start=0.4, stop=0.75, num=num, endpoint=False)
y1 = 0.3 * np.ones_like(x1)
x2 = step + x1
y2 = np.linspace(start=0.3 + step, stop=0.65, num=num, endpoint=True)
cm = 3.3 * np.ones_like(x1) # Propagation speed in [km/s]
d1 = np.stack((x1, y1, x2, y2, cm), axis=-1).round(8).tolist()
velmat += [d1[i] for i in range(len(d1))]
num = int((0.8 - 0.75) / step)
x1 = np.linspace(start=0.75, stop=0.8, num=num, endpoint=False)
y1 = 0.5 * np.ones_like(x1)
x2 = step + x1
y2 = np.linspace(start=0.65 + step, stop=0.7, num=num, endpoint=True)
cm = 3.3 * np.ones_like(x1) # Propagation speed in [km/s]
d2 = np.stack((x1, y1, x2, y2, cm), axis=-1).round(8).tolist()
velmat += [d2[i] for i in range(len(d2))]
#
velmat.append([0.85, 0.80, 0.90, 0.95, 3.6])
velmat.append([0.90, 0.65, 1.00, 1.00, 3.6])
velmat.append([0.00, 0.00, 0.00, 0.00, 1.5])  # Remaining domain
# Source definition in [m/ms^2] = [kN/g] 
possou = [0.35, 0.75]
# Source in [m/ms^2] = [kN/g]
Fsou = 1 / 2500
# Wavelenght factor for domain
F_wl = 16
# Domain aspect ratio (Lx/Ly)
AspRatio = 2
# Amplification domain factor
if CamComp:
    F_Inf = 1 + 1.75 / AspRatio
else:
    # F_Inf = 1 + 3.5/AspRatio # For 4.4 s
    F_Inf = 1 + 3 / AspRatio  # For 4.0 s
# Courant Number for F_R = 5 for ensuring f0
Courant = 0.16  # Cou_Max = 0.167

valdef = c_dist[-1][-1]
c_array = valdef*np.ones_like(c.vector().get_local())
x1 = [pmlx + c_dist[i][0]*Lx for i in range(len(c_dist) - 1)]
x2 = [pmlx + c_dist[i][2]*Lx for i in range(len(c_dist) - 1)]
y1 = [pmly + c_dist[i][1]*Ly for i in range(len(c_dist) - 1)]
y2 = [pmly + c_dist[i][3]*Ly for i in range(len(c_dist) - 1)]

for dof, coord in enumerate(c_coords):
    xc = coord[0]
    yc = coord[1]
    valvel = [c_dist[i][4] for i in range(
        len(c_dist)-1) if xc >= x1[i] and xc <= x2[i]
        and yc >= y1[i] and yc <= y2[i]]
    if len(valvel) > 0:
        c_array[dof] = valvel[0]
4.8kmx2.4