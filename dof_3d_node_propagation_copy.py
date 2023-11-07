import spyro
import numpy as np
import matplotlib.pyplot as plt


dt = 0.00001
# dt = float(sys.argv[1])

final_time = 1.0
dx = 0.006546536707079771
# dx = 0.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 3,  # p order
    "dimension": 3,  # dimension
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
    "Lz": 3.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 1.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.5-dx, 0.5+dx, 0.5+dx)],
    "frequency": 5.0,
    "delay": 1.5,
    "receiver_locations": [(-2.0-dx, 0.5+dx, 0.5+dx)],
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": dt,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 3000,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 3000,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_3d_output3by3by3copy.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}


Wave_obj = spyro.AcousticWave(dictionary=dictionary)
Wave_obj.set_mesh(mesh_parameters={"dx": 0.02, "periodic": True})

Wave_obj.set_initial_velocity_model(constant=1.5)
Wave_obj.forward_solve()

time = np.linspace(0.0, final_time, int(final_time/dt)+1)

rec_out = Wave_obj.receivers_output
np.save("dofs_3D_quads_p4_dt"+str(dt)+".npy", rec_out)

# plt.plot(time, Wave_obj.receivers_output)
# plt.show()

print("END")