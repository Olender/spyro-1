import matplotlib.pyplot as plt
import numpy as np
import firedrake as fire
import spyro


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.0, 1.2)],
    "frequency": 5.0,
    "delay": 0.3,
    "delay_type": "time",
    "receiver_locations": [(-1.2, 1.2)],
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,
}
dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

Wave_obj = spyro.AcousticWave(dictionary=dictionary)
analytical_p = spyro.utils.nodal_homogeneous_analytical(
        Wave_obj, 0.2, 1.5
    )
Wave_obj.set_mesh(mesh_parameters={"dx": 0.1})
Wave_obj.set_initial_velocity_model(constant=1.5)
Wave_obj.forward_solve()


time_vector = np.linspace(0.0, 1.0, int(1.0/0.001)+1)

print("END")
