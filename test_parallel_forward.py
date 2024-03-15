# from mpi4py.MPI import COMM_WORLD
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()

import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from firedrake import File
import firedrake as fire
import spyro


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def test_forward_5_shots():
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
        "source_locations": spyro.create_transect((-1.1, 1.2), (-1.1, 1.8), 3),
        "frequency": 5.0,
        "delay": 0.2,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-1.2, 1.2), (-1.2, 1.8), 3),
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

    sources = spyro.create_transect((-1.1, 1.2), (-1.1, 1.8), 3)
    for source in sources:
        dictionary["acquisition"]["source_locations"] = source
        Wave_obj = spyro.AcousticWave(dictionary=dictionary)
        comm = Wave_obj.comm
        Wave_obj.set_mesh(mesh_parameters={"dx": 0.1})

        Wave_obj.set_initial_velocity_model(constant=1.5)

        Wave_obj.forward_solve()
        analytical = np.load("analytical_solution_parallel_test.npy")
        numerical_p = Wave_obj.receivers_output
        numerical_p = numerical_p.flatten()

        time_vector = np.linspace(0.0, 1.0, int(1.0 / Wave_obj.dt) + 1)
        nt = len(time_vector)
        error = error_calc(numerical_p, analytical_p, nt)
        print("Error = {:.4e}".format(error))


print("END")