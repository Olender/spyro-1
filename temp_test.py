import spyro


def test_cpw_calc():
    grid_point_calculator_parameters = {
        # Experiment parameters
        # Here we define the frequency of the Ricker wavelet source
        "source_frequency": 5.0,
        # The minimum velocity present in the domain.
        "minimum_velocity_in_the_domain": 1.5,
        # if an homogeneous test case is used this velocity will be defined in
        # the whole domain.
        # Either or heterogeneous. If heterogeneous is
        "velocity_profile_type": "homogeneous",
        # chosen be careful to have the desired velocity model below.
        "velocity_model_file_name": None,
        # FEM to evaluate such as `KMV` or `spectral`
        # (GLL nodes on quads and hexas)
        "FEM_method_to_evaluate": "mass_lumped_triangle",
        "dimension": 2,  # Domain dimension. Either 2 or 3.
        # Either near or line. Near defines a receiver grid near to the source,
        "receiver_setup": "near",
        # line defines a line of point receivers with pre-established near and far
        # offsets.
        # Line search parameters
        "load_reference": True,
        "save_reference": False,
        "reference_degree": None,  # Degree to use in the reference case (int)
        # grid point density to use in the reference case (float)
        "C_reference": None,
        "desired_degree": 4,  # degree we are calculating G for. (int)
        "C_initial": 2.46,  # Initial G for line search (float)
        "accepted_error_threshold": 0.05,
        "C_accuracy": 0.01,
    }

    # Cpw_calc = spyro.tools.Meshing_parameter_calculator(grid_point_calculator_parameters)
    # cpw = Cpw_calc.find_minimum()
    # print(f"C calculated = {cpw}")

    # print("END")
    pass


if __name__ == "__main__":
    test_cpw_calc()
