import firedrake as fire

from ..io.basicio import parallel_print
from . import helpers
from .. import utils


def central_difference(wave):
    """
    Perform central difference time integration for wave propagation.

    Parameters:
    -----------
    wave: Spyro object
        The Wave object containing the necessary data and parameters.

    Returns:
    --------
        tuple:
            A tuple containing the forward solution and the receiver output.
    """
    if wave.sources is not None:
        source_id = wave.sources.current_source
        rhs_forcing = fire.Function(wave.function_space)
    else:
        source_id = 0

    filename, file_extension = wave.forward_output_file.split(".")
    output_filename = filename + "sn" + str(source_id) + "." + file_extension
    if wave.forward_output:
        parallel_print(f"Saving output in: {output_filename}", wave.comm)
    
    output = fire.File(output_filename, comm=wave.comm.comm)
    wave.comm.comm.barrier()

    t = wave.current_time
    nt = int(wave.final_time / wave.dt) + 1  # number of timesteps
    
    usol = [
        fire.Function(wave.function_space, name=wave.get_function_name())
        for t in range(nt)
        if t % wave.gradient_sampling_frequency == 0
    ]
    usol_recv = []
    save_step = 0
    for step in range(nt):
        # Basic way of applying sources
        wave.update_source_expression(t)
        fire.assemble(wave.rhs, tensor=wave.B)

        # More efficient way of applying sources
        if wave.sources is not None:
            rhs_forcing.assign(0.0)
            f = wave.sources.apply_source(rhs_forcing, wave.wavelet[step])
            B0 = wave.B.sub(0)
            B0 += f
        
        wave.solver.solve(wave.next_vstate, wave.B)

        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate

        usol_recv.append(wave.get_receivers_output())

        if step % wave.gradient_sampling_frequency == 0:
            usol[save_step].assign(wave.get_function())
            save_step += 1

        if (step - 1) % wave.output_frequency == 0:
            assert (
                fire.norm(wave.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            if wave.forward_output:
                output.write(wave.get_function(), time=t,
                             name=wave.get_function_name())

            helpers.display_progress(wave.comm, t)

        t = step * float(wave.dt)
    
    wave.current_time = t
    helpers.display_progress(wave.comm, t)

    usol_recv = helpers.fill(
        usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, wave.comm)
    wave.receivers_output = usol_recv

    wave.forward_solution = usol
    wave.forward_solution_receivers = usol_recv

    return usol, usol_recv