import firedrake as fire
import warnings

from .acoustic_wave import AcousticWave


class FullWaveformInversion(AcousticWave):
    def __init__(self, dictionary=None, comm=None):
        super().__init__(dictionary=dictionary, comm=comm)
        if self.running_fwi is False:
            warnings.warn("Dictionary FWI options set to not run FWI.")
        self.real_velocity_model = None
        self.real_velocity_model_file = None
        self.real_shot_record = None
        self.guess_shot_record = None
        self.gradient = None
        self.current_iteration = 0
        self.mesh_iteration = 0
        self.iteration_limit = 100
        self.inner_product = 'L2'

    def calculate_misfit(self):
        Wave_obj_guess = AcousticWave(dictionary=self.input_dictionary)
        if Wave_obj_guess.mesh is None and self.guess_mesh is not None:
            Wave_obj_guess.mesh = self.guess_mesh
        if Wave_obj_guess.initial_velocity_model is None:
            Wave_obj_guess.initial_velocity_model = self.guess_velocity_model
        Wave_obj_guess.forward_solve()
        self.guess_shot_record = Wave_obj_guess.forward_solution_receivers
        self.guess_forward_solution = Wave_obj_guess.forward_solution

        return self.real_shot_record - self.guess_shot_record

    def generate_real_shot_record(self):
        Wave_obj_real_velocity = SyntheticRealAcousticWave(dictionary=self.input_dictionary)
        if Wave_obj_real_velocity.mesh is None and self.real_mesh is not None:
            Wave_obj_real_velocity.mesh = self.real_mesh
        if Wave_obj_real_velocity.initial_velocity_model is None:
            Wave_obj_real_velocity.initial_velocity_model = self.real_velocity_model
        Wave_obj_real_velocity.forward_solve()
        self.real_shot_record = Wave_obj_real_velocity.real_shot_record

    def set_smooth_guess_velocity_model(self, real_velocity_model_file=None):
        if real_velocity_model_file is not None:
            real_velocity_model_file = real_velocity_model_file
        else:
            real_velocity_model_file = self.real_velocity_model_file

    def set_real_velocity_model(
        self,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
    ):
        super().set_initial_velocity_model(
            constant=constant,
            conditional=conditional,
            velocity_model_function=velocity_model_function,
            expression=expression,
            new_file=new_file,
            output=output,
        )
        self.real_velocity_model = self.initial_velocity_model

    def set_guess_velocity_model(
        self,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
    ):
        super().set_initial_velocity_model(
            constant=constant,
            conditional=conditional,
            velocity_model_function=velocity_model_function,
            expression=expression,
            new_file=new_file,
            output=output,
        )
        self.guess_velocity_model = self.initial_velocity_model

    def set_real_mesh(
        self,
        user_mesh=None,
        mesh_parameters=None,
    ):
        super().set_mesh(
            user_mesh=user_mesh,
            mesh_parameters=mesh_parameters,
        )
        self.real_mesh = self.get_mesh()

    def set_guess_mesh(
        self,
        user_mesh=None,
        mesh_parameters=None,
    ):
        super().set_mesh(
            user_mesh=user_mesh,
            mesh_parameters=mesh_parameters,
        )
        self.guess_mesh = self.get_mesh()


class SyntheticRealAcousticWave(AcousticWave):
    def __init__(self, dictionary=None, comm=None):
        super().__init__(dictionary=dictionary, comm=comm)
        print("END")

    def forward_solve(self):
        super().forward_solve()
        self.real_shot_record = self.receivers_output
