from classes.particle_filter_multiple import MultiObjectParticleFilter
from classes.observation import TransitionModel, ObservationModel
from classes.simulator import create_ground_truth, generate_random_balls, create_ground_truth_n_balls
from classes.evaluator import get_stats

import numpy as np
np.random.seed()

step_size = 0.05
num_steps = 120

num_particles = 1000
init_generator = "Sobol"


STATE_BOUNDS = [
    (0.0, 50.0),   # x bounds
    (0.0, 50.0),   # y bounds
    (-30.0, 30.0),   # vx bounds
    (0.0, 40.0)    # vy bounds
]

def create_test_scenario(
        true_states,
        dropout_start, 
        dropout_end, 
        process_noise,
        measurement_noise, 
        ):
    transition_model = TransitionModel(delta_t=step_size, process_noise=process_noise)
    observation_model = ObservationModel("Gaussian", measurement_noise=measurement_noise)
    true_trajectory, observations = create_ground_truth_n_balls(
        num_steps, dropout_start, dropout_end, true_states, transition_model, observation_model
    )
    return true_trajectory, observations, transition_model, observation_model

def run_one_test(num_particles=1000, 
                 n_objects=None,
                 state_bounds=STATE_BOUNDS,
                 transition_model=None,
                 observation_model=None,
                 init_generator="Sobol",
                 roughening_noise=0.0,
                 observations=None,
                 true_trajectory=None):
    
    pf = MultiObjectParticleFilter(
        num_particles=num_particles,
        n_balls=n_objects,
        state_bounds=state_bounds,
        transition_model=transition_model,
        observation_model=observation_model,
        init_generator=init_generator,
        roughening_noise=roughening_noise,
    )

    history = pf.run(observations)
    
    return get_stats(true_trajectory, observations, history, num_steps)
