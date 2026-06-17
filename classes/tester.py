# Runs tests for the Particle Filter for edge cases and prints out the results (No Visualization)
from simulator import *
from particle_filter import ParticleFilter
from observation import TransitionModel, ObservationModel
from simulator import create_ground_truth, generate_random_balls, create_ground_truth_n_balls
from plotting import plot_sim_n_balls_point_prediction, plot_particles_at_time, animate_particle_filter
from evaluator import print_stats
import numpy as np

# Test Cases: Random 1 Ball, Random 3 Balls, Random 10 Balls, Low Measurement Noise, High Measurement Noise,
# Similar Trajectories, Sensor Dropout

# Hyperparameters to Test
step_size = 0.1
num_steps = 60
dropout_start = 20
dropout_end = 30
process_noise = 5
measurement_noise = 25
roughening_noise = 0
trust_factor = 0.5
num_particles = 4000
init_generator = "Sobol"


state_bounds = [
    (0.0, 50.0),   # x bounds
    (0.0, 50.0),   # y bounds
    (-30.0, 30.0),   # vx bounds
    (0.0, 40.0)    # vy bounds
]

# Random 1 Ball
random_1_ball = generate_random_balls(1, state_bounds)

# Random 3 Balls
random_3_balls = generate_random_balls(3, state_bounds)

# Similar Trajectories
similar_trajectories = np.array([
    [5.0, 5.0, 18.0, 25.0],
    [5.0, 5.0, 18.0, 30.0],
    [5.0, 5.0, 18.0, 35.0],
    [5.0, 5.0, 18.0, 40.0]
], dtype=float)


