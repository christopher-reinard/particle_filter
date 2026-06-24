import numpy as np

# Generates the Simulations and Observations for the Particle Filter

def generate_random_balls(num_balls, state_bounds):
    balls = np.array([[np.random.uniform(low+(0.10*high), high-(0.10*high)) for low, high in state_bounds] for _ in range(num_balls)])
    return balls

true_state = np.array([5.0, 5.0, 15.0, 25.0])

def create_ground_truth(num_steps, dropout_start, dropout_end, true_state, transition_model, observation_model):
    true_trajectory, observations = [], []
    current_true_state = true_state.copy()
    for t in range(num_steps):
        true_trajectory.append(current_true_state.copy())

        if dropout_start <= t <= dropout_end:
            observations.append(None) # Sensor failure
        else:
            # Add measurement noise to the true x, y position
            obs = observation_model.propagate(current_true_state, ignore_noise=False)
            observations.append(obs)

        # Propagate state without any noise to determine real value
        current_true_state = transition_model.propagate(current_true_state, ignore_noise=True)

    return true_trajectory, observations

def propagate_true_multi(state, transition_model, ignore_noise=False):
    return np.vstack([transition_model.propagate(s, ignore_noise=ignore_noise) for s in state])

def observe_multi(state, observation_model, ignore_noise=False):
    return np.vstack([observation_model.propagate(s, ignore_noise=ignore_noise) for s in state])

def create_ground_truth_n_balls(num_steps, dropout_start, dropout_end, true_state, transition_model, observation_model):
    true_trajectory = []
    observations = []

    current_true_state = true_state.copy()   # shape: (n_balls, 4)

    for t in range(num_steps):
        true_trajectory.append(current_true_state.copy())

        if dropout_start <= t <= dropout_end:
            observations.append(None)  # keep dropout explicit
        else:
            obs = observe_multi(current_true_state, observation_model, ignore_noise=False)  # (n_balls, 2)
            observations.append(obs)

        current_true_state = propagate_true_multi(current_true_state, transition_model, ignore_noise=True)

    true_trajectory = np.stack(true_trajectory, axis=0)  # (num_steps, n_balls, 4)
    return true_trajectory, observations

