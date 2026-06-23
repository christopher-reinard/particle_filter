import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Literal, Optional, Dict
from .observation import TransitionModel, ObservationModel

class Particle:
    """
    A Particle representing a single ball's state and weight.
    s = {s, w} = {(x, y, vx, vy)^T, w}
    """
    def __init__(self, state: np.ndarray, weight: float = 1.0):
        self.state: np.ndarray = state   # shape: (4,)
        self.weight: float = weight


class ParticleSet:
    """
    A set of n particles for a single ball's filter.
    St = {s_t^1, ..., s_t^n}
    """
    def __init__(self, particles: List[Particle]) -> None:
        self.particles = particles

    def normalize_weights(self) -> None:
        """
        Normalizes weights to sum to 1.0.
        w_t^i = w_t^i / sum_j(w_t^j)
        """
        total_weight = sum(p.weight for p in self.particles)
        if total_weight < 1e-5:
            print(f"Warning: normalizing near-zero total weight {total_weight:.2e} — reseeding uniformly")
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
        else:
            for p in self.particles:
                p.weight /= total_weight

    def states(self) -> np.ndarray:
        """Shape: (num_particles, 4)"""
        return np.array([p.state for p in self.particles])

    def weights(self) -> np.ndarray:
        """Shape: (num_particles,)"""
        return np.array([p.weight for p in self.particles])

    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_i^2). Low ESS means weight degeneracy."""
        w = self.weights()
        return 1.0 / np.sum(w ** 2)

    def approximate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted mean and covariance of this filter's particle distribution.
        Returns:
            mean: shape (4,)
            cov:  shape (4, 4)
        """
        states = self.states()   # (num_particles, 4)
        weights = self.weights() # (num_particles,)

        mean = np.average(states, weights=weights, axis=0)
        diff = states - mean     # (num_particles, 4)
        cov = (weights[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
        cov += np.eye(4) * 1e-6  # regularize

        return mean, cov


class SingleBallParticleFilter:
    """
    A particle filter tracking a single ball in 4D state space (x, y, vx, vy).
    This is the original CONDENSE algorithm, unchanged in structure.

    Used as a building block inside MultiObjectParticleFilter.
    """
    def __init__(self,
                 num_particles: int,
                 state_bounds: List[Tuple[float, float]],
                 transition_model: TransitionModel,
                 observation_model: ObservationModel,
                 init_generator: Literal["PseudoRandom", "Sobol", "LHS"] = "PseudoRandom",
                 use_velocity_likelihood: bool = False,
                 velocity_sigma: float = 20.0,
                 min_velocity_likelihood: float = 0.01
                 ) -> None:
        
        self.num_particles = num_particles
        self.bounds = state_bounds
        self.state_dim = len(self.bounds)  # 4

        self.transition_model = transition_model
        self.observation_model = observation_model
        self.init_generator = init_generator

        # Optional Velocity Likelihood
        self.use_velocity_likelihood = use_velocity_likelihood
        self.velocity_sigma = velocity_sigma
        self.min_velocity_likelihood = min_velocity_likelihood

        self.particle_set: ParticleSet = self._initialize()
        self._current_mean: Optional[np.ndarray] = None
        self._current_cov: Optional[np.ndarray] = None

    def _initialize(self) -> ParticleSet:
        lows = np.array([b[0] for b in self.bounds])
        highs = np.array([b[1] for b in self.bounds])

        if self.init_generator == "PseudoRandom":
            states = np.random.uniform(lows, highs, size=(self.num_particles, self.state_dim))
        elif self.init_generator in ["Sobol", "LHS"]:
            from scipy.stats import qmc
            sampler = (qmc.Sobol(d=self.state_dim, scramble=True)
                       if self.init_generator == "Sobol"
                       else qmc.LatinHypercube(d=self.state_dim, scramble=True))
            samples = sampler.random(n=self.num_particles)
            states = qmc.scale(samples, lows, highs)
        else:
            raise NotImplementedError(f"Unknown init_generator: {self.init_generator}")

        initial_weight = 1.0 / self.num_particles
        particles = [Particle(state=s, weight=initial_weight) for s in states]
        return ParticleSet(particles)

    def resample(self) -> None:
        """
        Step 2: Multinomial resample based on weights.
        """
        weights = self.particle_set.weights()

        # Resample from the current particle set based on the weights
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights, replace=True)

        new_particles = []
        for i in indices:
            copied_state = np.copy(self.particle_set.particles[i].state)
            new_particles.append(Particle(state=copied_state, weight=1.0 / self.num_particles))

        self.particle_set = ParticleSet(new_particles)


    def propagate(self) -> None:
        """Step 3: Move each particle forward via the transition model."""
        for p in self.particle_set.particles:
            p.state = self.transition_model.propagate(p.state)

    def evaluate(self, observation: np.ndarray, predicted_position: np.ndarray) -> None:
        """
        Step 4: Weight each particle by how well it explains this observation.
        w_t^i = p(o_t | s_t^i)
        """
        use_velocity_likelihood = self.use_velocity_likelihood

        if predicted_position is not None:
            estimated_velocity = (observation - predicted_position[:2]) / self.transition_model.delta_t
        else:
            estimated_velocity = None  # first frame, no prior estimate yet

        for p in self.particle_set.particles:
            position_likelihood = self.observation_model.likelihood(observation, p.state)

            if estimated_velocity is not None:
                velocity_diff = p.state[2:] - estimated_velocity
                velocity_likelihood = np.exp(
                    -0.5 * np.sum(velocity_diff**2) / self.velocity_sigma**2
                )
                velocity_likelihood = max(velocity_likelihood, self.min_velocity_likelihood)
                if use_velocity_likelihood:
                    p.weight = position_likelihood * velocity_likelihood
                else:
                    p.weight = position_likelihood
            else:
                p.weight = position_likelihood
        self.particle_set.normalize_weights()

    
    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the current (mean, cov) estimate for this ball. Caches result."""
        self._current_mean, self._current_cov = self.particle_set.approximate()
        return self._current_mean, self._current_cov

    @property
    def predicted_state(self) -> Optional[np.ndarray]:
        """The last computed mean state — used for Hungarian cost matrix."""
        return self._current_mean

class MultiObjectParticleFilter:
    """
    Tracks n_balls independently using one SingleBallParticleFilter per ball.

    Assignment of observations to filters is solved optimally each frame using
    the Hungarian algorithm on a Mahalanobis-distance cost matrix, making
    crossing trajectories distinguishable by velocity even when positions coincide.

    Algorithm per frame:
        1. Propagate all filters
        2. Get predicted state from each filter
        3. Build (n_balls x n_obs) Mahalanobis cost matrix
        4. Hungarian algorithm → optimal 1-to-1 assignment
        5. Evaluate each filter with its assigned observation
        6. Resample each filter
        7. Estimate and return
    """
    def __init__(self,
                 num_particles: int,
                 n_balls: int,
                 state_bounds: List[Tuple[float, float]],
                 transition_model: TransitionModel,
                 observation_model: ObservationModel,
                 neighbor_assignment: Literal["GreedyKNN", "Hungarian"] = "Hungarian",
                 init_generator: Literal["PseudoRandom", "Sobol", "LHS"] = "PseudoRandom",
                 ess_resample_threshold: float = 0.5, # Controls when to resample based on effective sample size (ESS)
                 use_velocity_likelihood: bool = True,
                 velocity_sigma: float = 20.0,
                 min_velocity_likelihood: float = 0.01
                 ) -> None:
        self.n_balls = n_balls
        self.ess_threshold = ess_resample_threshold

        self.filters: List[SingleBallParticleFilter] = [
            SingleBallParticleFilter(
                num_particles=num_particles,
                state_bounds=state_bounds,
                transition_model=transition_model,
                observation_model=observation_model,
                init_generator=init_generator,
                use_velocity_likelihood=use_velocity_likelihood,
                velocity_sigma=velocity_sigma,
                min_velocity_likelihood=min_velocity_likelihood
            )
            for _ in range(n_balls)
        ]
        self.prev_observations = []
        self.prev_estimates = []

        if neighbor_assignment == "GreedyKNN":
            self._assign = self._assign_greedy_knn
        elif neighbor_assignment == "Hungarian":
            self._assign = self._assign_hungarian
        else:
            raise ValueError(
                f"Unknown neighbor_assignment {neighbor_assignment!r}; "
                f"expected 'GreedyKNN' or 'Hungarian'"
            )

    def _build_cost_matrix(self,
                           predicted_means: np.ndarray,
                           predicted_covs: np.ndarray,
                           observations: List[np.ndarray]) -> np.ndarray:
        """
        Builds a (n_balls x n_obs) cost matrix using Mahalanobis distance.

        Using the full 4D predicted covariance means two balls crossing at the
        same (x,y) position are still distinguishable by their velocity components,
        so the assignment survives trajectory crossings.

        cost[i, j] = sqrt( (obs_j - mean_i)^T @ inv(cov_i) @ (obs_j - mean_i) )

        If the observation is lower-dimensional than the state (e.g. only x,y),
        we slice the relevant rows/columns from the covariance.
        """
        n_obs = len(observations)
        obs_dim = len(observations[0])
        cost = np.zeros((self.n_balls, n_obs))

        for i in range(self.n_balls):
            mu = predicted_means[i, :obs_dim]           # slice to obs dimension
            cov_slice = predicted_covs[i, :obs_dim, :obs_dim]
            inv_cov = np.linalg.inv(cov_slice + np.eye(obs_dim) * 1e-6)

            for j, obs in enumerate(observations):
                diff = obs - mu
                cost[i, j] = np.sqrt(diff @ inv_cov @ diff)

        return cost

    def _assign_greedy_knn(self,
                           predicted_means: np.ndarray,
                           predicted_covs: np.ndarray,
                           observations: List[np.ndarray]) -> Dict[int, int]:
        """
        Greedy KNN assignment: for each filter, find the closest observation by Mahalanobis distance.
        """

        if self.n_balls > 1:
            cost = self._build_cost_matrix(predicted_means, predicted_covs, observations)
            
            assignment = {}
            for i in range(self.n_balls):
                closest_obs_idx = np.argmin(cost[i])
                # If this observation is already assigned to another filter, we have a conflict, 
                # so we assign to the next closest observation instead
                while closest_obs_idx in assignment.values():
                    cost[i, closest_obs_idx] = np.inf  # exclude this obs and find next closest
                    closest_obs_idx = np.argmin(cost[i])
                assignment[i] = closest_obs_idx
        else:
            assignment = {0: 0}

        return assignment


    def _assign_hungarian(self,
                predicted_means: np.ndarray,
                predicted_covs: np.ndarray,
                observations: List[np.ndarray]) -> Dict[int, int]:
        """
        Runs Hungarian algorithm on the cost matrix, isn't really hungarian, but uses scipy's implementation of the algorithm.
        Returns dict: filter_idx -> observation_idx
        If there are more filters than observations, unmatched filters get None.
        """

        if self.n_balls > 1:
            cost = self._build_cost_matrix(predicted_means, predicted_covs, observations)
            print(cost)

            # linear_sum_assignment minimizes total cost
            filter_indices, obs_indices = linear_sum_assignment(cost)

            assignment = {}
            for f_idx, o_idx in zip(filter_indices, obs_indices):
                assignment[f_idx] = o_idx
        else:
            assignment = {0: 0}

        return assignment  # filter_idx -> obs_idx (unmatched filters absent)

    def run(self,
            observations: List[Optional[List[np.ndarray]]],
            logs: List[str] = []) -> List[Dict]:
        """
        Main execution loop.

        Args:
            observations: list of frames. Each frame is either:
                - None: no detections this frame (filters still propagate)
                - List[np.ndarray]: one observation vector per detected ball
                  (can be 2D position [x,y] or full 4D state)

        Returns:
            history: list of dicts per frame:
                - time_step
                - observation
                - estimates       List of (4,) mean state per ball
                - estimate_covs   List of (4,4) covariance per ball
                - assignment      Dict[filter_idx -> obs_idx] (or None)
                - particle_states List of (num_particles, 4) arrays per filter
        """
        log_pf = "PF" in logs
        history = []

        # Initialize estimates so cost matrix works on first real frame
        predicted_means = np.zeros((self.n_balls, 4))
        predicted_covs = np.array([np.eye(4) * 1e6 for _ in range(self.n_balls)])  # wide prior

        for t, observation in enumerate(observations):

            # --- Step 1: Propagate all filters ---
            for f in self.filters:
                f.propagate()

            # --- Step 2: Get predicted states from each filter ---
            for i, f in enumerate(self.filters):
                mean, cov = f.estimate()
                predicted_means[i] = mean
                predicted_covs[i] = cov

            assignment = None

            if observation is not None:
                # --- Step 3 & 4: Build cost matrix and assign ---

                # Given the predictions and the observation, which filter should get which assignment?
                assignment = self._assign(predicted_means, predicted_covs, observation)

                # --- Step 5: Evaluate each filter with its assigned observation ---
                for f_idx, f in enumerate(self.filters):
                    if f_idx in assignment:
                        obs = observation[assignment[f_idx]]
                        f.evaluate(obs, predicted_means[f_idx][:2])
                    # If a filter got no observation (more filters than obs), skip evaluate
                    # — its weights stay uniform from last resample, which is the right
                    # thing to do: all locations equally plausible until we see it again.

                # --- Step 6: Resample each filter (adaptive ESS threshold) ---
                for f in self.filters:
                    ess = f.particle_set.effective_sample_size()
                    if ess < self.ess_threshold * f.num_particles:
                        f.resample()

            # --- Step 7: Final estimates ---
            estimates = []
            covs = []
            for i, f in enumerate(self.filters):
                mean, cov = f.estimate()
                estimates.append(mean)
                covs.append(cov)
                predicted_means[i] = mean
                predicted_covs[i] = cov

            if log_pf:
                print(f"t={t}  assignment={assignment}")
                for i, (m, _) in enumerate(zip(estimates, covs)):
                    print(f"  ball {i}: x={m[0]:.1f}  y={m[1]:.1f}  vx={m[2]:.2f}  vy={m[3]:.2f}")

            history.append({
                "time_step": t,
                "observation": observation,
                # Keep original multi-object keys for compatibility, but also
                # provide `estimate` as a single ndarray and `particle_states`
                # as a single stacked array so plotting code written for the
                # single-filter `ParticleFilter` works unchanged.
                "estimate": np.array([e.copy() for e in estimates]),        # shape (n_balls, 4)
                "estimate_covs": [c.copy() for c in covs],                  # List of (4,4)
                "assignment": assignment,                                    # Dict[filter_idx -> obs_idx]
                "particle_states": np.vstack([f.particle_set.states().copy() for f in self.filters])  # shape (n_balls*N,4)
            })

        return history


# ---------------------------------------------------------------------------
# Backwards-compatible alias so existing call sites don't break immediately
# ---------------------------------------------------------------------------
ParticleFilter = MultiObjectParticleFilter