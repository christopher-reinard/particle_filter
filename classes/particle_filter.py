import numpy as np
from typing import List, Tuple, Literal, Optional, Dict
from .observation import TransitionModel, ObservationModel
from .GMM import GaussianMixtureModel, KMeans


class Particle:
    """
    A Particle - consisting of state information and a weight
    s = {s, w} = {(x, y)^T, w}
    At each time step t and index i: s_t^i = {s_t^i, w_t^i} = {(x_t^i, y_t^i)^T, w_t^i}
    """
    def __init__(self, state: np.ndarray, weight: float = 1.0):
        self.state: np.ndarray = state
        self.weight: float = weight

class ParticleSet:
    """
    A Set consisting of n particles - each representing a state and weight at time t --> s_t
    St = {s_t^1, s_t^2, ..., s_t^n}
    The more particles a set consists of - the more accurate is the description of the possible states
    """
    def __init__(self, particles: List[Particle], approximate: Literal["GMM", "KMeans"] = "GMM") -> None:
        self.particles = particles
        if approximate == "GMM":
            self.gmm = GaussianMixtureModel()
        elif approximate == "KMeans":
            self.gmm = KMeans()

        self.gmm_weights = None
        self.gmm_covariances = None

    def update_particles(self, new_particles: List[Particle]) -> None:
        """Updates particles and resets stale GMM state for the next estimation."""
        self.particles = new_particles
        self.gmm_weights = self.gmm_weights

        #if self.gmm_covariances is not None:
        #noise_buffer = np.eye(4) * 2.0
        #noise_buffer = np.diag([2.0, 2.0, 0.1, 0.1]) # Scale properly
        #self.gmm_covariances = self.gmm_covariances #+ noise_buffer
        #self.gmm_covariances = None
        if self.gmm_covariances is not None:
            # Add small noise, to ensure invertability
            noise_buffer = np.eye(4) * 1e-6 
            self.gmm_covariances = self.gmm_covariances + noise_buffer

    def normalize_weights(self) -> None:
        """
        Normalizes the weights of all particles so they sum to 1.0.
        w_t^i = \frac{w_t^i}{\sum_jw_t^j}
        """
        total_weight = sum(p.weight for p in self.particles)

        if total_weight < 1e-5:
            print("Normalizing weights sum of ", total_weight)
        for p in self.particles:
            p.weight /= total_weight

    def states(self):
        return np.array([p.state for p in self.particles])

    def weights(self):
        return np.array([p.weight for p in self.particles])

    def approximate(self, n_objects = 1, estimated_psotion = None, log_gmm = False) -> np.ndarray:
        """
        Calculates the weighted average (the expected value) of the particle states.
        p(q_t|<o>_t) /approx SUM_i(w_t^i*\delta*(s_t^i - q_t))
        E[q_t] = \approx \sum_i w_t^i * s_t^i for 1 ball assumption <-- TODO: Adjustment for n balls add Parzen Window KDE, or K-means clustering
        """
        states = self.states()#np.array([p.state for p in self.particles])
        weights = self.weights()#np.array([p.weight for p in self.particles])

        if self.gmm_covariances is None:
            print("INIT COVARIANCE")
            global_cov = np.cov(states, rowvar=False) + np.eye(4) * 2.0#1e-3
            initial_covs = np.array([global_cov for _ in range(n_objects)])
            covs = initial_covs
        else:
            covs = self.gmm_covariances

        weights = self.gmm_weights if self.gmm_weights is not None else weights

        self.gmm.fit(states, n_components=n_objects,
                      means_init=estimated_psotion,
                      weights_init=weights,
                      covs_init=covs,
                      verbose=log_gmm)
        self.gmm_weights = self.gmm.weights
        self.gmm_covariances = self.gmm.covariances
        return self.gmm.means, self.gmm.covariances
    

class ParticleFilter:
    """
    Particle Filter implementing Condense-Algorithm.
    Step 1: Create initial particle set S_0 with uniform distribution (each state is similar possible).
    Step 2: Sample particles based on their weights (probability of occuring).
    Step 3: Propagate each particle with the state transition model (move particles according to physics model).
    Step 4: Evaluate each particle with the observation model and adjust the weights (could this particle exist according to physics)
    Step 5: Repeat at Step 2
    """
    def __init__(self, 
                 num_particles: int, 
                 state_bounds: List[Tuple[float, float]], 
                 transition_model: TransitionModel, 
                 observation_model: ObservationModel, 
                 init_generator=Literal["PseudoRandom", "Sobol", "LHS"],
                 roughening_noise: float = 0.0
                 ) -> None:
        self.num_particles = num_particles
        self.bounds = state_bounds
        self.state_dim = len(self.bounds)

        self.transition_model = transition_model
        self.observation_model = observation_model

        self.init_generator = init_generator
        self.roughening_noise = roughening_noise
        self.particle_set: ParticleSet = self._initialize()

    def _initialize(self) -> ParticleSet:
        """
        Step 1: Create initial particle set S_0 with uniform distribution.
        Uniform probability for each particle: w_0^i = 1/n
        """
        particles = []

        lows = np.array([b[0] for b in self.bounds])
        highs = np.array([b[1] for b in self.bounds])

        # states (Shape: [num_particles, state_dim])
        if self.init_generator == "PseudoRandom":
          states = np.random.uniform(lows, highs, size=(self.num_particles, self.state_dim))
        elif self.init_generator in ["Sobol", "LHS"]:
          from scipy.stats import qmc
          sampler = qmc.Sobol(d=self.state_dim, scramble=True, seed=4) if self.init_generator == "Sobol" else qmc.LatinHypercube(d=self.state_dim, scramble=True, seed=4)

          samples = sampler.random(n=self.num_particles)
          states = qmc.scale(samples, lows, highs)
        else:
            raise NotImplementedError()

        initial_weight = 1.0 / self.num_particles
        particles = [Particle(state=s, weight=initial_weight) for s in states]

        return ParticleSet(particles)

    def _resample(self, n_objects: int = 1, estimated_means: np.ndarray = None, estimated_covs: np.ndarray = None,  observations: np.ndarray = None) -> None:
        states = self.particle_set.states()
        weights = self.particle_set.weights()
        #states = np.array([p.state for p in self.particle_set.particles])
        #weights = np.array([p.weight for p in self.particle_set.particles])

        if n_objects == 1 or estimated_means is None or estimated_covs is None:
            print("Resampling - Var 1")
            indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights, replace=True)
            new_particles = []
            for i in indices:
                copied_state = np.copy(self.particle_set.particles[i].state)
                copied_state += np.random.normal(0, self.roughening_noise, size=copied_state.shape)  # Add roughening noise
                new_particles.append(Particle(state=copied_state, weight=1.0 / self.num_particles))
            #self.particle_set.particles = new_particles
            self.particle_set = ParticleSet(new_particles)

        else:
            new_particles = []

            # Use len(states) to ensure exact matrix dimension matching
            distances = np.zeros((len(states), n_objects))

            # Calc Mahalanobis distances for each estimated centroid: (particle_state-estimated_mean)
            for j in range(n_objects):
                mu = estimated_means[j]
                cov = estimated_covs[j]

                inv_cov = np.linalg.inv(cov + np.eye(4) * 1e-6)
                diff = states - mu

                mahalanobis_sq = np.sum((diff @ inv_cov) * diff, axis=1)
                distances[:, j] = np.sqrt(mahalanobis_sq)

            # Determine for each particle the closest distance to the next centroid
            cluster_assignments = np.argmin(distances, axis=1)

            valid_observations = None
            if observations is not None:
                valid_observations = np.asarray(observations)
                if valid_observations.ndim == 1:
                    valid_observations = valid_observations[None, :]
                valid_observations = valid_observations[~np.isnan(valid_observations).any(axis=1)]
                if len(valid_observations) == 0:
                    valid_observations = None

            for i in range(n_objects):
                cluster_indices = np.where(cluster_assignments == i)[0]

                base_count = self.num_particles // n_objects
                extra = 1 if i < (self.num_particles % n_objects) else 0
                target_count = base_count + extra

                if len(cluster_indices) > 0:
                    cluster_weights = weights[cluster_indices]
                    weight_sum = np.sum(cluster_weights)

                    if weight_sum > 1e-15:
                        cluster_weights /= weight_sum
                    else:
                        cluster_weights = np.ones(len(cluster_indices)) / len(cluster_indices)

                    # Given weights, randomly select particles from the cluster to be duplicated in the new set
                    chosen_indices = np.random.choice(cluster_indices, size=target_count, p=cluster_weights)

                    for idx in chosen_indices:
                        copied_state = np.copy(states[idx])
                        copied_state += np.random.normal(0, self.roughening_noise, size=copied_state.shape)
                        new_particles.append(Particle(state=copied_state, weight=1.0 / self.num_particles))
                else:
                    print("--> Cluster lost. Resampling around Estimation.")
                    amount_estimation = round(target_count * 0.50)
                    amount_observation = round(target_count * 0.50)
                    amount_estimation += target_count - (amount_estimation + amount_observation)

                    safe_cov = estimated_covs[i] + np.eye(self.state_dim) * 1e-6
                    if amount_estimation > 0:
                        rescued_states = np.random.multivariate_normal(
                            estimated_means[i],
                            safe_cov,
                            size=amount_estimation
                        )
                        new_particles.extend(
                            Particle(state=s, weight=1.0 / self.num_particles)
                            for s in rescued_states
                        )

                    if amount_observation > 0:
                        observation_states = np.random.multivariate_normal(
                            estimated_means[i],
                            safe_cov,
                            size=amount_observation
                        )
                        if valid_observations is not None:
                            obs_dim = min(valid_observations.shape[1], self.state_dim)
                            obs_diffs = valid_observations[:, :obs_dim] - estimated_means[i, :obs_dim]
                            nearest_obs = valid_observations[np.argmin(np.sum(obs_diffs * obs_diffs, axis=1))]
                            measurement_std = np.sqrt(self.observation_model.measurement_noise)
                            observation_states[:, :obs_dim] = (
                                nearest_obs[:obs_dim]
                                + np.random.normal(0, measurement_std, size=(amount_observation, obs_dim))
                            )
                        new_particles.extend(
                            Particle(state=s, weight=1.0 / self.num_particles)
                            for s in observation_states
                        )

            #self.particle_set = ParticleSet(new_particles)
            self.particle_set.update_particles(new_particles)

    def _resample_original(self) -> None:
        """
        Step 2: Sample particles based on their weights.
        Requires Resampling for each cluster
        """
        states = self.particle_set.states()
        weights = self.particle_set.weights()
        #states = np.array([p.state for p in self.particle_set.particles])
        #weights = np.array([p.weight for p in self.particle_set.particles])

        indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights, replace=True)

        new_particles = []
        for i in indices:
            old_particle = self.particle_set.particles[i]
            # Copy state to avoid reference sharing among duplicated particles
            copied_state = np.copy(old_particle.state)
            new_particles.append(Particle(state=copied_state, weight=1.0 / self.num_particles))

        self.particle_set = ParticleSet(new_particles)

    def _propagate(self) -> None:
        """
        Step 3: Propagate each particle with the state transition model.
        In Theory: p(q_t | q_{t-1}, a_{t-1}) or p(q_t | q_{t-1}) or p(q_t | q_{t-1}, o_{t-1})
        In Our case: Physics Model with noise: q_t = A_t q_{t-1} + B_t q_{t-1} + \epsilon_t
        """
        for p in self.particle_set.particles:
            p.state = self.transition_model.propagate(p.state)

    def _evaluate(self, observation: np.ndarray) -> None:
        """
        Step 4: Evaluate each particle with the observation model and adjust the weights.
        Weight-Update: w_t^i = p(o_t | (x_t^i, y_t^i)^T) = N(o_t | (x_t^i, y_t^i)^T, CovarianceMatrix)
            - Alternatively each distribution could be used. Doesnt has to be a Normal-Distr
            - We can use different statistical Classifier to calculate the probability for each sample for the current, given observation <-- TODO
        Normalization (reassure weights sum up to 1): w_t^i = \frac{w_t^i}{\sum_jw_t^j}
        """
        # TODO: Add log-likelihood of transition_model for the new state
        #for p in self.particle_set.particles:
        #    p.weight = self.observation_model.likelihood(observation, p.state)

        """Evaluates particle weights against the closest target to support GMM split."""
        for p in self.particle_set.particles:
            # Compute maximum or average likelihood across all observed targets
            likelihoods = [self.observation_model.likelihood(obs, p.state) for obs in observation]
            p.weight = np.max(likelihoods)  # Link particle to its nearest likely target

        # Normalize weights again to sum up to 1.0
        self.particle_set.normalize_weights()



    def run(self, observations: List[Optional[np.ndarray]], n_objects,
            change_resample_order=True, logs=["PF", "GMM"]) -> List[Dict]:
        """Execution loop, returning history for visualization"""
        print(f"Exec ParticleFilter with {self.num_particles} Particles and {n_objects} balls. change_resample_order = {change_resample_order}")
        history = []
        log_gmm = True if "GMM" in logs else False
        log_pf = True if "PF" in logs else False

        old_position = None
        old_covs = None

        t = 0
        for observation in observations:
            if (observation is not None) & (change_resample_order == False): # it doesnt make sense to resample if weights werent updated
                self._resample(n_objects=n_objects, estimated_means=old_position, estimated_covs=old_covs, observations=observation)

            self._propagate()

            if observation is not None:
                self._evaluate(observation)

            # Adjustment to Condensation-Algorithm. Resampling at the beginning doesnt make sense. First Iter always shows estimation of MEAN. Because we have sample uniform and weighting doesnt influence enough alone
            if (observation is not None) & (change_resample_order == True): # it doesnt make sense to resample if weights werent updated
                self._resample(n_objects=n_objects, estimated_means=old_position, estimated_covs=old_covs, observations=observation)

            new_position, new_covs = self.particle_set.approximate(
                n_objects=n_objects,
                estimated_psotion=old_position,
                log_gmm=log_gmm
            )

            # Assign the estimations properly to its previous partner
            if old_position is not None and n_objects > 1:
                ordered_means = np.zeros_like(new_position)
                ordered_covs = np.zeros_like(new_covs)
                available_clusters = list(range(n_objects))

                # For each prev_position calculate the likelihood, given the transformation model, in respect to the new estimated_position
                for i, prev_pos in enumerate(old_position):
                    # Evaluate the transition likelihood for all remaining clusters/positions
                    likelihoods = [
                        self.transition_model.transition_log_likelihood(prev_pos, new_position[j])
                        for j in available_clusters
                    ]

                    # Pop the cluster with the highest likelihood and assign it
                    best_idx = available_clusters.pop(np.argmax(likelihoods))
                    ordered_means[i] = new_position[best_idx]
                    ordered_covs[i] = new_covs[best_idx]

                new_position = ordered_means
                new_covs = ordered_covs

            if log_pf:
              print(f"Time Step: {t}, Estimated Position: \n{new_position}")

            history.append({
                "time_step": t,
                "observation": observation,
                "estimate": new_position.copy(),
                "particle_states": np.array([p.state for p in self.particle_set.particles])
            })

            old_position = new_position.copy()
            old_covs = new_covs.copy()
            t += 1

        return history
