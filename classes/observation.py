import numpy as np
from .classifier import GaussianClassifier
from typing import Literal


class TransitionModel:
    """
    Transition-Model describing the underlying physics
    q_t = A_t q_{t-1} + B_t a_{t-1} + epsilon_t
    """
    def __init__(self, delta_t: float, process_noise: float = 1e-3, g: float = 9.81) -> None:
        self.delta_t = delta_t
        dt = delta_t

        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        self.B = np.array([
            [0,           0, 0],
            [0, 0.5 * dt**2, 0],
            [0,           0, 0],
            [0,          dt, 0],
        ])

        self.a = np.array([0.0, -g, 0.0])
        # Pre-compute the deterministic bias term once: B @ a  shape (4,)
        self._bias = self.B @ self.a

        # TEST
        self.a = np.array([0.0, -g])
        self._bias = np.array([0.0, -0.5 * g * dt**2, 0.0, -g * dt])


        #self.process_noise = process_noise
        #self.process_cov = process_noise * np.eye(4)
        #self.inv_process_cov = np.linalg.inv(self.process_cov) if process_noise > 0 else np.zeros((4, 4))  # handle zero noise case
        
        # New Assumption, not just add some noise to all dimensions.
        # Problem: We would apply same noise to two different metrics (position, velocity)
        # Instead: We need to properly adjust them Continuous White Noise Process Noise --> Random Walk in Variance
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        self.process_cov = process_noise * np.array([
                [dt4/4, 0,     dt3/2, 0    ],
                [0,     dt4/4, 0,     dt3/2],
                [dt3/2, 0,     dt2,   0    ],
                [0,     dt3/2, 0,     dt2  ]
            ])
        
        self.process_cov = process_noise * np.array([
            [dt**3/3, 0,       dt**2/2, 0      ],
            [0,       dt**3/3, 0,       dt**2/2],
            [dt**2/2, 0,       dt,      0      ],
            [0,       dt**2/2, 0,       dt     ]
        ])


        self.inv_process_cov = np.linalg.inv(self.process_cov)
        self._L = np.linalg.cholesky(self.process_cov)
        # Normalization constant for 4D log-likelihood
        sign, logdet = np.linalg.slogdet(self.process_cov)
        self._log_normalizer = -0.5 * (4 * np.log(2 * np.pi) + logdet)
                                           

        #self.inv_process_cov = np.linalg.inv(self.process_cov) if process_noise > 0 else np.zeros((4, 4))

        # Cholesky factor of process_cov for fast batch noise sampling
        #self._L = np.linalg.cholesky(self.process_cov)  # (4, 4)

    # ------------------------------------------------------------------
    # Scalar API (unchanged — still used by legacy callers)
    # ------------------------------------------------------------------

    def propagate(self, state: np.ndarray, ignore_noise: bool = False) -> np.ndarray:
        """Apply physics to a single (4,) state."""
        predicted = self.A @ state + self._bias
        if not ignore_noise:
            noise = self._L @ np.random.standard_normal(4)
            predicted += noise
            #predicted += np.random.multivariate_normal(np.zeros(4), self.process_cov)
        return predicted

    def transition_log_likelihood(self, old_state: np.ndarray, new_state: np.ndarray) -> float:
        pred = self.A @ old_state + self._bias
        residual = new_state - pred
        mahalanobis = -0.5 * (residual @ self.inv_process_cov @ residual)
        return self._log_normalizer + mahalanobis


class ObservationModel:
    """
    Observation-Model describing the data retrieved from a sensor.
    o_t = C_t q_t + delta_t
    """
    def __init__(self, classifier: Literal["Gaussian"] = None, measurement_noise: float = 1e-1) -> None:
        self.classifier = GaussianClassifier(measurement_noise) if classifier == "Gaussian" else None
        self.measurement_noise = measurement_noise

        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.use_evaluate_log_likelihood = False

        # Pre-compute Gaussian normalisation constant for the batch path:
        #   p(o | mu) = 1/(2π σ²) * exp(-||o - mu||² / (2σ²))
        # stored so we don't recompute per call.
        self._inv_2sigma2 = 1.0 / (2.0 * measurement_noise)
        self._log_norm = -np.log(2.0 * np.pi * measurement_noise)  # for 2-D obs

    # ------------------------------------------------------------------
    # Scalar API (unchanged)
    # ------------------------------------------------------------------

    def propagate(self, true_state: np.ndarray, ignore_noise: bool = False) -> np.ndarray:
        dim = 2
        measurement_cov = self.measurement_noise * np.eye(dim)
        noise = np.random.multivariate_normal(np.zeros(2), measurement_cov) if not ignore_noise else 0.0
        return true_state[:2] + noise

    def likelihood(self, observation: np.ndarray, state: np.ndarray) -> float:
        """Single-particle likelihood — scalar API, unchanged."""
        predicted_measurement = self.C @ state  # (2,)

        if observation.ndim == 1:
            if self.use_evaluate_log_likelihood:
                return np.exp(self.classifier.evaluate_log_likelihood(observation, predicted_measurement))
            else:
                return self.classifier.evaluate(observation, predicted_measurement)

        elif observation.ndim == 2:
            if self.use_evaluate_log_likelihood:
                log_ls = [
                    self.classifier.evaluate_log_likelihood(obs, predicted_measurement)
                    for obs in observation if obs is not None
                ]
                return np.exp(max(log_ls)) if log_ls else 1e-15
            else:
                ls = [
                    self.classifier.evaluate(obs, predicted_measurement)
                    for obs in observation if obs is not None
                ]
                return max(ls) + 1e-15 if ls else 1e-15

        else:
            raise NotImplementedError("Observation array must be 1D or 2D.")

    # ------------------------------------------------------------------
    # Vectorised API — called by SingleBallParticleFilter.evaluate()
    # ------------------------------------------------------------------

    def likelihood_batch(self, observation: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Compute likelihood for all N particles against a single 1-D observation.

        Args:
            observation: (2,)   — one (x, y) detection
            states:      (N, 4) — all particle states for one filter
        Returns:
            weights:     (N,)   — unnormalised likelihoods (never exactly 0)
        """
        # Predicted measurements for every particle: (N, 2)
        predicted = states[:, :2]   # equivalent to states @ C.T, avoids matmul

        # Squared Euclidean distance for every particle: (N,)
        diff = observation - predicted          # (N, 2)
        sq_dist = (diff * diff).sum(axis=1)    # (N,)

        # Gaussian likelihood: exp(-||diff||² / (2σ²))
        # (normalisation constant cancels after weight normalisation, but kept
        #  for correctness when comparing absolute likelihoods)
        return np.exp(-sq_dist * self._inv_2sigma2) + 1e-300  # avoid exact zeros
# class TransitionModel:
#     """
#     Transition-Model describing the underlying physics
#     $$q_t = A_t q_{t-1} + B_t a_{t-1} + \epsilon_t$$
#     With noise '\epsilon_t'
#     """
#     def __init__(self, delta_t: float, process_noise: float = 1e-3, g: float = 9.81) -> None:
#         self.delta_t = delta_t
#         self.process_noise = process_noise
#         self.process_cov = process_noise * np.eye(4)
#         self.inv_process_cov = np.linalg.inv(self.process_cov)

#         dt = delta_t

#         self.A = np.array([
#             [1, 0, dt, 0],
#             [0, 1, 0, dt],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]
#         ])

#         self.B = np.array([
#             [0, 0, 0],
#             [0, 0.5 * dt**2, 0],
#             [0, 0, 0],
#             [0, dt, 0]
#         ])

#         self.a = np.array([0.0, -g, 0.0])

#     def propagate(self, state: np.ndarray, ignore_noise: bool = False) -> np.ndarray:
#         """Apply physics to a given state"""
#         # q_t = A_t q_{t-1} + B_t a_{t-1}
#         predicted = self.A @ state + self.B @ self.a

#         # apply noise
#         noise = np.random.multivariate_normal(mean=np.zeros(4), cov=self.process_cov) if not ignore_noise else 0.0
#         return predicted + noise

#     def transition_log_likelihood(self, old_state: np.ndarray, new_state: np.ndarray) -> float:
#         """Returns the log-likelihood of transitioning to the new_state."""
#         pred = self.propagate(old_state, ignore_noise=True)
#         residual = new_state - pred
#         # Log-likelihood is proportional to the negative squared Mahalanobis distance
#         return -0.5 * (residual.T @ self.inv_process_cov @ residual)


# class ObservationModel:
#     """
#     Observation-Model describing the data retrieved from a sensor
#     # o_t = C_t q_t + \delta_t <---
#     Different Statistical-Classifiers could be applied (KNN, Parzen, NaiveBayes, Gaussian, GMM, --> Mixture of all??)
#     """
#     def __init__(self, classifier: Literal["Gaussian"] = None, measurement_noise: float = 1e-1) -> None:
#         # Factory pattern. Create the classifier including the measurement noise
#         self.classifier = GaussianClassifier(measurement_noise) if classifier == "Gaussian" else None
#         self.measurement_noise = measurement_noise

#         self.C = np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0]
#         ])

#         self.use_evaluate_log_likelihood = False

#     def propagate(self, true_state: np.ndarray, ignore_noise: bool = False):
#         self.dim = 2 # TODO: REMOVE
#         measurement_cov = self.measurement_noise * np.eye(self.dim)
#         noise = np.random.multivariate_normal(np.zeros(2), measurement_cov) if not ignore_noise else 0.0
#         return true_state[:2] + noise


#     def likelihood(self, observation: np.ndarray, state: np.ndarray) -> float:
#         """Compute the likelihood for one particle state, given the observation model"""
#         # TODO: Evaluate whether to use MAX / SUM / PRODUCT
#         # likelihood_distance1=0.4; likelihood_distance2=0.4
#         # SUM - Result 0.8
#         # Product - likelihood_distance1=0.9; likelihood_distance2=0.1 ==> 0.09 despite it having a high likelihood for one observation
#         # ABS - Result 0.4
#         predicted_measurement = self.C @ state

#         if observation.ndim == 1:
#             if self.use_evaluate_log_likelihood:
#                 log_L = self.classifier.evaluate_log_likelihood(observation, predicted_measurement)
#                 return np.exp(log_L)
#             else:
#                 return self.classifier.evaluate(observation, predicted_measurement)

#         elif observation.ndim == 2:
#             if self.use_evaluate_log_likelihood:
#                 # Calculate the log-likelihood of this particle against EACH ball's observation
#                 log_likelihoods = [
#                     self.classifier.evaluate_log_likelihood(obs, predicted_measurement)
#                     for obs in observation if obs is not None
#                 ]

#                 if not log_likelihoods:
#                     return 1e-15

#                 # Use MAX to prevent ghost artifacts in the dead-space between targets
#                 return np.exp(max(log_likelihoods))

#             else:
#                 # Calculate the likelihood of this particle against EACH ball's observation
#                 likelihoods = [
#                     self.classifier.evaluate(obs, predicted_measurement)
#                     for obs in observation if obs is not None
#                 ]

#                 # Use MAX to ensure assignment to a single target
#                 return max(likelihoods) + 1e-15 if likelihoods else 1e-15

#         else:
#             raise NotImplementedError("Observation array must be 1D or 2D.")
