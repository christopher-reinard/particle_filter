from abc import ABC, abstractmethod

import numpy as np

# Classifiers for the Observation Model

class StatisticalClassifier(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, observation: np.ndarray, predicted_measurement: np.ndarray) -> float:
        pass

class GaussianClassifier(StatisticalClassifier):
    """
    Does it return a density or probability
    Using Mahalanobis distance for gaussian distributions
    """
    def __init__(self, measurement_noise: float = 1e-1):
        super().__init__("GaussianClassifier")

        self.dim = 2
        measurement_cov = measurement_noise * np.eye(self.dim)
        self.inv_cov = np.linalg.inv(measurement_cov)

        det_cov = np.linalg.det(measurement_cov)
        self.normalization_factor = 1.0 / np.sqrt(((2.0 * np.pi) ** self.dim) * det_cov)

        self.log_norm_factor = np.log(self.normalization_factor)

    def evaluate(self, observation: np.ndarray, predicted_measurement: np.ndarray) -> float:
        residual = observation - predicted_measurement
        exponent = -0.5 * (residual.T @ self.inv_cov @ residual)
        return self.normalization_factor * np.exp(exponent)

    def evaluate_log_likelihood(self, observation: np.ndarray, predicted_measurement: np.ndarray) -> float:
        residual = observation - predicted_measurement
        # Use log-likelihood: log(p) = -0.5 * (res^T @ inv_cov @ res) - 0.5 * log(det(cov)) - const
        # So we can sum it up easily. And constant doesnt matter, we integrate it still
        return self.log_norm_factor + (-0.5 * (residual.T @ self.inv_cov @ residual))
