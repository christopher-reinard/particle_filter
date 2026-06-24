
import numpy as np

class LcgEngine:
    def __init__(self, seed=0):
        self._current = seed
        self.a = 1103515245
        self.c = 12345
        self.m = 2**31

    def sample(self, n, dims):
        samples = np.zeros((n, dims))
        for i in range(n):
            for j in range(dims):
                self._current = (self.a * self._current + self.c) % self.m
                samples[i, j] = self._current / self.m
        return samples

class RandomDistribution:
    def __init__(self, engine):
        self.engine = engine

    def sample(self, n, dims):
        raise NotImplementedError()

    @staticmethod
    def estimate_parameters(samples):
        raise NotImplementedError()

engine = LcgEngine(seed=42)

class NormalDistribution(RandomDistribution):
    def __init__(self, engine, mean=np.zeros(1), std=np.ones(1), cov=None):
        super().__init__(engine)
        self._mean = mean
        self._std = std
        self._cov = cov

    def _standard_normal_samples(self, n, dims):
        u1 = self.engine.sample(n, dims)
        u2 = self.engine.sample(n, dims)
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return z0

    def sample(self, n, dims):
        z = self._standard_normal_samples(n, dims)

        if self._cov is not None:
            L = np.linalg.cholesky(self._cov)
            return self._mean + z @ L.T

        return self._mean + self._std * z

    def pdf(self, x):
        cov = self._cov if self._cov is not None else np.diag(self._std**2)

        x = np.array(x)
        k = self._mean.shape[0]

        det = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)

        norm_factor = 1.0 / np.sqrt(((2 * np.pi)**k) * det)

        dev = x - self._mean
        exponent = -0.5 * np.sum(dev @ inv_cov * dev, axis=1)

        result = norm_factor * np.exp(exponent)

        return result[0] if result.size == 1 else result

    def cdf(self, x):
        x = np.asarray(x)
        mean = self._mean.reshape(-1)
        k = mean.shape[0]

        if k == 1:
            mu = mean[0]
            sigma = self._std[0]

            z = (x - mu) / (sigma * np.sqrt(2.0))
            return 0.5 * (1.0 + np.erf(z))

        else:
            raise NotImplementedError(
                "CDF for multivariate normal not implemented."
            )

class UniformDistribution(RandomDistribution):
    def __init__(self, engine, a, b):
        super().__init__(engine)
        self._a = a
        self._b = b

    def sample(self, n, dims):
        samples = self.engine.sample(n, dims)
        return self._a + (self._b - self._a) * samples

    def pdf(self, x):
        x = np.asarray(x)
        x = np.atleast_2d(x)

        d = x.shape[1]
        a, b = self._a, self._b

        inside = np.all((x >= a) & (x <= b), axis=1)

        density = 1.0 / ((b - a) ** d)

        result = np.where(inside, density, 0.0)
        return result[0] if result.size == 1 else result

class MixtureDistribution(RandomDistribution):
    def __init__(self, engine, distributions, weights):
        super().__init__(engine)
        self.distributions = distributions
        self.weights = np.array(weights)

    def sample(self, n, dims):
        # 1. Determine how many samples come from each component
        component_choices = self.engine.sample(n, 1).flatten()
        cum_weights = np.cumsum(self.weights)

        samples = np.zeros((n, dims))
        for i, val in enumerate(component_choices):
            # Pick the distribution index based on the random roll
            target_idx = np.searchsorted(cum_weights, val)
            # Sample 1 point from that specific distribution
            samples[i, :] = self.distributions[target_idx].sample(1, dims)
        return samples

    def pdf(self, x):
        # Weighted sum of component PDFs: p(x) = sum(pi_n * N(x|mu_n, Sigma_n))
        pdf_val = 0
        for pi, dist in zip(self.weights, self.distributions):
            pdf_val += pi * dist.pdf(x)
        return pdf_val
    

class GaussianMixtureModel:
    def __init__(self):
        self.weights = None
        self.means = None
        self.covariances = None

    def _initialize_parameters(self, X, means_init=None, covs_init=None, weights_init=None):
        n_samples, n_features = X.shape

        # Initialize Weights
        self.weights = weights_init if weights_init is not None else np.full(self.n_components, 1.0 / self.n_components)

        # Initialize Means by estimated Mean
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = means_init if means_init is not None else X[indices].copy()

        # Initialize covariances as identity matrices
        self.covariances = covs_init if covs_init is not None else np.array([np.eye(n_features) for _ in range(self.n_components)])

    def fit(self, X, n_components, max_iter=100, stable_delta=1e-4,
            means_init=None, covs_init=None, weights_init=None, verbose=False):
        """Optional means_init for 'smart' initialization TODO: Also COV and Weights smart init?"""
        is_early_stopping = False
        self.n_components = n_components
        n_samples, n_features = X.shape

        # Step 1: Initialize Parameters
        self._initialize_parameters(X, means_init, covs_init, weights_init)

        prev_log_likelihood = -np.inf
        self.likelihoods = []

        # Iterate from Step 2 until log-likelihood becomes stable
        for i in range(max_iter):
            # Step 2: E-Step. Calculate the probabilities

            # Numerator: weighted probabilities
            weighted_probs = np.zeros((n_samples, self.n_components))
            for n in range(self.n_components):
                dist = NormalDistribution(engine, mean=self.means[n], cov=self.covariances[n])
                weighted_probs[:, n] = self.weights[n] * dist.pdf(X)

            # Denominator: sum across all components j
            sum_weighted_probs = weighted_probs.sum(axis=1, keepdims=True) + 1e-15
            weighted_probs += 1e-15

            try:
                r_in = weighted_probs / sum_weighted_probs
            except:
                print("ASDKASJLDLKAALSKDJLKASJDAKSDLJALSKJDLJASLDKJAJSDLAJSLDKASJDLAKSDLKA")
                print("ERROR: ", weighted_probs, sum_weighted_probs)

            R_n = r_in.sum(axis=0)

            # Step 3: M-Step. Re-estimate parameters using the current responsibilities
            for n in range(self.n_components):
                #current_R = R_n[n]
                current_R = max(R_n[n], 1e-15) # IMPORTANT FIX: Prevent 'divide by zero' (when cloud is vanishing)

                # 1. Update Weights
                self.weights[n] = current_R / n_samples

                # 2. Update Means
                self.means[n] = (r_in[:, [n]] * X).sum(axis=0) / current_R

                # 3. Update Covariances
                diff = X - self.means[n]
                self.covariances[n] = (r_in[:, n, np.newaxis] * diff).T @ diff / current_R

                # Covariance needs to be invertible!!! Diag shouldnt be 0
                self.covariances[n] += np.eye(n_features) * 1e-6

            # Check whether convergence is stable
            log_likelihood = np.sum(np.log(sum_weighted_probs + 1e-10))
            self.likelihoods.append(log_likelihood)
            if abs(log_likelihood - prev_log_likelihood) < stable_delta:
                if verbose:
                    print(f"Early Stopping after {i} iterations with delta: {abs(log_likelihood - prev_log_likelihood)}")
                is_early_stopping = True
                break
            prev_log_likelihood = log_likelihood

        if not is_early_stopping and verbose:
          print(f"Fitting Model finished after {max_iter} iterations. Delta {abs(log_likelihood - prev_log_likelihood)}")



class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()

        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            closest_clusters = np.argmin(distances, axis=1)

            new_centers = np.array([X[closest_clusters == k].mean(axis=0) for k in range(self.n_clusters)])

            if np.linalg.norm(new_centers - self.cluster_centers_) < self.tol:
                break

            self.cluster_centers_ = new_centers
