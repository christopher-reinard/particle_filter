"""
Evaluator Module for tracking performance between Estimates and Observations against Ground Truth.
The Commented out code are all the old versions of the evaluator code, which are kept for reference and comparison.

The current version is the simplest version of the evaluator. 
1. Take the euclidean distance between the true and the estimated/observed positions, 
2. Assign the estimated/observed positions to the true positions using the Hungarian algorithm (linear_sum_assignment) to minimize the total distance.
3. Calculates the mean, mse and rmse of the distances.

Limitation to this Approach:
- This could also not be the true observation or estimation error: 
    - if paths collide and the observations are somehow "lucky" and get assigned to the wrong true position.

Old Version 1: Euclidean Distance with Gating (Scrapped because it was not necessary to use gating, and AI suggested "Better" solution)

Old Version 2: Mahalanobis Distance with Gating (Scrapped because it was again not necessary to use gating, 
                Additionally, the distance calculation is wrong since the filter itself has its own distribution.
                Another point was that if the filter is more uncertain, the loss would be lower
                Finally, it's just too complex for the task at hand, so uncertainties are not taken into consideration

"""

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np


def _assign_and_distances(true_pos, points):
    """
    Assign points to true_pos using Hungarian algorithm (min Euclidean cost).
    Returns distances aligned to true_pos order, NaN for unmatched targets.
    """
    n = true_pos.shape[0]

    if points is None:
        return np.full(n, np.nan)

    pts = np.asarray(points)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts = pts[:, :2]

    cost = cdist(true_pos, pts, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost)

    dists = np.full(n, np.nan)
    for r, c in zip(row_ind, col_ind):
        dists[r] = cost[r, c]

    return dists


def get_stats(true_trajectory, observations, history, num_steps, average_time):
    """
    Aggregate per-frame Euclidean assignment errors over the full trajectory.

    Args:
        true_trajectory:  list of (n, 4) or (n, 2) arrays
        observations:     list of (m, 2) arrays or Nones
        history:          list of dicts with key 'estimate'
        num_steps:        int
        average_time:     float, seconds per step

    Returns dict with MAE, MSE, RMSE for both observations and estimates.
    """
    all_obs_dists = []
    all_est_dists = []

    for t in range(num_steps):
        true_pos = np.asarray(true_trajectory[t])[:, :2]
        obs      = observations[t]
        est      = history[t]['estimate']

        obs_d = _assign_and_distances(true_pos, obs)
        est_d = _assign_and_distances(true_pos, est)

        # Only score frames where BOTH have a valid assignment for the same target
        valid = ~np.isnan(obs_d) & ~np.isnan(est_d)
        all_obs_dists.extend(obs_d[valid])
        all_est_dists.extend(est_d[valid])

    obs_d = np.asarray(all_obs_dists)
    est_d = np.asarray(all_est_dists)

    def metrics(arr):
        return {
            'mae':  float(np.mean(arr)), # Mean Error 
            'mse':  float(np.mean(arr ** 2)), # Mean Squared Error (in squared units)
            'rmse': float(np.sqrt(np.mean(arr ** 2))), # MSE error but in the same units as the original data
        }

    obs_metrics = metrics(obs_d)
    est_metrics = metrics(est_d)

    rmse_improvement = 100 * (obs_metrics['rmse'] - est_metrics['rmse']) / obs_metrics['rmse']
    mse_improvement  = 100 * (obs_metrics['mse']  - est_metrics['mse'])  / obs_metrics['mse']
    mae_improvement  = 100 * (obs_metrics['mae']  - est_metrics['mae'])  / obs_metrics['mae']

    return {
        'obs':  obs_metrics,
        'est':  est_metrics,
        'rmse_improvement': rmse_improvement,
        'mse_improvement':  mse_improvement,
        'mae_improvement':  mae_improvement,
        'average_time': average_time,
        'raw': {
            'true_trajectory': true_trajectory,
            'observations':    observations,
            'history':         history,
            'num_steps':       num_steps,
        }
    }

# from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist
# import numpy as np

# Old evaluator code (commented out) for reference.
# Computes the cost matrix 
# def _assign_and_distances(true_pos, points, miss_cost=1e6, gate=None):
#     """
#     Assign `points` to `true_pos` (both Nx2 arrays) minimizing total Euclidean cost.
#     Returns distances aligned to true_pos order (np.nan for unmatched).
#     """
#     n = true_pos.shape[0] #Number of true Targets

#     # Returns nans if there is dropout and no points to assign
#     if points is None:
#         return np.full(n, np.nan), None, None

#     pts = np.asarray(points)
#     if pts.ndim == 1:
#         pts = pts.reshape(1, -1)
#     pts = pts[:, :2]

#     # cost matrix (n_true x n_pts)
#     cost = cdist(true_pos, pts, metric='euclidean')

#     if gate is not None:
#         cost = np.where(cost > gate, miss_cost, cost)

#     row_ind, col_ind = linear_sum_assignment(cost)
#     dists = np.full(n, np.nan)
#     for r, c in zip(row_ind, col_ind):
#         # If we used a gate and the chosen cost is the miss_cost, treat as unmatched
#         if cost[r, c] >= miss_cost:
#             dists[r] = np.nan
#         else:
#             dists[r] = cost[r, c]

#     return dists, row_ind, col_ind

# def get_distance_one_point(true_states, observations, estimates, gate=None):
#     """
#     Compute per-target assignment distances for observations and estimates.

#     - true_states: (n,4) array (x,y,vx,vy) or (n,2)
#     - observations: (m,2) array or None
#     - estimates: (k,4) or (k,2) array or None
#     - gate: optional float distance threshold to reject far matches

#     Returns dict with:
#       - 'obs_dists', 'est_dists': arrays length n (aligned to true_states indices), NaN for unmatched
#       - 'obs_mean','obs_mse','est_mean','est_mse' : scalars (NaN-safe)
#       - 'obs_assign','est_assign': (row_ind, col_ind) tuples from solver (or None)
#     """
#     true_pos = np.asarray(true_states)[:, :2]

#     obs_dists, obs_r, obs_c = _assign_and_distances(true_pos, observations, gate=gate)
#     est_dists, est_r, est_c = _assign_and_distances(true_pos, estimates, gate=gate)

#     # Replace NaNs in observation distances (caused by sensor dropout)
#     # with the mean of the non-NaN observation distances when possible.
#     # If all observation distances are NaN (no observations at this frame), keep them as NaN.
#     # if not np.all(np.isnan(obs_dists)):
#     #     obs_mean_replace = float(np.nanmean(obs_dists))
#     #     obs_dists = np.where(np.isnan(obs_dists), obs_mean_replace, obs_dists)

#     def stats(arr):
#         if np.all(np.isnan(arr)):
#             return np.nan, np.nan, np.nan

#         mean = float(np.nanmean(arr))
#         mse  = float(np.nanmean(np.square(arr)))
#         rmse = float(np.sqrt(mse))

#         return mean, mse, rmse

#     obs_mean, obs_mse, obs_rmse = stats(obs_dists)
#     est_mean, est_mse, est_rmse = stats(est_dists)
#     return {
#         'obs_dists': obs_dists,
#         'est_dists': est_dists,
#         'obs_mean': obs_mean,
#         'obs_mse': obs_mse,
#         'obs_rmse': obs_rmse,
#         'est_mean': est_mean,
#         'est_mse': est_mse,
#         'est_rmse': est_rmse,
#         'obs_assign': (obs_r, obs_c) if obs_r is not None else None,
#         'est_assign': (est_r, est_c) if est_r is not None else None
#     }

# def get_stats(true_trajectory, observations, history, num_steps, average_time, gate=None):
#     all_obs_dists = []
#     all_est_dists = []

#     for t in range(num_steps):
#         true = true_trajectory[t]
#         obs  = observations[t]
#         est  = history[t]['estimate']

#         res = get_distance_one_point(true, obs, est, gate=gate)

#         obs_d = res['obs_dists']
#         est_d = res['est_dists']

#         # Only compare on frames where BOTH have a valid assignment
#         valid = ~np.isnan(obs_d) & ~np.isnan(est_d)
#         all_obs_dists.extend(obs_d[valid])
#         all_est_dists.extend(est_d[valid])

#     all_est_dists = np.asarray(all_est_dists)
#     all_obs_dists = np.asarray(all_obs_dists)

#     est_rmse = np.sqrt(np.mean(np.square(all_est_dists)))
#     obs_rmse = np.sqrt(np.mean(np.square(all_obs_dists)))

#     est_mean_error = np.mean(all_est_dists)
#     obs_mean_error = np.mean(all_obs_dists)

#     rmse_improvement = 100 * (obs_rmse - est_rmse) / obs_rmse if obs_rmse > 0 else float('inf')
#     mse_improvement  = 100 * (obs_rmse**2 - est_rmse**2) / obs_rmse**2 if obs_rmse > 0 else float('inf')

#     output = {
#         'est_mean_error': est_mean_error,
#         'est_rmse': est_rmse,
#         'obs_mean_error': obs_mean_error,
#         'obs_rmse': obs_rmse,
#         'mse_improvement': mse_improvement,
#         "rmse_improvement": rmse_improvement,
#         'raw': {
#             "true_trajectory": true_trajectory,
#             "observations": observations,
#             "history": history,
#             "num_steps": num_steps
#         },
#         "average_time": average_time,
#     }
#     return output



# Begin of Evaluator 2

# from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist
# from scipy.stats import chi2
# import numpy as np


# def _mahalanobis_gate(true_pos, pts, measurement_noise, confidence=0.95):
#     """
#     Returns a boolean (n_true x n_pts) gate matrix.
#     True means the pair is within the chi2 gate at the given confidence level.
#     The gate is noise-normalised: a threshold of chi2(0.95, df=2) ~ 5.99
#     means 'within the 95% confidence ellipse of the observation noise',
#     regardless of whether measurement_noise is 10 or 35.
#     """
#     threshold = chi2.ppf(confidence, df=2)
#     # R = sigma^2 * I  =>  inv(R) = (1/sigma^2) * I
#     inv_sigma2 = 1.0 / (measurement_noise ** 2)
#     # Mahalanobis^2 = (dx^2 + dy^2) / sigma^2  (isotropic noise)
#     cost = cdist(true_pos, pts, metric='euclidean') ** 2 * inv_sigma2
#     return cost <= threshold  # True = inside gate


# def _assign_and_distances(true_pos, points, measurement_noise, confidence=0.95, miss_cost=1e6):
#     """
#     Assign `points` to `true_pos` (both Nx2 arrays) minimizing total Euclidean cost,
#     with a chi-squared Mahalanobis gate to reject implausible assignments.

#     Args:
#         true_pos:           (n, 2) array of ground-truth positions
#         points:             (m, 2) array of observed/estimated positions, or None
#         measurement_noise:  scalar sigma used for the isotropic noise model
#         confidence:         chi2 confidence level for the gate (default 0.95)
#         miss_cost:          large sentinel cost assigned to gated-out pairs

#     Returns:
#         dists:    (n,) array of Euclidean distances aligned to true_pos; NaN for unmatched
#         row_ind:  row indices from linear_sum_assignment (or None)
#         col_ind:  col indices from linear_sum_assignment (or None)
#     """
#     n = true_pos.shape[0]

#     if points is None:
#         return np.full(n, np.nan), None, None

#     pts = np.asarray(points)
#     if pts.ndim == 1:
#         pts = pts.reshape(1, -1)
#     pts = pts[:, :2]

#     # Euclidean cost matrix (n_true x n_pts)
#     cost = cdist(true_pos, pts, metric='euclidean')

#     # Apply chi-squared Mahalanobis gate: pairs outside the gate get miss_cost
#     gate_mask = _mahalanobis_gate(true_pos, pts, measurement_noise, confidence)
#     cost = np.where(gate_mask, cost, miss_cost)

#     row_ind, col_ind = linear_sum_assignment(cost)

#     dists = np.full(n, np.nan)
#     for r, c in zip(row_ind, col_ind):
#         if cost[r, c] < miss_cost:  # only accept if within gate
#             dists[r] = cdist(true_pos[r:r+1], pts[c:c+1])[0, 0]

#     return dists, row_ind, col_ind


# def get_distance_one_point(true_states, observations, estimates, measurement_noise, confidence=0.95):
#     """
#     Compute per-target assignment distances for observations and estimates.

#     Args:
#         true_states:        (n, 4) or (n, 2) ground-truth array
#         observations:       (m, 2) array or None
#         estimates:          (k, 4) or (k, 2) array or None
#         measurement_noise:  scalar sigma for the chi-squared gate
#         confidence:         chi2 confidence level for the gate (default 0.95)

#     Returns dict with:
#         obs_dists, est_dists : (n,) arrays aligned to true_states, NaN for unmatched
#         obs_mean, obs_mse, obs_rmse : scalars (NaN-safe)
#         est_mean, est_mse, est_rmse : scalars (NaN-safe)
#         obs_assign, est_assign      : (row_ind, col_ind) tuples or None
#     """
#     true_pos = np.asarray(true_states)[:, :2]

#     obs_dists, obs_r, obs_c = _assign_and_distances(
#         true_pos, observations, measurement_noise, confidence
#     )
#     est_dists, est_r, est_c = _assign_and_distances(
#         true_pos, estimates, measurement_noise, confidence
#     )

#     def stats(arr):
#         if np.all(np.isnan(arr)):
#             return np.nan, np.nan, np.nan
#         mean = float(np.nanmean(arr))
#         mse  = float(np.nanmean(np.square(arr)))
#         rmse = float(np.sqrt(mse))
#         return mean, mse, rmse

#     obs_mean, obs_mse, obs_rmse = stats(obs_dists)
#     est_mean, est_mse, est_rmse = stats(est_dists)

#     return {
#         'obs_dists':  obs_dists,
#         'est_dists':  est_dists,
#         'obs_mean':   obs_mean,
#         'obs_mse':    obs_mse,
#         'obs_rmse':   obs_rmse,
#         'est_mean':   est_mean,
#         'est_mse':    est_mse,
#         'est_rmse':   est_rmse,
#         'obs_assign': (obs_r, obs_c) if obs_r is not None else None,
#         'est_assign': (est_r, est_c) if est_r is not None else None,
#     }


# def get_stats(true_trajectory, observations, history, num_steps, average_time, measurement_noise, confidence=0.95):
#     """
#     Aggregate per-frame statistics over the full trajectory.

#     Survivorship bias fix: obs and est distances are only compared on frames
#     where BOTH have a valid (non-NaN) assignment for the same target, so the
#     two error metrics are computed on identical sets of targets and frames.

#     Args:
#         true_trajectory:    list of (n, 4) arrays
#         observations:       list of (m, 2) arrays or Nones
#         history:            list of dicts with key 'estimate'
#         num_steps:          int
#         average_time:       float, seconds per step
#         measurement_noise:  scalar sigma — passed through to the chi-squared gate
#         confidence:         chi2 gate confidence level (default 0.95)

#     Returns dict with aggregate metrics and raw data.
#     """
#     all_obs_dists = []
#     all_est_dists = []

#     for t in range(num_steps):
#         true = true_trajectory[t]
#         obs  = observations[t]
#         est  = history[t]['estimate']

#         res = get_distance_one_point(true, obs, est, measurement_noise, confidence)

#         obs_d = res['obs_dists']
#         est_d = res['est_dists']

#         # Only include targets where BOTH observation and estimate are valid.
#         # This prevents survivorship bias: we compare on the same set of
#         # targets/frames so neither metric gets an unfair advantage.
#         valid = ~np.isnan(obs_d) & ~np.isnan(est_d)
#         all_obs_dists.extend(obs_d[valid])
#         all_est_dists.extend(est_d[valid])

#     all_est_dists = np.asarray(all_est_dists)
#     all_obs_dists = np.asarray(all_obs_dists)

#     est_rmse = np.sqrt(np.mean(np.square(all_est_dists)))
#     obs_rmse = np.sqrt(np.mean(np.square(all_obs_dists)))

#     est_mean_error = np.mean(all_est_dists)
#     obs_mean_error = np.mean(all_obs_dists)

#     # RMSE improvement: how much better is the filter than raw observations (%)
#     rmse_improvement = 100 * (obs_rmse - est_rmse) / obs_rmse if obs_rmse > 0 else float('inf')

#     # MSE improvement: same but in squared-error space (different scale, both useful)
#     obs_mse = obs_rmse ** 2
#     est_mse = est_rmse ** 2
#     mse_improvement = 100 * (obs_mse - est_mse) / obs_mse if obs_mse > 0 else float('inf')

#     return {
#         'est_mean_error':  est_mean_error,
#         'est_rmse':        est_rmse,
#         'obs_mean_error':  obs_mean_error,
#         'obs_rmse':        obs_rmse,
#         'mse_improvement': mse_improvement,
#         'rmse_improvement': rmse_improvement,
#         'raw': {
#             'true_trajectory': true_trajectory,
#             'observations':    observations,
#             'history':         history,
#             'num_steps':       num_steps,
#         },
#         'average_time': average_time,
#     }

# Begin of Evaluator 3
