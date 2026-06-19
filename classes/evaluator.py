from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def _assign_and_distances(true_pos, points, miss_cost=1e6, gate=None):
    """
    Assign `points` to `true_pos` (both Nx2 arrays) minimizing total Euclidean cost.
    Returns distances aligned to true_pos order (np.nan for unmatched).
    """
    n = true_pos.shape[0] #Number of true Targets

    # Returns nans if there is dropout and no points to assign
    if points is None:
        return np.full(n, np.nan), None, None

    pts = np.asarray(points)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts = pts[:, :2]

    # cost matrix (n_true x n_pts)
    cost = cdist(true_pos, pts, metric='euclidean')

    if gate is not None:
        cost = np.where(cost > gate, miss_cost, cost)

    row_ind, col_ind = linear_sum_assignment(cost)
    dists = np.full(n, np.nan)
    for r, c in zip(row_ind, col_ind):
        # If we used a gate and the chosen cost is the miss_cost, treat as unmatched
        if cost[r, c] >= miss_cost:
            dists[r] = np.nan
        else:
            dists[r] = cost[r, c]

    return dists, row_ind, col_ind

def get_distance_one_point(true_states, observations, estimates, gate=None):
    """
    Compute per-target assignment distances for observations and estimates.

    - true_states: (n,4) array (x,y,vx,vy) or (n,2)
    - observations: (m,2) array or None
    - estimates: (k,4) or (k,2) array or None
    - gate: optional float distance threshold to reject far matches

    Returns dict with:
      - 'obs_dists', 'est_dists': arrays length n (aligned to true_states indices), NaN for unmatched
      - 'obs_mean','obs_mse','est_mean','est_mse' : scalars (NaN-safe)
      - 'obs_assign','est_assign': (row_ind, col_ind) tuples from solver (or None)
    """
    true_pos = np.asarray(true_states)[:, :2]

    obs_dists, obs_r, obs_c = _assign_and_distances(true_pos, observations, gate=gate)
    est_dists, est_r, est_c = _assign_and_distances(true_pos, estimates, gate=gate)

    # Replace NaNs in observation distances (caused by sensor dropout)
    # with the mean of the non-NaN observation distances when possible.
    # If all observation distances are NaN (no observations at this frame), keep them as NaN.
    # if not np.all(np.isnan(obs_dists)):
    #     obs_mean_replace = float(np.nanmean(obs_dists))
    #     obs_dists = np.where(np.isnan(obs_dists), obs_mean_replace, obs_dists)

    def stats(arr):
        if np.all(np.isnan(arr)):
            return np.nan, np.nan, np.nan

        mean = float(np.nanmean(arr))
        mse  = float(np.nanmean(np.square(arr)))
        rmse = float(np.sqrt(mse))

        return mean, mse, rmse

    obs_mean, obs_mse, obs_rmse = stats(obs_dists)
    est_mean, est_mse, est_rmse = stats(est_dists)
    return {
        'obs_dists': obs_dists,
        'est_dists': est_dists,
        'obs_mean': obs_mean,
        'obs_mse': obs_mse,
        'obs_rmse': obs_rmse,
        'est_mean': est_mean,
        'est_mse': est_mse,
        'est_rmse': est_rmse,
        'obs_assign': (obs_r, obs_c) if obs_r is not None else None,
        'est_assign': (est_r, est_c) if est_r is not None else None
    }

def get_stats(true_trajectory, observations, history, num_steps):
    all_obs_dists = []
    all_est_dists = []

    for t in range(num_steps):
        true = true_trajectory[t]
        obs  = observations[t]
        est  = history[t]['estimate']

        res = get_distance_one_point(true, obs, est, gate=20.0)

        all_est_dists.extend(
            res['est_dists'][~np.isnan(res['est_dists'])]
        )
        all_obs_dists.extend(
            res['obs_dists'][~np.isnan(res['obs_dists'])])
        
    all_est_dists = np.asarray(all_est_dists)
    all_obs_dists = np.asarray(all_obs_dists)

    est_rmse = np.sqrt(np.mean(np.square(all_est_dists)))
    est_mean_error = np.mean(all_est_dists)

    obs_rmse = np.sqrt(np.mean(np.square(all_obs_dists)))
    obs_mean_error = np.mean(all_obs_dists)

    print("Estimate Mean error:", est_mean_error)
    print("Estimate RMSE:", est_rmse)
    print("Observation Mean error:", obs_mean_error)
    print("Observation RMSE:", obs_rmse)

    return est_mean_error, est_rmse, obs_mean_error, obs_rmse