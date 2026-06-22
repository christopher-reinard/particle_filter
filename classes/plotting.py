from matplotlib import pyplot as plt
import os
import io
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

def plot_particles_at_time(true_trajectory, history, time):
    particles = history[time]["particle_states"][:, :2]
    observation = history[time]["observation"]
    true_points = true_trajectory[time][:, :2]
    estimations = history[time]["estimate"][:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(particles[:, 0], particles[:, 1], alpha=0.5, label="Particles")
    plt.scatter(true_points[:, 0], true_points[:, 1], color="green", marker="*", s=200, label="True Positions")
    plt.scatter(estimations[:, 0], estimations[:, 1], color="orange", marker="o", s=100, label="GMM Estimates")
    if observation is not None:
        plt.scatter(observation[:, 0], observation[:, 1], color="red", marker="x", s=100, label="Observations")
    plt.title(f"Particle Distribution at Time Step {time}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()

def plot_particle_filter_step(true_trajectory, observations, history, time, num_steps=None, save_path=None, title=None, bins=80):
    """Plot one particle-filter time step with trajectory context and particle diagnostics."""
    time = int(time)
    if time < 0 or time >= len(history):
        raise IndexError(f"time must be in [0, {len(history) - 1}], got {time}")

    true_xy = np.asarray(true_trajectory)
    if true_xy.ndim == 2:
        true_xy = true_xy[:, None, :2]
    else:
        true_xy = true_xy[:, :, :2]

    stop = time + 1
    estimate_xy = np.stack([
        np.atleast_2d(np.asarray(step["estimate"]))[:, :2]
        for step in history[:stop]
    ])
    particles = np.asarray(history[time]["particle_states"])[:, :2]

    def observation_at(frame):
        if frame < 0:
            return None
        obs = observations[frame] if observations is not None else history[frame].get("observation")
        if obs is None:
            return None
        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs[None, :]
        return obs[:, :2]

    prev_time = time - 1 if time > 0 else None
    current_observation = observation_at(time)
    previous_observation = observation_at(prev_time) if prev_time is not None else None
    current_estimate = estimate_xy[-1]
    previous_estimate = estimate_xy[-2] if prev_time is not None else None
    previous_particles = np.asarray(history[prev_time]["particle_states"])[:, :2] if prev_time is not None else None

    point_sets = [
        true_xy[:stop].reshape(-1, 2),
        estimate_xy.reshape(-1, 2),
        particles,
    ]
    if previous_particles is not None:
        point_sets.append(previous_particles)
    if current_observation is not None:
        point_sets.append(current_observation)
    if previous_observation is not None:
        point_sets.append(previous_observation)
    points = np.vstack([p for p in point_sets if p.size])
    points = points[np.isfinite(points).all(axis=1)]
    if points.size:
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        span = max(x_max - x_min, y_max - y_min, 1.0)
        margin = span * 0.08
        x_limits = (x_min - margin, x_max + margin)
        y_limits = (y_min - margin, y_max + margin)
    else:
        x_limits = (-1.0, 1.0)
        y_limits = (-1.0, 1.0)

    fig_traj, ax_traj = plt.subplots(figsize=(16, 8))
    fig_density = plt.figure(figsize=(16, 8))
    ax_density = fig_density.add_subplot(111, projection="3d")
    fig_cloud, ax_cloud = plt.subplots(figsize=(16, 8))
    figs = (fig_traj, fig_density, fig_cloud)
    palette = plt.cm.tab20.colors

    for target_idx in range(true_xy.shape[1]):
        color = palette[target_idx % len(palette)]
        ax_traj.plot(
            true_xy[:stop, target_idx, 0],
            true_xy[:stop, target_idx, 1],
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.55,
            label="Truth trajectory" if target_idx == 0 else None,
        )

    for estimate_idx in range(estimate_xy.shape[1]):
        color = palette[estimate_idx % len(palette)]
        ax_traj.plot(
            estimate_xy[:, estimate_idx, 0],
            estimate_xy[:, estimate_idx, 1],
            color=color,
            linewidth=1.8,
            alpha=0.85,
            label="Estimate trajectory" if estimate_idx == 0 else None,
        )

    def add_step_markers(ax):
        if prev_time is not None:
            ax.scatter(
                true_xy[prev_time, :, 0],
                true_xy[prev_time, :, 1],
                color="green",
                marker="*",
                s=100,
                alpha=0.35,
                label="Truth t-1",
                zorder=5,
            )
            ax.scatter(
                previous_estimate[:, 0],
                previous_estimate[:, 1],
                facecolors="none",
                edgecolors="orange",
                marker="o",
                s=100,
                linewidths=1.4,
                alpha=0.75,
                label="Estimate t-1",
                zorder=6,
            )
            if previous_observation is not None:
                ax.scatter(
                    previous_observation[:, 0],
                    previous_observation[:, 1],
                    color="red",
                    marker="+",
                    s=100,
                    alpha=0.6,
                    label="Observation t-1",
                    zorder=7,
                )

        ax.scatter(
            true_xy[time, :, 0],
            true_xy[time, :, 1],
            color="green",
            marker="*",
            s=180,
            edgecolors="black",
            linewidths=0.5,
            label="Truth t",
            zorder=8,
        )
        ax.scatter(
            current_estimate[:, 0],
            current_estimate[:, 1],
            color="orange",
            marker="o",
            s=90,
            edgecolors="black",
            linewidths=0.5,
            label="Estimate t",
            zorder=9,
        )
        if current_observation is not None:
            ax.scatter(
                current_observation[:, 0],
                current_observation[:, 1],
                color="red",
                marker="x",
                s=110,
                linewidths=1.8,
                label="Observation t",
                zorder=10,
            )

    def add_step_markers_3d(ax, z_level):
        if prev_time is not None:
            ax.scatter(
                true_xy[prev_time, :, 0],
                true_xy[prev_time, :, 1],
                z_level,
                color="green",
                marker="*",
                s=100,
                alpha=0.35,
                label="Truth t-1",
            )
            ax.scatter(
                previous_estimate[:, 0],
                previous_estimate[:, 1],
                z_level,
                facecolors="none",
                edgecolors="orange",
                marker="o",
                s=100,
                linewidths=1.4,
                alpha=0.75,
                label="Estimate t-1",
            )
            if previous_observation is not None:
                ax.scatter(
                    previous_observation[:, 0],
                    previous_observation[:, 1],
                    z_level,
                    color="red",
                    marker="+",
                    s=100,
                    alpha=0.6,
                    label="Observation t-1",
                )

        ax.scatter(
            true_xy[time, :, 0],
            true_xy[time, :, 1],
            z_level,
            color="green",
            marker="*",
            s=180,
            edgecolors="black",
            linewidths=0.5,
            label="Truth t",
        )
        ax.scatter(
            current_estimate[:, 0],
            current_estimate[:, 1],
            z_level,
            color="orange",
            marker="o",
            s=90,
            edgecolors="black",
            linewidths=0.5,
            label="Estimate t",
        )
        if current_observation is not None:
            ax.scatter(
                current_observation[:, 0],
                current_observation[:, 1],
                z_level,
                color="red",
                marker="x",
                s=110,
                linewidths=1.8,
                label="Observation t",
            )

    add_step_markers(ax_traj)

    max_density = 1.0
    if len(particles):
        density, x_edges, y_edges = np.histogram2d(
            particles[:, 0],
            particles[:, 1],
            bins=bins,
            range=[x_limits, y_limits],
        )
        previous_density = None
        if previous_particles is not None and len(previous_particles):
            previous_density, _, _ = np.histogram2d(
                previous_particles[:, 0],
                previous_particles[:, 1],
                bins=[x_edges, y_edges],
            )
        max_density = max(
            float(density.max()),
            float(previous_density.max()) if previous_density is not None else 1.0,
            1.0,
        )
        x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
        y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
        surface = ax_density.plot_surface(
            x_grid,
            y_grid,
            density,
            cmap="YlGn",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        if previous_density is not None:
            ax_density.plot_wireframe(
                x_grid,
                y_grid,
                previous_density,
                color="royalblue",
                linewidth=0.8,
                alpha=0.7,
                rstride=2,
                cstride=2,
                label="Density t-1",
            )
        fig_density.colorbar(surface, ax=ax_density, shrink=0.72, pad=0.08, label="Particles/bin")
    add_step_markers_3d(ax_density, max_density * 1.08)
    ax_density.set_zlim(0, max_density * 1.18)
    ax_density.set_zlabel("Particle density")
    ax_density.view_init(elev=28, azim=-58)
    ax_density.set_facecolor("#edf7e7")

    ax_cloud.scatter(
        particles[:, 0],
        particles[:, 1],
        color="steelblue",
        s=7,
        alpha=0.08,
        linewidths=0,
        rasterized=True,
        label="Particles",
    )
    add_step_markers(ax_cloud)

    ax_traj.set_title(f"Trajectories through t={time}")
    ax_density.set_title("Particle landscape")
    ax_cloud.set_title("Particle cloud")
    if title:
        fig_traj.suptitle(title, fontsize=14, fontweight="bold")
        fig_density.suptitle(title, fontsize=14, fontweight="bold")
        fig_cloud.suptitle(title, fontsize=14, fontweight="bold")

    for ax in (ax_traj, ax_cloud):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(fontsize=8, loc="best")

    ax_density.set_xlim(*x_limits)
    ax_density.set_ylim(*y_limits)
    ax_density.set_xlabel("X Position")
    ax_density.set_ylabel("Y Position")
    ax_density.legend(fontsize=8, loc="best")

    for fig in figs:
        fig.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        ext = ext or ".png"
        fig_traj.savefig(f"{base}_trajectories{ext}", bbox_inches="tight")
        fig_density.savefig(f"{base}_landscape{ext}", bbox_inches="tight")
        fig_cloud.savefig(f"{base}_cloud{ext}", bbox_inches="tight")
    else:
        plt.show()
    return figs, (ax_traj, ax_density, ax_cloud)

def animate_particle_filter(true_trajectory, history, save_path="particle_filter_animation.gif"):
    fig, ax = plt.subplots(figsize=(16, 8))

    # Compute global axis limits for a stable view across frames
    true_trajectory_xy = true_trajectory[:, :, :2]
    estimated_trajectory = np.stack([h["estimate"][:, :2] for h in history])
    all_particles = np.concatenate([h["particle_states"][:, :2] for h in history])
    all_true_points = true_trajectory_xy.reshape(-1, 2)
    all_estimates = estimated_trajectory.reshape(-1, 2)
    observed_points = [obs[:, :2] for obs in (h["observation"] for h in history) if obs is not None]
    all_observations = np.concatenate(observed_points) if observed_points else np.empty((0, 2))
    all_points = np.concatenate([all_particles, all_true_points, all_estimates, all_observations], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    margin = 1.0

    for target_idx in range(true_trajectory_xy.shape[1]):
        ax.plot(
            true_trajectory_xy[:, target_idx, 0],
            true_trajectory_xy[:, target_idx, 1],
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.2,
            label="True Trajectory" if target_idx == 0 else None,
        )
    for estimate_idx in range(estimated_trajectory.shape[1]):
        ax.plot(
            estimated_trajectory[:, estimate_idx, 0],
            estimated_trajectory[:, estimate_idx, 1],
            color="orange",
            linewidth=1.5,
            alpha=0.2,
            label="Estimated Trajectory" if estimate_idx == 0 else None,
        )

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    empty_offsets = np.empty((0, 2))
    particle_plot = ax.scatter([], [], alpha=0.01, label="Particles")
    true_plot = ax.scatter([], [], color="green", marker="*", s=200, label="True Positions")
    estimate_plot = ax.scatter([], [], color="orange", marker="o", s=100, label="GMM Estimates")
    observation_plot = ax.scatter([], [], color="red", marker="x", s=100, label="Observations")
    title = ax.set_title("")
    ax.legend()

    def update(time):
        particles = history[time]["particle_states"][:, :2]
        observation = history[time]["observation"]
        particle_plot.set_offsets(particles)
        true_plot.set_offsets(true_trajectory_xy[time])
        estimate_plot.set_offsets(estimated_trajectory[time])
        if observation is not None:
            observation_plot.set_offsets(observation[:, :2])
            observation_plot.set_visible(True)
        else:
            observation_plot.set_offsets(empty_offsets)
            observation_plot.set_visible(False)

        title.set_text(f"Particle Distribution at Time Step {time}")
        return particle_plot, true_plot, estimate_plot, observation_plot, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(len(history)),
        interval=200,       # milliseconds between frames
        repeat=True
    )

    ani.save(save_path, writer="pillow", fps=2)
    plt.close(fig)
    
    print(f"Animation saved to {save_path}")
    return ani


def plot_sim_n_balls_point_prediction(true_trajectory, observations, history, dropout_start=-1, dropout_end=-1, save_path=None, title=None):
    plt.figure(figsize=(16, 8))

    # Dynamically extract the number of targets from the ground truth array
    n_objects = true_trajectory.shape[1]

    # Pull a dynamic color palette from Matplotlib (supports up to 20 distinct colors natively)
    palette = plt.cm.tab20.colors

    # 1. Plot Ground Truth Trajectories (dashed lines for reference)
    for b in range(n_objects):
        # Use modulo to wrap around if tracking more than 20 objects
        color = palette[b % len(palette)]

        ball_traj = true_trajectory[:, b, :]
        plt.plot(ball_traj[:, 0], ball_traj[:, 1], linestyle="--", color=color, alpha=0.4, label=f"Target {b} True Path")
        plt.scatter(ball_traj[0, 0], ball_traj[0, 1], color=color, marker="o", s=100, edgecolors='black', zorder=4)

    # 2. Plot Noisy Observations as single background markers
    obs_x, obs_y = [], []
    for obs in observations:
        if obs is not None:
            for target_obs in obs:
                obs_x.append(target_obs[0])
                obs_y.append(target_obs[1])
    plt.scatter(obs_x, obs_y, marker="x", color="red", alpha=0.3, label="Sensor Detections")

    # 3. Plot GMM Point Estimates by Track Index
    # Initialize dynamic lists for n tracks
    track_x = [[] for _ in range(n_objects)]
    track_y = [[] for _ in range(n_objects)]

    for step in history:
        means = step["estimate"]
        if means is not None:
            for track_idx, component in enumerate(means):
                if track_idx < n_objects: # Safety bound
                    track_x[track_idx].append(component[0])
                    track_y[track_idx].append(component[1])

    # Plot each track with its corresponding distinct color
    idx = 0
    for track_idx in range(n_objects):
        color = palette[track_idx % len(palette)]

        # Plot the connecting line (Solid to distinguish from ground truth)
        # plt.plot(track_x[track_idx], track_y[track_idx], color=color, alpha=0.8, linestyle="-", zorder=4)

        # Plot the point estimates and Highlight the Sensor dropouts
        if dropout_start < idx and idx < dropout_end:
          plt.scatter(track_x[track_idx], track_y[track_idx], color=color, marker="X", s=35, alpha=1.0, zorder=5, edgecolors='black', linewidths=0.5, label=f"Track {track_idx} Prediction")
        else:
          plt.scatter(track_x[track_idx], track_y[track_idx], color=color, marker="o", s=35, alpha=1.0, zorder=5, edgecolors='black', linewidths=0.5, label=f"Track {track_idx} Prediction")
        idx += 1

    # Dynamic Title
    plt.title(f"Parameters: {title}", fontsize=14, fontweight='bold')
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=":", alpha=0.7)

    # Move legend outside the plot to prevent covering data
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
