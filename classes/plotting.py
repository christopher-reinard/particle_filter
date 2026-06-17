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

def animate_particle_filter(true_trajectory, history, save_path="particle_filter_animation.gif"):
    fig, ax = plt.subplots(figsize=(16, 8))

    # Compute global axis limits for a stable view across frames
    all_particles = np.concatenate([h["particle_states"][:, :2] for h in history])
    all_true_points = true_trajectory[:, :, :2].reshape(-1, 2)
    all_estimates = np.concatenate([h["estimate"][:, :2] for h in history])
    observed_points = [obs[:, :2] for obs in (h["observation"] for h in history) if obs is not None]
    all_observations = np.concatenate(observed_points) if observed_points else np.empty((0, 2))
    all_points = np.concatenate([all_particles, all_true_points, all_estimates, all_observations], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    margin = 1.0

    def update(time):
        ax.clear()
        ax.grid(True, linestyle=":", alpha=0.7)
        particles = history[time]["particle_states"]
        observation = history[time]["observation"]
        true_points = true_trajectory[time][:, :2]
        estimations = history[time]["estimate"][:, :2]

        ax.scatter(particles[:, 0], particles[:, 1], alpha=0.01, label="Particles")
        ax.scatter(true_points[:, 0], true_points[:, 1], color="green", marker="*", s=200, label="True Positions")
        ax.scatter(estimations[:, 0], estimations[:, 1], color="orange", marker="o", s=100, label="GMM Estimates")
        if observation is not None:
            ax.scatter(observation[:, 0], observation[:, 1], color="red", marker="x", s=100, label="Observations")

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_title(f"Particle Distribution at Time Step {time}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()

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


def plot_sim_n_balls_point_prediction(true_trajectory, observations, history, dropout_start, dropout_end):
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
    plt.title(f"Multi-Target Particle Filter: {n_objects}-Target Consistent Identity Tracking", fontsize=14, fontweight='bold')
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=":", alpha=0.7)

    # Move legend outside the plot to prevent covering data
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()
