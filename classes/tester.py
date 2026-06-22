from classes.particle_filter import ParticleFilter
from classes.particle_filter_multiple import MultiObjectParticleFilter
from classes.observation import TransitionModel, ObservationModel
from classes.simulator import create_ground_truth, generate_random_balls, create_ground_truth_n_balls
from classes.evaluator import get_stats
from classes.plotting import animate_particle_filter, plot_sim_n_balls_point_prediction

import numpy as np
from typing import Literal

step_size = 0.05
num_steps = 120

num_particles = 1000
init_generator = "Sobol"


STATE_BOUNDS = [
    (0.0, 50.0),   # x bounds
    (0.0, 50.0),   # y bounds
    (-30.0, 30.0),   # vx bounds
    (0.0, 40.0)    # vy bounds
]

def create_test_scenario(
        true_states,
        dropout_start, 
        dropout_end, 
        process_noise,
        measurement_noise, 
        ):
    transition_model = TransitionModel(delta_t=step_size, process_noise=process_noise)
    observation_model = ObservationModel("Gaussian", measurement_noise=measurement_noise)
    true_trajectory, observations = create_ground_truth_n_balls(
        num_steps, dropout_start, dropout_end, true_states, transition_model, observation_model
    )
    return true_trajectory, observations, transition_model, observation_model

def run_one_test(true_states,
                dropout_start, 
                dropout_end, 
                process_noise,
                measurement_noise, 
                num_particles=1000,
                state_bounds=STATE_BOUNDS,
                neighbor_assignment="Hungarian",
                init_generator="Sobol",
                ess_resample_threshold=0.5,
                use_velocity_likelihood=False,
                velocity_sigma=20.0,
                min_velocity_likelihood=0.01,
                save_path=None,
                model: Literal["MultObjectParticleFilter", "SingleParticleFilter"]="MultObjectParticleFilter"):
    
    n_objects = len(true_states)
    true_trajectory, observations, transition_model, observation_model = create_test_scenario(
        true_states,
        dropout_start,
        dropout_end,
        process_noise,
        measurement_noise
    )

    if model == "MultiObjectParticleFilter":
        pf = MultiObjectParticleFilter(
            num_particles=num_particles,
            n_balls=n_objects,
            state_bounds=state_bounds,
            transition_model=transition_model,
            observation_model=observation_model,
            neighbor_assignment=neighbor_assignment,
            init_generator=init_generator,
            ess_resample_threshold=ess_resample_threshold,
            use_velocity_likelihood=use_velocity_likelihood,
            velocity_sigma=velocity_sigma,
            min_velocity_likelihood=min_velocity_likelihood
        )

        history = pf.run(observations, logs=[]) # Don't log for the test, we just want the final estimates
    
    elif model == "SingleParticleFilter":
        pf = ParticleFilter(
            num_particles=num_particles,
            state_bounds=state_bounds,
            transition_model=transition_model,
            observation_model=observation_model,
            init_generator=init_generator,
            roughening_noise=0.0
        )
        
        history = pf.run(
            observations=observations, 
            n_objects=n_objects,
            change_resample_order=True,
            logs=[]
        )
    else:
        raise ValueError("Unknown Model-type")
   
    if save_path:
        print(f"Saving plot to {save_path}")
        plot_sim_n_balls_point_prediction(true_trajectory, 
                                          observations, 
                                          history, 
                                          dropout_start,
                                          dropout_end,
                                          save_path)
        print("Animating particle filter...")
        animate_particle_filter(true_trajectory, history, save_path=save_path.replace(".png", ".gif"))

    return get_stats(true_trajectory, observations, history, num_steps)

"""
ParticleFilterTester: a small experiment-runner class built around your
existing `run_one_test` function.

The idea:
    - `default_parameters` is a dict of everything that stays FIXED across
      all runs in an experiment (state_bounds, transition_model, observations,
      true_trajectory, num_particles, etc.)
    - A "sweep" varies one (or more) named parameters across a list/tuple of
      values, holding everything else at the default. One call to
      `run_one_test` happens per value, and the returned stats dict (from
      `get_stats`) is collected into a pandas DataFrame so you can directly
      compare e.g. neighbor_assignment="Hungarian" vs "GreedyKNN".

Usage
-----
    defaults = dict(
        num_particles=1000,
        n_objects=3,
        state_bounds=STATE_BOUNDS,
        transition_model=transition_model,
        observation_model=observation_model,
        init_generator="Sobol",
        roughening_noise=0.0,
        observations=observations,
        true_trajectory=true_trajectory,
    )

    tester = ParticleFilterTester(defaults, save_dir="plots")

    # vary a single parameter
    results = tester.sweep("neighbor_assignment", ["Hungarian", "GreedyKNN"])

    # vary multiple parameters together, index-wise (like zip)
    results = tester.sweep_multi(
        {"neighbor_assignment": ["Hungarian", "GreedyKNN"],
         "roughening_noise":    [0.0, 0.05]},
        mode="zip",
    )

    # vary multiple parameters as a full grid (every combination)
    results = tester.sweep_multi(
        {"neighbor_assignment": ["Hungarian", "GreedyKNN"],
         "num_particles":       [500, 1000, 2000]},
        mode="grid",
    )

    tester.results_df()   # all runs so far, as a DataFrame
"""

import itertools
from typing import Any, Dict, Iterable, List, Optional, Literal
import numpy as np
import pandas as pd


class ParticleFilterTester:
    """
    Wraps a single-test function (default: `run_one_test`) with a fixed set
    of default parameters, and provides sweep utilities that vary one or
    more parameters while keeping everything else constant.
    """

    def __init__(self,
                 default_parameters: Dict[str, Any],
                 run_fn=None,
                 save_dir: Optional[str] = None,
                 model: Literal["MultObjectParticleFilter", "SingleParticleFilter"]="MultObjectParticleFilter"):
        """
        Args:
            default_parameters: dict of kwargs to pass to `run_fn` for every
                test, unless overridden by a sweep.
            run_fn: the function that runs a single test and returns a stats
                dict, e.g. your `run_one_test`. Must accept the keys in
                `default_parameters` (plus overrides) as kwargs, and a
                `save_path` kwarg for optional plotting.
                Defaults to `run_one_test` from this module's globals if not
                given explicitly (set it once, or pass it in).
            save_dir: if given, each run's plot is saved here as
                f"{save_dir}/{label}.png". If None, no plots are generated.
        """
        if run_fn is None:
            run_fn = globals().get("run_one_test")
            if run_fn is None:
                raise ValueError(
                    "No run_fn given and no `run_one_test` found in scope. "
                    "Pass run_fn=run_one_test explicitly."
                )
        self.run_fn = run_fn
        self.default_parameters = dict(default_parameters)
        self.save_dir = save_dir
        if self.save_dir:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.model = model

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run(self, label: str = "default", seed: Optional[int] = None, **overrides) -> Dict[str, Any]:
        """
        Run one test: defaults merged with `overrides` (overrides win).
        If `seed` is given, np.random.seed(seed) is called right before
        run_fn, so e.g. neighbor_assignment="Hungarian" vs "GreedyKNN" start
        from identical particle initialization/noise draws and are a fair
        head-to-head comparison rather than two independent random runs.
        Stores and returns a flat record: {"label", **overrides, **stats}.
        """
        params = {**self.default_parameters, **overrides}

        save_path = None
        if self.save_dir:
            safe_label = label.replace("/", "_").replace(" ", "_")
            save_path = f"{self.save_dir}/{safe_label}.png"

        if seed is not None:
            np.random.seed(seed)

        stats = self.run_fn(save_path=save_path, model=self.model, **params)

        record = {"label": label, **overrides, **stats}
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------

    def sweep(self,
              test_parameter_name: str,
              test_parameter_values: Iterable[Any],
              seed: Optional[int] = None) -> pd.DataFrame:
        """
        Vary a single parameter across `test_parameter_values`, holding
        everything else at `default_parameters`. One run per value.
        If `seed` is given, np.random.seed(seed) is reset before EVERY run
        in the sweep (same seed each time), so the only thing that differs
        between runs is the parameter being swept -- a fair, controlled
        comparison rather than each run drawing different randomness.
        """
        for value in test_parameter_values:
            label = f"{test_parameter_name}={value}"
            self.run(label=label, seed=seed, **{test_parameter_name: value})
        return self.results_df()

    def sweep_multi(self,
                     test_parameters: Dict[str, Iterable[Any]],
                     mode: str = "zip",
                     seed: Optional[int] = None) -> pd.DataFrame:
        """
        Vary multiple parameters at once.

        mode="zip":  pairs values index-wise across all lists (like zip()).
                     All value lists must be the same length. Use this when
                     you want N specific configurations, e.g. comparing two
                     full presets against each other.
        mode="grid": full cartesian product of every combination. Use this
                     when you want every value of A tested against every
                     value of B.
        If `seed` is given, np.random.seed(seed) is reset before EVERY run.
        """
        names = list(test_parameters.keys())
        value_lists = list(test_parameters.values())

        if mode == "zip":
            lengths = {len(v) for v in value_lists}
            if len(lengths) > 1:
                raise ValueError(
                    f"sweep_multi(mode='zip') requires all value lists to be "
                    f"the same length, got lengths {lengths} for {names}"
                )
            combos = zip(*value_lists)
        elif mode == "grid":
            combos = itertools.product(*value_lists)
        else:
            raise ValueError("mode must be 'zip' or 'grid'")

        for combo in combos:
            overrides = dict(zip(names, combo))
            label = "_".join(f"{k}={v}" for k, v in overrides.items())
            print(f"Running test: {label}")

            self.run(label=label, seed=seed, **overrides)

        return self.results_df()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results_df(self) -> pd.DataFrame:
        """All runs collected so far, as a DataFrame (one row per run)."""
        return pd.DataFrame(self.results)

    def reset(self) -> None:
        """Clear collected results (defaults are kept)."""
        self.results = []