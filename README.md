# Particle Filter for Ball position prediction

Solution of undistinguishable ball assignment problem for Portfolio 2 of **Reasoning and Decision-Making under Uncertainty**

<img src="img/ParticleFilter_2Balls_LabelSwapping.jpeg">

## Author: Peter Möhle, Christopher Reinard Kohar

# Installation and Testing

1. Install required libraries from requirements.txt
2. Review the Outputs in 'main.ipynb'
3. Alter parameters and execute further cells

# Explanations of Class Files

Below are quick explanations of the purpose of each file in the classes directory

## observation.py

- Contains the Observation_Model and the Transition_Model

### Observation_Model

- Outputs the observations
- Computes the likelihood of the particle given the observations
- Applies noise based on a given noise level

### Transition_Model

- Moves the states to new states
- Physics model given
- Noise model wasnt given:
  - First Idea: Applying the same process_noise on all coordinates ([x,y,vx,vy])
    - Problem: Not Physical correct. Dimensionalities have different units. Noise should be approproate in respect to all Coordinates
  - Later Solution: 'Continuous-Time White Noise Acceleration model'. Process_noise mutliplied according to physics on each coordinate. Noise is more realistic - as in physics behavior.

## Filter Implementation
For the n-balls requirement the core idea was to not have a 4D State per Ball. But instead to have one 4D-Landscape in which balls can be distinguished - even with same coordinates - by the velocity.
4D State per Ball wuold result in a curse of dimensionality, exploding the amounts of balls needed.

## particle_filter.py
First Idea: Use 1 filter for all the balls.
Distinguish them using a GMM.
--> However: To encounter problems about collapsing trajectories we had to focus on the assignment-problem. We therefore developed an alternative solution which has a cleaerer OOP concept.

## particle_filter_multiple.py

Idea: Use n Filters. Where each Filter tracks one ball.
--> 1 Ball solution was always pretty simple to solve for the particle filter. Therefore if we solve the assignment problem good enough it would collapse into Single-Ball-Problems.

- Problem to solve: given n estimates and n observations, how do we assign which filter to each ball?
  - Chosen Solution was to compute distances between each ball (Mahalanobis, Euclidean)
  - Minimize the distance with linear_sum_assignment from scipy

## classifier.py

- Evaluates the likelihood of a point given a gaussian
- Potential alternative distributions could be used. Gaussian however is preferred, as it fits to the given scenario, noise and usage of GMM.

## evaluator.py

- To evaluate the distance of the balls, assigning the balls is necessary.
- Gets the distance between the true location and estimates, then assigns the minimum distance between them as the loss
- When all estimates are at the same location of the true location, distance would be 0

## plotting.py

Contains plotting functions that allow for visualizations and animation to evaluate results

particle_filter.py and particle_filter_multiple.py

## Explanation of Pipeline

## Ideas Explored

1. Use of a single Particle Filter for all balls
   - Exploration of using a Gaussian Mixture Model to predict ball location from the particles
2. Use of Multiple Particle Filters for each ball
   - Use of Mahalanobis Distance

## Tests Run

1. Increasing number of balls (Performance)
