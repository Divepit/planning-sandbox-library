# Multi-Agent Planning and Reinforcement Learning Environment

## Overview

This project implements a multi-agent planning and reinforcement learning environment for task allocation and path planning. It includes a custom grid-based environment, reinforcement learning algorithms, and benchmarking tools for performance evaluation.

## Features

- Custom grid-based environment with agents, goals
- Support for multi-agent scenarios with skill-based task allocation
- Reinforcement Learning (RL) implementation using Stable Baselines 3.
- Numerical solving for optimal solutions
- Visualization tools for environment and agent behavior
- Benchmarking scripts for performance analysis

## Installation

1. Clone the repository:
   ```sh
   git clone git@github.com:Divepit/planning-sandbox-library.git
   cd planning-sandbox-library
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -e .
   ```

## Project Structure

- `planning_sandbox/`: Core environment classes and utilities
  - `maps/`:
    - contains a couple of different elevation maps that can be used for the environments
  - `environment_class.py`: Main environment implementation
  - `agent_class.py`: Agent implementation
  - `goal_class.py`: Goal implementation
  - `grid_map_class.py`: Grid map
  - `scheduler_class.py`: Task scheduling
  - `visualiser_class.py`: Environment visualization
  - `benchmark_class.py`: Benchmarking time taken for tasks
- `ImitationLearning/`: Reinforcement Learning implementation
  - `IL_env.py`: RL / Imitation Learning environment wrapper (Stable Baselines 3 / gymnasium)
  - `IL_training.py`: Training script
  - `IL_replay.py`: Continuous replay of latest autosave of trained model
  - `IL_data_collection.py`: Continuous generation of optimal/expert solutions for Imitation Learning
  - `IL_behavioural_cloning.py`: (WIP) Actual imitation learning implementation
- `BruteForce/`: Numerical Solving / Combinatorially Solving the Problem
  - `numerical_solving.py`: Example script to use the above tools for algorithm comparison.

## Usage

I suggest you use the script under `Examples/simple_numerical.py` to follow the coming steps. The script is more or less analog to what is described below and you can test the differences visually like that.

### Create a new environment

When creating the environment, we define the grid size (only square grids size x size), the number of agents and the number of goals as well as the number of skills in the game. The environment will randomly distribute skills between agents and goals but it will ensure that the problem is solvable.

```python
my_environment = Environment(size=200, num_agents=3, num_goals=5, num_skills=2, use_geo_data=True)
```

Note that the `use_geo_data` input determines whether an elevation map will be used or not. If we set this to false, we will be working with a flat, empty map with no obstacles and the weight of the travelled paths will just be the sum of the steps taken to pass from starting positions to goal.
   
### Solve numerically

First we have the environment generate a solution:

```python
my_environment.find_numerical_solution(solve_type='optimal')
```

There are two main algorithms (solve types) for solving the environment numerically - `'fast'` and `'optimal'`. 

#### Fast algorithm:

The fast algorithm works along these lines:

- Check which goal is the cheapest to solve right now and which agents will be occupied
- Check which agents are free and which is the next cheapest goal that can be solved

It repeats this until all agents are occupied. As soon as a goal is claimed, this process starts over.

Since this algorithm just checks the next cheapest goal, it can support a large number of agents and goals but it is not optimal, since it does not consider that claiming one goal before the other could have better performance results.

#### Optimal algorithm:

This is simply a brute-force solving of every possible solution in the system. It calculates every path through every goal for every combination of agents and thus the computational requirements rise exponentially with the amount of agents and goals.


### Work with the solution

By using the `find_numerical_solution()` function, we have now populated our `my_environment.full_solution` member variable. It now contains a Dict with agents as keys. Each agent in the dict has a list of goals as values. This format is what is called a 'solution' in this repo. When we run the solution, each agent will simply visit each goal in its list in the order it was inserted (FIFO).

#### Step through the solution

We can have the environment go through the full solution step-by-step without visualising the process. This will have the agents step through all their scheduled goals and count their steps taken and waited, accumulate cost and count the amount of claimed goals. The function returns us the sum for these stats of all agents.

```python
total_steps, steps_waited, total_cost, solve_time, amount_of_claimed_goals = my_environment.solve_full_solution()
```

Furthermore we can now check whether all goals have been claimed:

```python
all_goals_claimed_bool = my_environment.scheduler.all_goals_claimed()
```

Sometimes we don't need to have all of the stats but only need to know whether a solution actually leads to all goals being claimed. The `solve_full_solution()` function by default is, however, rather slow as depending on the algorithm used, it has to calculate a lot of a_star paths. We can bypass this by passing the argument `fast=True`, so:

```python
total_steps, steps_waited, total_cost, solve_time, amount_of_claimed_goals = my_environment.solve_full_solution(fast=True)
```

In this case, `total_steps, steps_waited, total_cost = 0` will all not be populated, since instead of actually having the agents walk their paths to their goals, we simply teleport them there. This results to the agents not actually walking any steps but since they will arrive at their goals in the right order, we can actually check whether all goals can be claimed with the current solution's goal orders distributed over the agents.

In the future we might have a function that can numerically determine whether the solution leads to all goals being claimed but for the time being, this is all we have :^)

#### Get the solution cost only

In most cases, we just want the cost of the solution itself, we do this by calling:

```python
my_solution = my_environment.full_solution
total_cost = my_environment.calculate_cost_of_closed_solution(my_solution)
```

The cost of the solution is calculated using the slope of the terrain as well as the distance travelled. The weight calculation is defined in the grid_map_class in the function `_calculate_weight()`. At the time of writing, the forumla used is `slope + 0.1 * MPP` where the slope is simply the slope between two consecutive grid squares and MPP stands for meters per pixel (or meters per grid square).

#### Resetting

There are two reset options:

- `my_environment.soft_reset()`: Simply resets agents to their initial positions and clears all running variables. With this you can reuse the same environment with the same starting conditions and the same agents, goals and skills and start from scratch. This can be used to compare multiple algorithms on the same environment.
- `my_environment.reset()`: Deletes all agents and goals and sets up a completely new problem. This is still faster than making a new environment since the base map and graph does not have to be regenerated.

### Visualisation

When a solution is in the system (`my_environment.full_solution` is populated) we can visualise this solution.

```python
my_visualizer: Visualizer = Visualizer(my_environment, speed=speed)
my_visualizer.visualise_full_solution()
```

Apart from the visualisation, this process is equivalent to:

```python
my_environment.soft_reset()
my_environment.solve_full_solution()
```

but it will simply show you everything that happens on the map.

## Customization

You can customize various parameters of the environment and training process by modifying the respective scripts. Key parameters include:

- Number of agents
- Number of goals
- Grid size
- Number of skills

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Create a pull request

## Contact

[Marco Trentini] - [marcotr@ethz.ch]

Project Link: [https://github.com/Divepit/planning-sandbox-library](https://github.com/Divepit/planning-sandbox-library)