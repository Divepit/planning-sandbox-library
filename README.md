# Multi-Agent Planning and Reinforcement Learning Environment

## Overview

This project implements a multi-agent planning and reinforcement learning environment for task allocation and path planning. It includes a custom grid-based environment, reinforcement learning algorithms, and benchmarking tools for performance evaluation.

## Features

- Custom grid-based environment with agents, goals, and obstacles
- Support for multi-agent scenarios with skill-based task allocation
- Reinforcement Learning (RL) implementation using Proximal Policy Optimization (PPO)
- Numerical solving for optimal solutions
- Visualization tools for environment and agent behavior
- Benchmarking scripts for performance analysis

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:Divepit/planning-sandbox-library.git
   cd planning-sandbox-library
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: If you're using GPU acceleration, you may need to install PyTorch separately. Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions specific to your system.

## Project Structure

- `planning_sandbox/`: Core environment classes and utilities
  - `environment_class.py`: Main environment implementation
  - `agent_class.py`: Agent implementation
  - `goal_class.py`: Goal implementation
  - `grid_map_class.py`: Grid map and obstacle handling
  - `scheduler_class.py`: Task scheduling
  - `visualiser_class.py`: Environment visualization
- `RL/`: Reinforcement Learning implementation
  - `RL_mover_env.py`: RL environment wrapper
  - `RL_SP_env.py`: Shortest path RL environment
  - `train_GPU.py`: GPU-accelerated training script
  - `train_RL.py`: CPU training script
  - `visualize_model.py`: Script to visualize trained models
- `BruteForce/`: Numerical Solving / Combinatorially Solving the Problem
  - `numerical_solving_benchmark.py`: Benchmarking script for numerical solutions
  - `numerical_solving.py`: Implementation of numerical solving methods

## Usage

### Running Numerical Solving

To run the numerical solving simulation:

```
python BruteForce/numerical_solving.py
```

This will initialize the environment and run the simulation using the numerical solving approach.

### Training the RL Model

To train the RL model using CPU:

```
python RL/train_RL.py
```

For GPU-accelerated training:

```
python RL/train_GPU.py
```

### Visualizing the Trained Model

To visualize the behavior of a trained model:

```
python RL/visualize_model.py
```

### Running Benchmarks

To run benchmarks and generate performance metrics:

```
python BruteForce/numerical_solving_benchmark.py
```

This will generate benchmark results and save them as CSV files and plots.

## Customization

You can customize various parameters of the environment and training process by modifying the respective scripts. Key parameters include:

- Number of agents
- Number of goals
- Grid size
- Number of skills
- Number of obstacles
- Training hyperparameters (in `train_RL.py` and `train_GPU.py`)

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Create a pull request

## Contact

[Your Name] - [marcotr@ethz.ch]

Project Link: [https://github.com/Divepit/planning-sandbox-library](https://github.com/Divepit/planning-sandbox-library)