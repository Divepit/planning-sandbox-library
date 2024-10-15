import os
import numpy as np
import pickle

from planning_sandbox.environment_class import Environment
from imitation.data.types import Trajectory

from imitation.data import types

dir_path = os.path.dirname(os.path.realpath(__file__))

# Function to load existing trajectories if the file exists
def load_existing_trajectories(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return []

# Function to save trajectories to a file
def save_trajectories(filename, trajectories):
    with open(filename, 'wb') as f:
        pickle.dump(trajectories, f)

# Variables
filename = dir_path+'/trajectories.pkl'
amount_of_trajectories = 100000  # Adjust based on your need
autosave_after = 100  # Adjust based on your need
size = 100  # Example size
num_agents = 3  # Example number of agents
num_goals = 5  # Example number of goals
num_skills = 2  # Example number of skills
solve_type = 'optimal'  # Example solve type

# Initialize environment
sandboxEnv = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True, solve_type=solve_type)

# Load existing trajectories if any
trajectories = load_existing_trajectories(filename)
print(f"Loaded {len(trajectories)} existing trajectories.")

try:
    # Infinite loop to allow continuous generation of trajectories
    while True:
        for _ in range(amount_of_trajectories):
            print(f"Progress: {len(trajectories)+1}/{amount_of_trajectories} (Total: {len(trajectories)})", end='\r')
            obs_1 = sandboxEnv.get_observation_vector()
            dict_obs_1 = types.DictObs(obs_1)
            sandboxEnv.find_numerical_solution(solve_type=solve_type)
            obs_2 = sandboxEnv.get_observation_vector()
            dict_obs_2 = types.DictObs(obs_2)
            act = sandboxEnv.get_action_vector()
            new_trajectory = Trajectory(
                obs=np.array([dict_obs_1, dict_obs_2]),
                acts=np.array([act]),
                infos=[{} for _ in range(1)],
                terminal=True
            )
            trajectories.append(new_trajectory)
            sandboxEnv.reset()

            # Periodically save to avoid data loss
            if len(trajectories) % autosave_after == 0:  # Adjust this based on how often you want to save
                save_trajectories(filename, trajectories)
                # print(f"\nAutosaved {len(trajectories)} trajectories so far.")

except KeyboardInterrupt:
    # Save current progress when interrupted by Ctrl + C
    save_trajectories(filename, trajectories)
    print(f"\nProcess interrupted. {len(trajectories)} trajectories saved to {filename}.")