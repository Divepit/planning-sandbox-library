import os
import numpy as np
import h5py
from planning_sandbox.environment_class import Environment

current_directory = os.path.dirname(os.path.abspath(__file__))

# Parameters (adjust as needed)
num_agents = 3
num_goals = 5
num_skills = 2
size = 100
use_geo_data = True

num_samples = 1000000  # Number of samples to generate in this run

# Placeholder functions for data generation
# def generate_input(env: Environment):
#     input_vector = []
#     for agent in env.agents:
#         input_vector.append(agent.position[0])
#         input_vector.append(agent.position[1])
#         for i in range(env.num_skills):
#             input_vector.append(int(i in agent.skills))
#     return np.array(input_vector)

def generate_input(env: Environment):
    goals_map = []
    for goal in env.goals:
        goals_map.append(goal.position[0]/env.size)
        goals_map.append(goal.position[1]/env.size)
    goals_map = np.array(goals_map, dtype=np.float32)
    agents_map = []
    for agent in env.agents:
        agents_map.append(agent.position[0]/env.size)
        agents_map.append(agent.position[1]/env.size)
    agents_map = np.array(agents_map, dtype=np.float32)

    goal_required_skills = np.array([[(1 if skill in goal.required_skills else 0) for skill in range(env.num_skills)] for goal in env.goals], dtype=np.float32).flatten()
    agent_skills = np.array([[(1 if skill in agent.skills else 0) for skill in range(env.num_skills)] for agent in env.agents], dtype=np.float32).flatten()

    observation_vector = np.concatenate((goals_map, agents_map, goal_required_skills, agent_skills))
    return observation_vector

def generate_output(env: Environment):
    env.find_numerical_solution(solve_type='optimal')
    return np.array(env.get_action_vector())

# File to store data
data_filename = current_directory+'/dataset.h5'

# Open HDF5 file in append mode
with h5py.File(data_filename, 'a') as h5f:

    # Initialize or get datasets
    if 'X' in h5f and 'y' in h5f:
        X_dset = h5f['X']
        y_dset = h5f['y']
    else:
        # Generate a sample to get the shape
        env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                          num_skills=num_skills, use_geo_data=use_geo_data)
        env.reset()
        input_array = generate_input(env)
        output_array = generate_output(env)
        input_shape = input_array.shape
        output_shape = output_array.shape

        # Create datasets with maxshape set to None to allow resizing
        X_dset = h5f.create_dataset('X', shape=(0,) + input_shape,
                                    maxshape=(None,) + input_shape, chunks=True, compression="gzip")
        y_dset = h5f.create_dataset('y', shape=(0,) + output_shape,
                                    maxshape=(None,) + output_shape, chunks=True, compression="gzip")

    # Keep track of how many samples we already have
    num_existing_samples = X_dset.shape[0]

    env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                      num_skills=num_skills, use_geo_data=use_geo_data)

    try:
        for i in range(num_samples):
            print(f"Generating sample {num_existing_samples + i + 1}", end='\r')
            env.reset()
            input_array = generate_input(env)
            output_array = generate_output(env)

            # Resize datasets to accommodate new data
            X_dset.resize((num_existing_samples + i + 1, ) + X_dset.shape[1:])
            y_dset.resize((num_existing_samples + i + 1, ) + y_dset.shape[1:])

            # Add new data
            X_dset[num_existing_samples + i] = input_array
            y_dset[num_existing_samples + i] = output_array

    except KeyboardInterrupt:
        print("\nData generation interrupted by user. Data saved to file.")

    else:
        print("\nData generation completed. Data saved to file.")