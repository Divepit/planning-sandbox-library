import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

import pickle

from planning_sandbox.environment_class import Environment

from copy import deepcopy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_environments(env, num_clones):
    seed_envs = []
    print("Clone environments...")
    for _ in range(num_clones):
        print(f"Progress: {len(seed_envs)+1}/{num_clones}", end='\r')
        env.reset()
        seed_envs.append(deepcopy(env))
    return seed_envs

def main():

    save_file = '/Users/marco/Programming/PlanningEnvironmentLibrary/ImitationLearning/expert_data.npz'
    
    iterations = 50
    num_goals: int = 6
    num_agents: int = 3
    size: int = 100
    num_skills: int = 2
    solve_type = 'optimal'
    
    
    logging.info("Setting up environment...")
    env = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True)
    
    seed_envs = clone_environments(env, iterations)

    action_vectors = []
    observation_vectors = []
    
    for env in seed_envs:
        print(f"Progress: {100*(seed_envs.index(env))/iterations:.2f}%", end='\r')
        env_copy: Environment = deepcopy(env)
        env_copy.solve_type = solve_type
        env_copy.find_numerical_solution()
        observation_vector = env_copy.get_observation_vector()
        action_vector = env_copy.get_action_vector()
        action_vectors.append(action_vector)
        observation_vectors.append(observation_vector)

    # Saving data code generated using ChatGPT (https://chatgpt.com/share/6703dfb2-b59c-8008-8f42-dae07eea9b96)

    # Check if save file already exists
    pkl_save_file = save_file.replace('.npz', '.pkl')
    if os.path.exists(pkl_save_file):
        # Load existing data from the pickle file
        with open(pkl_save_file, 'rb') as f:
            data = pickle.load(f)
        
        # Concatenate new data with existing data
        expert_observations = data['observations'] + observation_vectors
        expert_actions = data['actions'] + action_vectors
    else:
        # No existing file, use new data
        expert_observations = observation_vectors
        expert_actions = action_vectors

    # Save the updated expert data using pickle
    with open(pkl_save_file, 'wb') as f:
        pickle.dump({'observations': expert_observations, 'actions': expert_actions}, f)

    # Load the expert data again (to test)
    with open(pkl_save_file, 'rb') as f:
        data = pickle.load(f)
    
    test_expert_observations = data['observations']
    test_expert_actions = data['actions']

    print(f"Number of observations: {len(test_expert_observations)}")
    print(f"Number of actions: {len(test_expert_actions)}")

    

if __name__ == "__main__":
    main()