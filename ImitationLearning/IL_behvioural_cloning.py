import os
import logging
import numpy as np
import pickle

from planning_sandbox.environment_class import Environment
from IL_env import ILEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc

from copy import deepcopy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_environments(env, num_clones):
    seed_envs = []
    print("Clone environments...")
    for _ in range(num_clones):
        print(f"Progress: {len(seed_envs)+1}/{num_clones}", end='\r')
        seed_envs.append(deepcopy(env))
    return seed_envs

def test_env(env):
    t_env: Environment = deepcopy(env)
    t_env.find_numerical_solution(solve_type='optimal')
    action_vector = t_env.get_action_vector()
    solution_from_action_vector = t_env.get_full_solution_from_action_vector(action_vector)
    assert solution_from_action_vector == t_env.full_solution, "Solution from action vector does not match full solution"
    logging.info("Test passed")



def main():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    num_goals: int = 5
    num_agents: int = 3
    size: int = 100
    num_skills: int = 2
    solve_type = 'optimal'
    
    
    logging.info("Setting up and testing environment...")
    env_for_test = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True, solve_type=solve_type)
    env_for_check = ILEnv(sandboxEnv=env_for_test)
    test_env(env_for_test)
    check_env(env_for_check, warn=True)

    # Save the trajectories to a file
    with open(dir_path+'/trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    sandboxEnv = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True, solve_type=solve_type)
    training_env = ILEnv(sandboxEnv=sandboxEnv)
    

    bc_trainer = bc.BC(
        observation_space=training_env.observation_space,
        action_space=training_env.action_space,
        demonstrations=trajectories,
        rng=np.random.default_rng(),
        batch_size=1,
    )

    bc_trainer.train(n_epochs=1077)

    reward, _ = evaluate_policy(model=bc_trainer.policy, env=training_env, n_eval_episodes=10, deterministic=True, render=True, return_episode_rewards=True)
    logging.info(f"Mean reward: {reward}")

    

if __name__ == "__main__":
    main()