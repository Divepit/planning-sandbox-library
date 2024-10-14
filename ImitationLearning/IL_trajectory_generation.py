import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import time
import gymnasium as gym

from planning_sandbox.environment_class import Environment
from IL_env import ILEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Trajectory
from imitation.algorithms import sqil


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
    
    num_goals: int = 6
    num_agents: int = 3
    size: int = 100
    num_skills: int = 2
    solve_type = 'optimal'
    
    amount_of_trajectories = 50
    
    
    logging.info("Setting up and testing environment...")
    env_for_test = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True, solve_type=solve_type)
    env_for_check = ILEnv(sandboxEnv=env_for_test)
    test_env(env_for_test)
    check_env(env_for_check, warn=True)

    logging.info("Generating trajectories...")
    trajectories = []
    sandboxEnv = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True, solve_type=solve_type)
    for _ in range(amount_of_trajectories):
        print(f"Progress: {len(trajectories)+1}/{amount_of_trajectories}", end='\r')
        sandboxEnv.find_numerical_solution(solve_type=solve_type)
        new_trajectory = Trajectory(
            obs=sandboxEnv.get_observation(),
            acts=sandboxEnv.get_action_vector(),
            infos={},
            terminal=True
        )
        trajectories.append(new_trajectory)
        sandboxEnv.reset()


    sqil_trainer = sqil.SQIL(
        venv=sandboxEnv,
        demonstrations=trajectories,
        policy="MlpPolicy",
    )

    reward_before_training, _ = evaluate_policy(sqil_trainer.policy, sandboxEnv, 10)
    logging.info(f"Reward before training: {reward_before_training}")

    sqil_trainer.train(
    total_timesteps=1_000,
    )  # Note: set to 1_000_000 to obtain good results
    reward_after_training, _ = evaluate_policy(sqil_trainer.policy, sandboxEnv, 10)
    logging.info(f"Reward after training: {reward_after_training}")
    

if __name__ == "__main__":
    main()