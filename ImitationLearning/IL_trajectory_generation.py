import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import time

from planning_sandbox.environment_class import Environment
from IL_env import ILEnv
from stable_baselines3.common.env_checker import check_env
from imitation.data.types import Trajectory


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
    
    iterations = 50
    num_goals: int = 6
    num_agents: int = 3
    size: int = 100
    num_skills: int = 2
    solve_type = 'optimal'
    
    
    logging.info("Setting up environment...")
    env = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True)
    gym_env = ILEnv(sandboxEnv=env)
    check_env(gym_env, warn=True)
    test_env(env)

    # seed_envs = clone_environments(env, iterations)


    
    # for env in seed_envs:
    #     print(f"Progress: {100*(seed_envs.index(env))/iterations:.2f}%", end='\r')
    #     env_copy: Environment = deepcopy(env)
    #     env_copy.solve_type = solve_type
    #     env_copy.find_numerical_solution()
    #     observation_vector = env_copy.get_observation_vector()
    #     action_vector = env_copy.get_action_vector()
    

if __name__ == "__main__":
    main()