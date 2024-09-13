import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning_sandbox.environment_class import Environment
from RL_mover_env import RLEnv
from stable_baselines3 import PPO
from planning_sandbox.visualiser_class import Visualizer
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the trained model
try:
    # model = PPO.load("ppo_custom_env_optimized_gpu")
    model = PPO.load("ppo_custom_env_improved_goal_assignment")
    # model = PPO.load("/Users/marco/Programming/PlanningEnvironmentLibrary/RL/saved_models/easy_test_2g")
    print("Loaded trained model")
except FileNotFoundError:
    print("Trained model not found. Please make sure the model file exists.")
    exit(1)

def run_sim():
    num_agents = 3
    num_goals = 2
    num_obstacles = 2
    size = 10
    num_skills = 2


    sandboxEnv = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, num_skills=num_skills)
    RLenv = RLEnv(sandboxEnv)

    visualizer = Visualizer(sandboxEnv, cell_size=30)
    RLenv = DummyVecEnv([lambda: RLenv])


    # Handle different return types from env.reset()
    reset_return = RLenv.reset()
    if isinstance(reset_return, tuple):
        obs = reset_return[0]
    else:
        obs = reset_return

    done = False
    step = 0

    while not done:

        model_action, _ = model.predict(obs, deterministic=False)
        
        # Handle different return types from env.step()
        step_return = RLenv.step(model_action)
        if len(step_return) == 5:
            obs, reward, terminated, truncated, info = step_return
            done = terminated or truncated
        else:
            obs, reward, done, info = step_return
        
        step += 1
        visualizer.run_step()


while(True):
    run_sim()

