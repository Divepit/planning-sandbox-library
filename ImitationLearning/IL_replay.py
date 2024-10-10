import sys
import os
import glob
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C

from IL_env import ILEnv
from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

learning_sandbox_env: Environment = Environment(size=100, num_agents=3, num_goals=5, num_skills=2, use_geo_data=True)
visualizer = Visualizer(learning_sandbox_env, speed=200)

learningEnv = ILEnv(sandboxEnv=learning_sandbox_env)
check_env(learningEnv, warn=True)

save_path = "/Users/marco/Programming/PlanningEnvironmentLibrary/ImitationLearning/model_logs/"
files = glob.glob(os.path.join(save_path, "*.zip"))
latest_file = max(files, key=os.path.getmtime)

model = A2C.load(latest_file)

obs, _ = learningEnv.reset()
action, _ = model.predict(obs, deterministic=True)
# action = learningEnv.action_space.sample()

predicted_full_solution = learning_sandbox_env.get_full_solution_from_action_vector(action)
learning_sandbox_env.full_solution = predicted_full_solution
learning_sandbox_env.solve_full_solution()
visualizer.visualise_full_solution()