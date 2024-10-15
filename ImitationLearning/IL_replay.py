import sys
import os
import glob
import logging
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C

from IL_env import ILEnv
from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

learning_sandbox_env: Environment = Environment(size=100, num_agents=3, num_goals=5, num_skills=2, use_geo_data=True, replan_on_goal_claim=False)

learningEnv = ILEnv(sandboxEnv=learning_sandbox_env)
check_env(learningEnv, warn=True)

while True:

    save_path = "/Users/marco/Programming/PlanningEnvironmentLibrary/ImitationLearning/model_logs/"
    files = glob.glob(os.path.join(save_path, "*.zip"))
    latest_file = max(files, key=os.path.getmtime)

    print(f"Loading model from {latest_file}")

    model = A2C.load(latest_file)
    obs, _ = learningEnv.reset()
    optimalEnv = copy.deepcopy(learning_sandbox_env)
    fastEnv = copy.deepcopy(learning_sandbox_env)

    action, _ = model.predict(obs, deterministic=True)

    predicted_full_solution = learning_sandbox_env.get_full_solution_from_action_vector(action)
    predicted_full_solution_cost = learning_sandbox_env.calculate_cost_of_closed_solution(predicted_full_solution)
    learning_sandbox_env.full_solution = predicted_full_solution
    learning_sandbox_env.solve_full_solution()
    if not learning_sandbox_env.deadlocked:
        learningEnv.render()
    else:
        print("Deadlocked - retrying...")
        continue


    optimal_solution = optimalEnv.find_numerical_solution(solve_type='optimal')
    optimal_solution_cost = optimalEnv.calculate_cost_of_closed_solution(optimal_solution)
    vis = Visualizer(optimalEnv)
    vis.visualise_full_solution()

    fast_solution = fastEnv.find_numerical_solution(solve_type='fast')
    fast_solution_cost = fastEnv.solve_full_solution()[2]

    print(f"Optimal solution cost: {optimal_solution_cost}")
    print(f"Fast solution cost: {fast_solution_cost}")
    print(f"Predicted solution cost: {predicted_full_solution_cost}")

