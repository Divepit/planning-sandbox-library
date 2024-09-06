from planning_sandbox.environment_class import Environment
from RL_mover_env import RLEnv
from stable_baselines3 import PPO
from planning_sandbox.visualiser_class import Visualizer
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the trained model
try:
    model = PPO.load("ppo_custom_env_improved_goal_assignment")
    print("Loaded trained model")
except FileNotFoundError:
    print("Trained model not found. Please make sure the model file exists.")
    exit(1)



num_agents = 3
num_goals = 3
num_obstacles = 0
width = 8
height = 8
num_skills = 2
max_steps = width * height

sandboxEnv = Environment(width=width, height=height, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, num_skills=num_skills)

visualizer = Visualizer(width=sandboxEnv.width, height=sandboxEnv.height, obstacles=sandboxEnv.obstacles, agents=sandboxEnv.agents, goals=sandboxEnv.goals)
RLenv = RLEnv(sandboxEnv)
RLenv = DummyVecEnv([lambda: RLenv])


# Handle different return types from env.reset()
reset_return = RLenv.reset()
if isinstance(reset_return, tuple):
    obs = reset_return[0]
else:
    obs = reset_return

done = False
step = 0

while not done and step < max_steps:

    model_action, _ = model.predict(obs, deterministic=False)
    
    # Handle different return types from env.step()
    step_return = RLenv.step(model_action)
    if len(step_return) == 5:
        obs, reward, terminated, truncated, info = step_return
        done = terminated or truncated
    else:
        obs, reward, done, info = step_return
    
    step += 1

    action_vector = model_action[0]
    actions = model_action[0]

    for i, agent in enumerate(sandboxEnv.agents):
            agent_action = sandboxEnv.controller.action_map[actions[i]]
            if agent_action in sandboxEnv.controller.get_valid_actions(agent):
                agent.apply_action(agent_action)
                print(f"Agent {i} moved {agent_action}")
            sandboxEnv.scheduler.update_goal_statuses()
            claimed_goals = sandboxEnv.scheduler._get_normalized_claimed_goals()
            print(f"Claimed goals: {claimed_goals}")
        
        
    visualizer.run_step()
    done = step >= max_steps or sandboxEnv.scheduler.all_goals_claimed()
    print("Reward: ", reward)

