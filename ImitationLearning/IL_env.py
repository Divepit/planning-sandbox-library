import gymnasium as gym
import copy
import numpy as np

from gymnasium.spaces import Dict, MultiDiscrete, Discrete, Box

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

class ILEnv(gym.Env):
    def __init__(self, env):
        super(ILEnv, self).__init__()
        self.sandboxEnv: Environment = env
        self.obs_size = 0
        self.step_count = 0
        self.max_steps = self.sandboxEnv.size ** 2
        self.reward = 0
        self.total_reward = 0

        self.action_space = gym.spaces.MultiDiscrete([len(self.sandboxEnv.goals) + 1] * len(self.sandboxEnv.goals) * len(self.sandboxEnv.agents)) 
        
        self.observation_space = Dict(
            {
                "goal_positions": Box(low=-float('inf'), high=float('inf'), shape=(len(self.sandboxEnv.goals), 2)),
                "goal_required_skills": MultiDiscrete([2]*self.sandboxEnv.num_skills),
                "agent_positions": Box(low=-float('inf'), high=float('inf'), shape=(len(self.sandboxEnv.agents), 2)),
                "agent_skills": MultiDiscrete([2]*self.sandboxEnv.num_skills),
            }
        )
        

        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_invalid_actions = 0
        self.episode_stay_actions = 0
        self.episode_goal_stay_actions = 0

    def step(self, action):
        reward = 0
        done = False

        # Split the flattened action vector back into agent-wise goal lists
        agent_actions = [
            action[i * len(self.sandboxEnv.goals):(i + 1) * len(self.sandboxEnv.goals)] 
            for i in range(len(self.sandboxEnv.agents))
        ]
        
        for agent_index, actions in enumerate(agent_actions):
            valid_goals = [goal for goal in actions if goal != -1]  # Filter out `-1`
            self.sandboxEnv.cheapest_solution[self.sandboxEnv.agents[agent_index]] = [self.sandboxEnv.goals[goal] for goal in valid_goals]

        total_steps_moved, total_steps_waited, total_cost, solve_time, claimed_goals = self.sandboxEnv.solve_cheapest_solution(max_iterations=self.max_steps)

        reward = -total_cost-solve_time

        self.step_count += total_steps_moved + total_steps_waited
        self.episode_reward += reward
        self.episode_claimed_goals += claimed_goals
        self.episode_invalid_actions += 0
        self.episode_stay_actions += total_steps_waited
        self.episode_goal_stay_actions += 0

        if done:
            info = {"episode": {"r": self.episode_reward, "l": self.step_count, "claimed_goals": self.episode_claimed_goals, "invalid_actions": self.episode_invalid_actions, "stay_actions": self.episode_stay_actions, "goal_stay_actions": self.episode_goal_stay_actions}}
        else:
            info = {}

        return self.sandboxEnv.get_observation_vector(), reward, done, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_invalid_actions = 0
        self.episode_stay_actions = 0
        self.episode_goal_stay_actions = 0
        self.step_count = 0
        self.sandboxEnv.reset()
        return self.sandboxEnv.get_observation_vector(), {}
    
    def render(self):
        vis = Visualizer(env=self.sandboxEnv, speed=200)
        vis.visualise_cheapest_solution()


        