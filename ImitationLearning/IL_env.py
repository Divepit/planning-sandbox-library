import gymnasium as gym
import copy
import numpy as np
import logging

from gymnasium.spaces import Dict, MultiDiscrete, Discrete, Box, MultiBinary

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ILEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, sandboxEnv: Environment, render_mode="human"):
        super(ILEnv, self).__init__()
        self.render_mode = render_mode
        self.sandboxEnv: Environment = sandboxEnv
        self.max_episode_attempts = 150
        

        self.step_count = 0
        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_attempts = 0
        self.episode_distributed_goals = 0
        self.episode_cost = 0
        self.episode_deadlocks = 0
        self.episode_unclaimed_goals = 0


        self.action_space = gym.spaces.MultiDiscrete(
            nvec=[len(self.sandboxEnv.goals)+1]*len(self.sandboxEnv.agents)*len(self.sandboxEnv.goals),
            start=[0]*len(self.sandboxEnv.agents)*len(self.sandboxEnv.goals),
            dtype=np.int64
        )

        
        self.observation_space = Dict(
            {   
                "claimed_goals": Box(low=0, high=1, shape=(len(self.sandboxEnv.goals),), dtype=np.int16),
                "map_elevations": Box(low=-1, high=1, shape=(self.sandboxEnv.size * self.sandboxEnv.size,), dtype=np.float32),
                "goal_positions": Box(low=0, high=1, shape=(self.sandboxEnv.size * self.sandboxEnv.size,), dtype=np.int16),
                "agent_positions": Box(low=0, high=1, shape=(self.sandboxEnv.size * self.sandboxEnv.size,), dtype=np.int16),
                "goal_required_skills": MultiDiscrete(
                nvec=[2]*self.sandboxEnv.num_skills*len(self.sandboxEnv.goals),
                start=[0]*self.sandboxEnv.num_skills*len(self.sandboxEnv.goals),
                dtype=np.int16
                ),
                "agent_skills": MultiDiscrete(
                nvec=[2]*self.sandboxEnv.num_skills*len(self.sandboxEnv.agents),
                start=[0]*self.sandboxEnv.num_skills*len(self.sandboxEnv.agents),
                dtype=np.int16
                ),
            }
        )

    def step(self, action):
        corrected_action = action-1
        self.episode_attempts += 1
        self.sandboxEnv.soft_reset()
        reward = 0
        distributed_goals = 0

        logging.debug("New Step")
        logging.debug("Action: {}".format(action))
        logging.debug("Corrected Action: {}".format(corrected_action))

        agent_actions = [
            corrected_action[i * len(self.sandboxEnv.goals):(i + 1) * len(self.sandboxEnv.goals)] 
            for i in range(len(self.sandboxEnv.agents))
        ]
        
        for agent_index, actions in enumerate(agent_actions):
            agent = self.sandboxEnv.agents[agent_index]
            valid_goals = [goal for goal in actions if goal != -1]  # Filter out `-1`
            self.sandboxEnv.full_solution[agent] = [self.sandboxEnv.goals[goal] for goal in valid_goals]
            distributed_goals += len(valid_goals)
        
        total_cost = self.sandboxEnv._calculate_cost_of_closed_solution(self.sandboxEnv.full_solution, max_cost=np.inf)
        self.sandboxEnv.solve_full_solution(fast=True)
        all_goals_claimed = self.sandboxEnv.scheduler.all_goals_claimed()





        reward -= total_cost/len(self.sandboxEnv.agents)

        if not all_goals_claimed or self.sandboxEnv.deadlocked:
            reward -= 100*len(self.sandboxEnv.scheduler.unclaimed_goals)

        self.episode_reward += reward
        self.episode_distributed_goals += distributed_goals
        self.episode_cost += total_cost
        self.episode_deadlocks += int(self.sandboxEnv.deadlocked)
        self.episode_unclaimed_goals += len(self.sandboxEnv.scheduler.unclaimed_goals)

        logging.debug("Reward: {}".format(reward))
        logging.debug("Total Cost: {}".format(total_cost))
        logging.debug("Unclaimed Goals: {}".format(len(self.sandboxEnv.scheduler.unclaimed_goals)))


        done = self.episode_attempts >= self.max_episode_attempts

        if done:
            info = {"episode": {"r": self.episode_reward/self.episode_attempts, "l": self.episode_attempts,"distributed_goals": self.episode_distributed_goals/self.episode_attempts, "cost": self.episode_cost/self.episode_attempts, "unclaimed_goals": self.episode_unclaimed_goals/self.episode_attempts, "episode_attempts": self.episode_attempts, "deadlocks": self.episode_deadlocks/self.episode_attempts}}
        else:
            info = {}

        logging.debug("Observation: {}".format(self.sandboxEnv.get_observation_vector()))
        return self.sandboxEnv.get_observation_vector(), float(reward), done, False, info


    def reset(self, seed=None, options=None):
        logging.debug("Resetting environment")
        super().reset(seed=seed)
        self.episode_attempts = 0
        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_distributed_goals = 0
        self.episode_cost = 0
        self.step_count = 0
        self.episode_deadlocks = 0
        self.sandboxEnv.reset()
        self.episode_unclaimed_goals = 0
        return self.sandboxEnv.get_observation_vector(), {}
    
    def render(self):
        vis = Visualizer(env=self.sandboxEnv, speed=200)
        vis.visualise_full_solution()
        del vis
        