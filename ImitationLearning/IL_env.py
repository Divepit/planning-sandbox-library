import gymnasium as gym
import copy
import numpy as np
import logging

from gymnasium.spaces import Dict, MultiDiscrete, Discrete, Box

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ILEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, sandboxEnv: Environment, render_mode="human"):
        super(ILEnv, self).__init__()
        self.render_mode = render_mode
        self.sandboxEnv: Environment = sandboxEnv
        # self.vis = Visualizer(env=self.sandboxEnv, speed=200)
        self.step_count = 0
        self.max_steps = self.sandboxEnv.size ** 2
        self.max_episode_attempts = 2000
        self.reward = 0
        self.total_reward = 0
        
        # self.action_space = gym.spaces.MultiDiscrete(
        #     nvec=[[len(self.sandboxEnv.goals)+1]*len(self.sandboxEnv.goals)]*len(self.sandboxEnv.agents),
        #     start=[[0]*len(self.sandboxEnv.goals)]*len(self.sandboxEnv.agents),
        #     dtype=np.int64
        #     )
        self.action_space = gym.spaces.MultiDiscrete(
            nvec=[len(self.sandboxEnv.goals)+1]*len(self.sandboxEnv.agents)*len(self.sandboxEnv.goals),
            start=[0]*len(self.sandboxEnv.agents)*len(self.sandboxEnv.goals),
            dtype=np.int64
)
        
        self.observation_space = Dict(
            {   
                "map_elevations": Box(low=-1500, high=1500, shape=(self.sandboxEnv.size * self.sandboxEnv.size,), dtype=np.float32),
                "goal_positions": MultiDiscrete(
                nvec=[self.sandboxEnv.size]*len(self.sandboxEnv.goals)*2,
                start=[0]*2*len(self.sandboxEnv.goals),
                dtype=np.int64
                ),
                "goal_required_skills": MultiDiscrete(
                nvec=[2]*self.sandboxEnv.num_skills*len(self.sandboxEnv.goals),
                start=[0]*self.sandboxEnv.num_skills*len(self.sandboxEnv.goals),
                dtype=np.int64
                ),
                "agent_positions": MultiDiscrete(
                nvec=[self.sandboxEnv.size]*len(self.sandboxEnv.agents)*2,
                start=[0]*2*len(self.sandboxEnv.agents),
                dtype=np.int64
                ),
                "agent_skills": MultiDiscrete(
                nvec=[2]*self.sandboxEnv.num_skills*len(self.sandboxEnv.agents),
                start=[0]*self.sandboxEnv.num_skills*len(self.sandboxEnv.agents),
                dtype=np.int64
                ),
                "claimed_goals": Discrete(len(self.sandboxEnv.goals)+1, start=0),
            }
        )
        

        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_attempts = 0
        self.episode_distributed_goals = 0
        self.episode_cost = 0

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
            valid_goals = [goal for goal in actions if goal != -1]  # Filter out `-1`
            self.sandboxEnv.full_solution[self.sandboxEnv.agents[agent_index]] = [self.sandboxEnv.goals[goal] for goal in valid_goals]
            distributed_goals += len(valid_goals)
        


        total_steps_moved, total_steps_waited, total_cost, solve_time, claimed_goals = self.sandboxEnv.solve_full_solution(fast=True)

        unclaimed_goals = len(self.sandboxEnv.scheduler.unclaimed_goals)

        reward -= total_cost
        reward -= unclaimed_goals * 500
        reward -= distributed_goals * 200

        if self.step_count >= self.max_steps and not self.sandboxEnv.scheduler.all_goals_claimed():
            reward -= 1000

        self.step_count += total_steps_moved + total_steps_waited
        self.episode_reward += reward
        self.episode_claimed_goals += claimed_goals
        self.episode_distributed_goals += distributed_goals
        self.episode_cost += total_cost

        logging.debug("Reward: {}".format(reward))
        logging.debug("Total Steps Moved: {}".format(total_steps_moved))
        logging.debug("Total Steps Waited: {}".format(total_steps_waited))
        logging.debug("Total Cost: {}".format(total_cost))
        logging.debug("Solve Time: {}".format(solve_time))
        logging.debug("Claimed Goals: {}".format(claimed_goals))
        logging.debug("Unclaimed Goals: {}".format(len(self.sandboxEnv.scheduler.unclaimed_goals)))


        done = self.sandboxEnv.scheduler.all_goals_claimed() or self.episode_attempts >= self.max_episode_attempts

        if done:
            info = {"episode": {"r": self.episode_reward/self.episode_attempts, "l": self.step_count/self.episode_attempts, "distributed_goals": self.episode_distributed_goals/self.episode_attempts, "cost": self.episode_cost/self.episode_attempts, "claimed_goals": self.episode_claimed_goals/self.episode_attempts}}
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
        self.sandboxEnv.reset()
        return self.sandboxEnv.get_observation_vector(), {}
    
    def render(self):
        vis = Visualizer(env=self.sandboxEnv, speed=200)
        vis.visualise_full_solution(max_iterations=self.max_steps)
        del vis
        