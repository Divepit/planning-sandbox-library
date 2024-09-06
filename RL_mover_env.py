import gymnasium as gym
import numpy as np
import copy

from planning_sandbox.environment_class import Environment
from planning_sandbox.controller_class import Controller
from planning_sandbox.scheduler_class import Scheduler
from planning_sandbox.goal_class import Goal


class RLEnv(gym.Env):
    def __init__(self, env):
        super(RLEnv, self).__init__()
        self.sandboxEnv: Environment = env
        self.initial_env = copy.deepcopy(env)
        
        self.obs_size = 0

        self.step_count = 0
        self.max_steps = self.sandboxEnv.width * self.sandboxEnv.height
        self.reward = 0
        self.total_reward = 0
        self.action_space = gym.spaces.MultiDiscrete([5]*(self.sandboxEnv.num_agents))
        # [action_of_agent_1, action_of_agent_2, ..., action_of_agent_n]
        # 1 - move up, 2 - move down, 3 - move left, 4 - move right, 0 - stay

        self._update_obs_values()

        
        print(f"Observation size: {self.obs_size}")

    def step(self, action):
        actions = np.array(action)
        reward = 0
        self.step_count += 1
        
        amount_previously_unclaimed_goals = len(self.sandboxEnv.scheduler._get_unclaimed_goals())

        for i, agent in enumerate(self.sandboxEnv.agents):
            agent_action = self.sandboxEnv.controller.action_map[actions[i]]
            if agent_action in self.sandboxEnv.controller.get_valid_actions(agent):
                if agent_action == 0:
                    reward -= 0.25
                agent.apply_action(agent_action)
            else:
                reward -= 5
            self.sandboxEnv.scheduler.update_goal_statuses()
        
        amount_newly_claimed_goals = len(self.sandboxEnv.scheduler._get_unclaimed_goals()) - amount_previously_unclaimed_goals
        reward += amount_newly_claimed_goals
                
        done = (self.step_count >= self.max_steps) or self.sandboxEnv.scheduler.all_goals_claimed()

        if done:
            for goal in self.sandboxEnv.goals:
                if not goal.claimed:
                    reward -= 5
        
        self.episode_reward += reward

        if done:
            info = {"episode": {"r": self.episode_reward, "l": self.step_count}}
        else:
            info = {}

        return self._get_obs(), reward, done, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_reward = 0
        self.step_count = 0
        self.sandboxEnv._reset_environment()
        return self._get_obs(), {}
    
    def _update_obs_values(self):

        self.normalized_obstacle_positions = self.sandboxEnv.grid_map.get_normalized_positions(self.sandboxEnv.grid_map.obstacles)
        self.normalized_goal_positions = self.sandboxEnv.grid_map.get_normalized_positions([goal.position for goal in self.sandboxEnv.goals])
        self.normalized_goal_skill_vectors = self.sandboxEnv.get_normalized_skill_vectors_for_all_goals()
        self.normalized_agent_positions = self.sandboxEnv.grid_map.get_normalized_positions([agent.position for agent in self.sandboxEnv.agents])
        self.normalized_agent_skill_vectors = self.sandboxEnv.get_normalized_skill_vectors_for_all_agents()
        self.normalized_claimed_goals_vector = self.sandboxEnv.scheduler._get_normalized_claimed_goals()

        self.obs_size = (2*len(self.normalized_obstacle_positions) +
                        2*len(self.normalized_goal_positions) +
                        self.sandboxEnv.num_skills*len(self.normalized_goal_skill_vectors) +
                        2*len(self.normalized_agent_positions) +
                        self.sandboxEnv.num_skills*len(self.normalized_agent_skill_vectors) +
                        len(self.normalized_claimed_goals_vector))
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,), dtype=np.float32)

    def _get_obs(self):
        
        self._update_obs_values()

        obs = np.zeros(self.obs_size, dtype=np.float32)
        offset = 0

        for i, pos in enumerate(self.normalized_obstacle_positions):
            obs[i] = pos[0]
            obs[i + 1] = pos[1]
            offset += 2
        
        for i, pos in enumerate(self.normalized_goal_positions):
            obs[i + offset] = pos[0]
            obs[i + 1 + offset] = pos[1]
            offset += 2
        
        for i, skill_vector in enumerate(self.normalized_goal_skill_vectors):
            for j, skill in enumerate(skill_vector):
                obs[j + offset] = skill
            offset += len(skill_vector)
        
        for i, pos in enumerate(self.normalized_agent_positions):
            obs[i + offset] = pos[0]
            obs[i + 1 + offset] = pos[1]
            offset += 2
        
        for i, skill_vector in enumerate(self.normalized_agent_skill_vectors):
            for j, skill in enumerate(skill_vector):
                obs[j + offset] = skill
            offset += len(skill_vector)
        
        for i, claimed in enumerate(self.normalized_claimed_goals_vector):
            obs[i + offset] = claimed
            offset += 1
        


        return obs