import gymnasium as gym
import numpy as np
import copy
import logging

from planning_sandbox.environment_class import Environment
from planning_sandbox.scheduler_class import Scheduler
from planning_sandbox.goal_class import Goal


class RLEnv(gym.Env):
    def __init__(self, env):
        super(RLEnv, self).__init__()
        self.sandboxEnv: Environment = env
        self.initial_env = copy.deepcopy(env)
        self.obs_size = 0
        self.step_count = 0
        self.max_steps = self.sandboxEnv.size ** 2
        self.reward = 0
        self.total_reward = 0
        self.action_space = gym.spaces.MultiDiscrete([5]*(len(self.sandboxEnv.agents)))
        self._update_obs_values()
        logging.debug(f"Observation size: {self.obs_size}")

    def step(self, action):
        
        self.step_count += 1

        actions = np.array(action)
        reward = 0
        done = False
        amount_of_claimed_goals = 0
        

        for i, agent in enumerate(self.sandboxEnv.agents):

            action_is_valid = self.sandboxEnv.grid_map.validate_action(agent, actions[i])
            
            if action_is_valid:
                if agent_action == 0 or agent_action == 'stay':
                    self.episode_stay_actions += 1
                    if not agent.position in [goal.position for goal in self.sandboxEnv.goals]:
                        reward -= 10
                        done = True
                agent.apply_action(agent_action)
                distance_to_closest_goal = self.sandboxEnv.get_manhattan_distance_to_closest_unclaimed_goal(agent)
                reward += (3 - (distance_to_closest_goal/self.sandboxEnv.map_diagonal)*3)**2
            else:
                done = True
                reward -= 10
                self.episode_invalid_actions += 1
            amount_of_claimed_goals += self.sandboxEnv.update()
        
        reward += amount_of_claimed_goals*1000

        reward = reward / self.step_count

        done = done or (self.step_count >= self.max_steps) or self.sandboxEnv.scheduler.all_goals_claimed()
         

        self.episode_reward += reward
        self.episode_claimed_goals += amount_of_claimed_goals        

        if done:
            info = {"episode": {"r": self.episode_reward, "l": self.step_count, "claimed_goals": self.episode_claimed_goals, "invalid_actions": self.episode_invalid_actions, "stay_actions": self.episode_stay_actions, "goal_stay_actions": self.episode_goal_stay_actions}}
        else:
            info = {}

        return self._get_obs(), reward, done, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_invalid_actions = 0
        self.episode_stay_actions = 0
        self.episode_goal_stay_actions = 0
        self.step_count = 0
        self.sandboxEnv.reset()
        return self._get_obs(), {}
    
    def _update_obs_values(self):
        
        # self.normalized_grid = self.sandboxEnv.get_normalized_grid()
        # self.normalized_obstacle_positions = self.sandboxEnv.grid_map.get_normalized_positions(self.sandboxEnv.grid_map.obstacles)
        # self.normalized_goal_positions = self.sandboxEnv.grid_map.get_normalized_positions([goal.position for goal in self.sandboxEnv.goals])
        # self.normalized_goal_skill_vectors = self.sandboxEnv.get_normalized_skill_vectors_for_all_goals()
        # self.normalized_agent_positions = self.sandboxEnv.grid_map.get_normalized_positions([agent.position for agent in self.sandboxEnv.agents])

        self.manhattan_distances = self.sandboxEnv.get_manhattan_distances_for_all_agents_to_all_goals()
        # self.normalized_agent_skill_vectors = self.sandboxEnv.get_normalized_skill_vectors_for_all_agents()
        
        # self.normalized_step_count = self.step_count / self.max_steps
        # self.normalized_goals_unclaimed = len(self.sandboxEnv.scheduler.unclaimed_goals) / len(self.sandboxEnv.goals)

        self.obs_size = (
                        # len(self.normalized_grid.flatten()) +
                        # 2*len(self.normalized_obstacle_positions) +
                        2*len(self.sandboxEnv.goals) +
                        # len(self.normalized_goal_skill_vectors) +
                        2*len(self.sandboxEnv.agents) +
                        len(self.manhattan_distances) + 
                        # len(self.normalized_agent_skill_vectors) +
                        # 1 +
                        1
                        )
        
        self.observation_space = gym.spaces.Box(low=0, high=max(self.max_steps, self.sandboxEnv.size), shape=(self.obs_size,), dtype=np.float32)

    def _get_obs(self):
        self._update_obs_values()

        obs = np.zeros(self.obs_size, dtype=np.float32)
        offset = 0

        # obs[offset:offset+len(self.normalized_grid.flatten())] = self.normalized_grid.flatten()
        # offset += len(self.normalized_grid.flatten())

        # for pos in self.sandboxEnv.obstacles:
        #     obs[offset] = pos[0]
        #     obs[offset + 1] = pos[1]
        #     offset += 2
        
        for goal in self.sandboxEnv.goals:
            obs[offset] = goal.position[0]
            obs[offset + 1] = goal.position[1]
            offset += 2
        
        # obs[offset:offset+len(self.normalized_goal_skill_vectors)] = self.normalized_goal_skill_vectors
        # offset += len(self.normalized_goal_skill_vectors)
        
        for agent in self.sandboxEnv.agents:
            obs[offset] = agent.position[0]
            obs[offset + 1] = agent.position[1]
            offset += 2
        
        # obs[offset:offset+len(self.normalized_agent_skill_vectors)] = self.normalized_agent_skill_vectors
        # offset += len(self.normalized_agent_skill_vectors)

        for distance in self.manhattan_distances:
            obs[offset] = distance
            offset += 1
        

        obs[offset] = self.step_count
        offset += 1

        # obs[offset] = self.normalized_goals_unclaimed
        # offset += 1
        
        assert offset == self.obs_size, f"Observation size mismatch: {offset} vs {self.obs_size}"
        
        return obs