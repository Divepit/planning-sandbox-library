import gymnasium as gym
import numpy as np
import copy
import logging

from planning_sandbox.environment_class import Environment


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
        self.action_space = gym.spaces.MultiDiscrete([len(self.sandboxEnv.goals)-1]*len(self.sandboxEnv.agents))
        # [goal_to_claim_for_agent_1, goal_to_claim_for_agent_2, ..., goal_to_claim_for_agent_n]
        # 1 - move up, 2 - move down, 3 - move left, 4 - move right, 0 - stay

        self._update_obs_values()

        
        logging.debug(f"Observation size: {self.obs_size}")

    def step(self, action):
        actions = np.array(action)
        reward = 0
        self.step_count += 1
        done = False
        
        amount_of_claimed_goals = 0

        for i, agent in enumerate(self.sandboxEnv.agents):
            goal = self.sandboxEnv.goals[actions[i]]
            path = self.sandboxEnv.grid_map.generate_shortest_path_for_agent(agent, goal)
            self.sandboxEnv.grid_map.assign_path_to_agent(agent, path)
            agent_action, action_cost = self.sandboxEnv.grid_map.get_move_and_cost_to_reach_next_position(agent)
            action_is_valid = self.sandboxEnv.grid_map.validate_action(agent, agent_action)
            if action_is_valid:
                if agent_action == 0 or agent_action == 'stay':
                    self.episode_stay_actions += 1
                    if self.sandboxEnv.scheduler.is_goal_position(agent.position):
                        goal_at_position = self.sandboxEnv.scheduler.get_goal_at_position(agent.position)
                        goal_already_claimed = goal_at_position.claimed
                        if not goal_already_claimed:
                            skill_match = self.sandboxEnv.scheduler.agent_has_one_or_more_required_skills_for_goal(agent, goal_at_position)
                            if skill_match:
                                self.episode_goal_stay_actions += 1
                                reward += 2
                            else:
                                reward -= 0.5
                        else:
                            reward -= 0.5
                    else:
                        reward -= 0.5
                agent.apply_action(agent_action)
            else:
                self.episode_invalid_actions += 1
                reward -= 100
                done = True
            amount_of_claimed_goals += self.sandboxEnv.update()
        
        reward += amount_of_claimed_goals*30
        reward -= 5*(self.step_count/self.max_steps)
                
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

        self.normalized_obstacle_positions = self.sandboxEnv.grid_map.get_normalized_positions(self.sandboxEnv.grid_map.obstacles)
        self.normalized_goal_positions = self.sandboxEnv.grid_map.get_normalized_positions([goal.position for goal in self.sandboxEnv.goals])
        self.normalized_goal_skill_vectors = self.sandboxEnv.get_normalized_skill_vectors_for_all_goals()
        self.normalized_agent_positions = self.sandboxEnv.grid_map.get_normalized_positions([agent.position for agent in self.sandboxEnv.agents])
        self.normalized_agent_skill_vectors = self.sandboxEnv.get_normalized_skill_vectors_for_all_agents()
        self.normalized_claimed_goals_vector = self.sandboxEnv.scheduler.get_normalized_claimed_goals()
        self.normalized_step_count = self.step_count / self.max_steps

        self.obs_size = (2*len(self.normalized_obstacle_positions) +
                        2*len(self.normalized_goal_positions) +
                        len(self.normalized_goal_skill_vectors) +
                        2*len(self.normalized_agent_positions) +
                        len(self.normalized_agent_skill_vectors) +
                        len(self.normalized_claimed_goals_vector) +
                        1)
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,), dtype=np.float32)

    def _get_obs(self):
        self._update_obs_values()

        obs = np.zeros(self.obs_size, dtype=np.float32)
        offset = 0

        for pos in self.normalized_obstacle_positions:
            obs[offset] = pos[0]
            obs[offset + 1] = pos[1]
            offset += 2
        
        for pos in self.normalized_goal_positions:
            obs[offset] = pos[0]
            obs[offset + 1] = pos[1]
            offset += 2
        
        obs[offset:offset+len(self.normalized_goal_skill_vectors)] = self.normalized_goal_skill_vectors
        offset += len(self.normalized_goal_skill_vectors)
        
        for pos in self.normalized_agent_positions:
            obs[offset] = pos[0]
            obs[offset + 1] = pos[1]
            offset += 2
        
        obs[offset:offset+len(self.normalized_agent_skill_vectors)] = self.normalized_agent_skill_vectors
        offset += len(self.normalized_agent_skill_vectors)
        
        obs[offset:offset+len(self.normalized_claimed_goals_vector)] = self.normalized_claimed_goals_vector
        offset += len(self.normalized_claimed_goals_vector)

        obs[offset] = self.normalized_step_count
        offset += 1
        
        assert offset == self.obs_size, f"Observation size mismatch: {offset} vs {self.obs_size}"
        
        return obs