import numpy as np

class Planner:
    def __init__(self, agents, grid_map):
        self.agents = agents
        
        self.grid_map = grid_map

        self.paths = {}

    def _get_agent_current_goal(self, agent):
        if agent in self.paths:
            return self.paths[agent][-1]
        return None
    
    def _get_current_index_on_path(self, agent):
        return self.paths[agent].index(agent.position)
    
    def _get_move_to_reach_position(self, agent, next_position):
        current_position = agent.position
        next_position = next_position
        if current_position[0] == next_position[0] and current_position[1] == next_position[1] - 1:
            return 'down'
        elif current_position[0] == next_position[0] and current_position[1] == next_position[1] + 1:
            return 'up'
        elif current_position[0] == next_position[0] - 1 and current_position[1] == next_position[1]:
            return 'right'
        elif current_position[0] == next_position[0] + 1 and current_position[1] == next_position[1]:
            return 'left'
        elif current_position[0] == next_position[0] and current_position[1] == next_position[1]:
            return 'stay'
        else:
            return 'moveto'

    def reset(self):
        self.paths = {}

    def generate_shortest_path_for_agent(self, agent, goal):
        current_agent_goal = self._get_agent_current_goal(agent)
        if current_agent_goal is not None and current_agent_goal == goal.position:
            path_index = self._get_current_index_on_path(agent)
            return self.paths[agent][path_index:]
        start = agent.position
        goal = goal.position
        path = self.grid_map.shortest_path(start, goal)
        return path
    
    def assign_path_to_agent(self, agent, path):
        self.paths[agent] = path
    
    def get_next_index_on_path(self, agent):
        if self._get_current_index_on_path(agent) == len(self.paths[agent]) - 1:
            return self._get_current_index_on_path(agent)
        return self._get_current_index_on_path(agent) + 1
    
    def get_next_position_on_path(self, agent):
        if agent in self.paths:
            return self.paths[agent][self.get_next_index_on_path(agent)]
        return agent.position

    def get_move_to_reach_next_position(self, agent):
        next_position = self.get_next_position_on_path(agent)
        return self._get_move_to_reach_position(agent, next_position)
    
    
    def assign_shortest_path_for_goal_to_agent(self, agent, goal):
        path = self.generate_shortest_path_for_agent(agent, goal)
        self.assign_path_to_agent(agent, path)