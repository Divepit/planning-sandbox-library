import numpy as np

class Planner:
    def __init__(self, agents, grid_map):
        self.agents = agents
        
        self.grid_map = grid_map

        self.paths = {}
    def reset(self):
        self.paths = {}

    def generate_shortest_path_for_agent(self, agent, goal):
        current_agent_goal = self.get_agent_current_goal(agent)
        if current_agent_goal is not None and current_agent_goal == goal.position:
            return self.paths[agent]
        start = agent.position
        goal = goal.position
        path = self.grid_map.shortest_path(start, goal)
        return path
    
    def assign_path_to_agent(self, agent, path):
        self.paths[agent] = path

    def get_plan_for_agent(self, agent):
        return self.paths[agent]
    
    def get_current_index_on_path(self, agent):
        return self.paths[agent].index(agent.position)
    
    def get_next_index_on_path(self, agent):
        if self.get_current_index_on_path(agent) == len(self.paths[agent]) - 1:
            return self.get_current_index_on_path(agent)
        return self.get_current_index_on_path(agent) + 1
    
    def get_next_position_on_path(self, agent):
        return self.paths[agent][self.get_next_index_on_path(agent)]
    
    def get_move_to_reach_position(self, agent, next_position):
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

    def get_move_to_reach_next_position(self, agent):
        next_position = self.get_next_position_on_path(agent)
        return self.get_move_to_reach_position(agent, next_position)
    
    def get_next_moves_for_all_agents(self):
        next_moves = {}
        for agent in self.agents:
            next_moves[agent] = self.get_move_to_reach_next_position(agent)
        return next_moves
    
    def get_agent_current_goal(self, agent):
        if agent in self.paths:
            return self.paths[agent][-1]
        return None
    
    def assign_shortest_path_for_goal_to_agent(self, agent, goal):
        path = self.generate_shortest_path_for_agent(agent, goal)
        self.assign_path_to_agent(agent, path)