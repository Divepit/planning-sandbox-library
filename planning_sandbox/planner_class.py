from typing import List

from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal
from planning_sandbox.grid_map_class import GridMap

import networkx as nx

class Planner:
    def __init__(self, agents, grid_map):
        self.agents: List[Agent] = agents
        
        self.grid_map: GridMap = grid_map

        self.paths = {}
    
    def _get_current_index_on_path(self, agent: Agent):
        return self.paths[agent].index(agent.position)
    
    def _get_move_to_reach_position(self, agent: Agent, next_position):
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

    def generate_shortest_path_for_agent(self, agent: Agent, goal: Goal):
        if agent in self.paths:
            current_agent_goal = self.paths[agent][-1]
        else:
            current_agent_goal = None

        if current_agent_goal is not None and current_agent_goal == goal.position:
            path_index = self._get_current_index_on_path(agent)
            return self.paths[agent][path_index:]
        
        start = agent.position
        goal = goal.position
        path = self.grid_map.shortest_path(start, goal)

        return path
    
    def assign_path_to_agent(self, agent: Agent, path):
        self.paths[agent] = path
    
    def get_next_index_on_path(self, agent: Agent):
        if self._get_current_index_on_path(agent) == len(self.paths[agent]) - 1:
            return self._get_current_index_on_path(agent)
        return self._get_current_index_on_path(agent) + 1
    
    def get_next_position_on_path(self, agent: Agent):
        if agent in self.paths:
            return self.paths[agent][self.get_next_index_on_path(agent)]
        return agent.position

    def get_move_and_cost_to_reach_next_position(self, agent: Agent):
        current_position = agent.position
        next_position = self.get_next_position_on_path(agent)
        move_cost = self.grid_map.get_cost_for_move(current_position, next_position)
        return self._get_move_to_reach_position(agent, next_position), move_cost
    
    
    def assign_shortest_path_for_goal_to_agent(self, agent: Agent, goal: Goal):
        path = self.generate_shortest_path_for_agent(agent, goal)
        self.assign_path_to_agent(agent, path)

    def _calculate_path_cost(self, path):
        return nx.path_weight(self.grid_map.graph, path, weight="weight")