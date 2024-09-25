from planning_sandbox.grid_map_class import GridMap
from planning_sandbox.agent_class import Agent

class Controller:
    def __init__(self, grid_map):
        
        self.grid_map: GridMap = grid_map

        self.action_map = {
            0: 'stay',
            1: 'left',
            2: 'right',
            3: 'up',
            4: 'down'
        }


    def validate_action(self, agent: Agent, action):
        position = agent.position
        if action == 'left' or action == 1:
            position = (position[0] - 1, position[1])
        elif action == 'right' or action == 2:
            position = (position[0] + 1, position[1])
        elif action == 'up' or action == 3:
            position = (position[0], position[1] - 1)
        elif action == 'down' or action == 4:
            position = (position[0], position[1] + 1)
        elif action == 'stay' or action == 0:
            return True
        is_valid = self.grid_map.is_valid_position(position)
        return is_valid

    def reset(self):
        pass