import numpy as np

class Controller:
    def __init__(self, agents, grid_map):
        self.agents = agents
        
        self.grid_map = grid_map

        self.action_map = {
            0: 'stay',
            1: 'left',
            2: 'right',
            3: 'up',
            4: 'down'
        }

    def validate_action(self, agent, action):
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
        return self.grid_map.is_valid_position(position)
    
    def get_valid_actions(self, agent):
        valid_actions = []
        for action in self.action_map.values():
            if self.validate_action(agent, action):
                valid_actions.append(action)
        return valid_actions
    
    def get_random_valid_action(self, agent):
        valid_actions = self.get_valid_actions(agent)
        return np.random.choice(valid_actions)
