class Agent:
    def __init__(self, initial_position):
        self.initial_position = initial_position
        self.position = initial_position
        self.skills = []
        self.paths_and_costs_to_goals = {}

    def reset(self, position=None):
        if position is not None:
            self.position = position
        else:
            self.position = self.initial_position
        self.skills.clear()
        self.paths_and_costs_to_goals.clear()

    def _move_left(self):
        self.position = (self.position[0] - 1, self.position[1])

    def _move_right(self):
        self.position = (self.position[0] + 1, self.position[1])

    def _move_up(self):
        self.position = (self.position[0], self.position[1] - 1)

    def _move_down(self):
        self.position = (self.position[0], self.position[1] + 1)

    def _stay(self):
        pass

    def move_to_position(self, position):
        self.position = position

    def apply_action(self, action):
        if action == 1 or action == 'left':
            self._move_left()
        elif action == 2 or action == 'right':
            self._move_right()
        elif action == 3 or action == 'up':
            self._move_up()
        elif action == 4 or action == 'down':
            self._move_down()
        elif action == 0 or action == 'stay':
            self._stay()
        else:
            raise ValueError("Invalid action: " + str(action))
    
    def add_skill(self, skill):
        self.skills.append(skill)

    def add_path_to_goal(self, goal, path, cost):
        self.paths_and_costs_to_goals[goal] = (path, cost)


            
