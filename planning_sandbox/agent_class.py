
class Agent:
    def __init__(self, initial_position):
        self.initial_position = initial_position
        self.position = initial_position
        self.skills = []

    def reset(self):
        self.position = self.initial_position

    def move_left(self):
        self.position = (self.position[0] - 1, self.position[1])

    def move_right(self):
        self.position = (self.position[0] + 1, self.position[1])

    def move_up(self):
        self.position = (self.position[0], self.position[1] - 1)

    def move_down(self):
        self.position = (self.position[0], self.position[1] + 1)

    def remain_still(self):
        pass

    def move_to_position(self, position):
        self.position = position

    def apply_action(self, action):
        if action == 1 or action == 'left':
            self.move_left()
        elif action == 2 or action == 'right':
            self.move_right()
        elif action == 3 or action == 'up':
            self.move_up()
        elif action == 4 or action == 'down':
            self.move_down()
        elif action == 0 or action == 'stay':
            self.remain_still()
    
    def add_skill(self, skill):
        self.skills.append(skill)

    def get_skills(self):
        if len(self.skills) == 0:
            return None
        return self.skills
