import numpy as np

class Goal:
    def __init__(self, position):
        self.position = position
        self.required_skills = []
        self.claimed = False

    def add_skill(self, skill):
        self.required_skills.append(skill)
    
    def claim(self):
        # print('Goal claimed')
        self.claimed = True 

    def reset(self, position=None):
        self.claimed = False
        self.required_skills.clear()
        if position is not None:
            self.position = position

