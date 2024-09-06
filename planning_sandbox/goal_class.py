import numpy as np

class Goal:
    def __init__(self, position):
        self.position = position
        self.required_skills = []
        self.claimed = False

    def add_skill(self, skill):
        self.required_skills.append(skill)

    def get_required_skills(self):
        if len(self.required_skills) == 0:
            return None
        return self.required_skills
    
    def claim(self):
        self.claimed = True 

    def reset(self, position=None):
        self.claimed = False
        self.required_skills = []
        if position is not None:
            self.position = position

