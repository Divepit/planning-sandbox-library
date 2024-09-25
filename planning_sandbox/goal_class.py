import numpy as np

from typing import List
from itertools import combinations

from planning_sandbox.agent_class import Agent

class Goal:
    def __init__(self, position):
        self.position = position
        self.required_skills = []
        self.claimed = False
        self.agents_which_have_required_skills: List[Agent] = []
        self.agent_combinations_which_solve_goal = {}
        self.cost_to_reach_other_goals = {}
        self.cheapest_combination = (None,np.inf)

    def add_skill(self, skill):
        self.required_skills.append(skill)
    
    def claim(self):
        self.claimed = True 

    def reset(self, position=None):
        self.claimed = False
        self.required_skills.clear()
        self.agents_which_have_required_skills.clear()
        self.agent_combinations_which_solve_goal.clear()
        self.cost_to_reach_other_goals.clear()
        if position is not None:
            self.position = position

    def soft_reset(self):
        self.agents_which_have_required_skills.clear()
        self.agent_combinations_which_solve_goal.clear()
        self.cheapest_combination = (None,np.inf)
        
    def add_agent_which_has_required_skills(self, agent):
        self.agents_which_have_required_skills.append(agent)

    def add_cost_to_reach_other_goal(self, goal, cost):
        self.cost_to_reach_other_goals[goal] = cost

    def generate_agent_combinations_which_solve_goal(self):
        for r in range(1, len(self.agents_which_have_required_skills)+1):
            new_combinations = combinations(self.agents_which_have_required_skills, r)
            for combination in new_combinations:
                if combination not in self.agent_combinations_which_solve_goal and set(self.required_skills).issubset(set([skill for agent in combination for skill in agent.skills])):
                    combination_cost = 0
                    for agent in combination:
                        if self in agent.costs_to_reach_goals:
                            combination_cost += agent.costs_to_reach_goals[self]
                    self.agent_combinations_which_solve_goal[combination] = combination_cost
        

                
    

