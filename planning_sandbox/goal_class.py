import numpy as np
from itertools import combinations

class Goal:
    def __init__(self, position):
        self.position = position
        self.required_skills = []
        self.claimed = False
        self.agents_which_have_required_skills = []
        self.agent_combinations_which_solve_goal = {}
        self.cost_to_reach_other_goals = {}
        self.cheapest_combination = (None,np.inf)

    def add_skill(self, skill):
        self.required_skills.append(skill)
    
    def claim(self):
        # print('Goal claimed')
        self.claimed = True 

    def reset(self, position=None):
        self.claimed = False
        self.required_skills.clear()
        self.agents_which_have_required_skills.clear()
        self.agent_combinations_which_solve_goal.clear()
        self.cost_to_reach_other_goals.clear()
        if position is not None:
            self.position = position

    def add_agent_which_has_required_skills(self, agent):
        self.agents_which_have_required_skills.append(agent)
        self.append_agent_combinations_which_solve_goal(agent)

    def add_cost_to_reach_other_goal(self, goal, cost):
        self.cost_to_reach_other_goals[goal] = cost

    def append_agent_combinations_which_solve_goal(self, new_agent):
        for r in range(1, len(self.agents_which_have_required_skills)+1):
            new_combinations = combinations(self.agents_which_have_required_skills, r)
            for combination in new_combinations:
                if combination not in self.agent_combinations_which_solve_goal and set(self.required_skills).issubset(set([skill for agent in combination for skill in agent.skills])):
                    combination_cost = 0
                    for agent in combination:
                        if self in agent.costs_to_reach_goals:
                            combination_cost += agent.costs_to_reach_goals[self]
                    self.agent_combinations_which_solve_goal[combination] = combination_cost
                    if combination_cost < self.cheapest_combination[1]:
                        self.cheapest_combination = (combination, combination_cost)
        

                
    

