import numpy as np

from typing import List, Dict, Tuple
from itertools import combinations

from planning_sandbox.agent_class import Agent

class Goal:
    def __init__(self, position):
        self.position = position
        self.initial_position = position
        self.required_skills = []
        self.claimed = False
        self.agents_which_have_required_skills: List[Agent] = []
        self.agent_combinations_which_solve_goal: Dict[Tuple[Agent], np.float32] = {}
        self.cheapest_combination = (None,np.inf)
        self.paths_and_costs_to_other_goals = {}

    def add_skill(self, skill):
        self.required_skills.append(skill)

    def reset(self, position=None):
        self.claimed = False
        self.required_skills.clear()
        self.agents_which_have_required_skills.clear()
        self.agent_combinations_which_solve_goal.clear()
        self.paths_and_costs_to_other_goals.clear()
        self.cheapest_combination = (None,np.inf)
        if position is not None:
            self.position = position
            self.initial_position = position

    def soft_reset(self):
        self.claimed = False
        
    def add_agent_which_has_required_skills(self, agent):
        if agent not in self.agents_which_have_required_skills:
            self.agents_which_have_required_skills.append(agent)

    def add_path_to_other_goal(self, goal, path, cost):
        self.paths_and_costs_to_other_goals[goal] = (path, cost)


    def generate_agent_combinations_which_solve_goal(self):
        for r in range(1, len(self.agents_which_have_required_skills)+1):
            new_combinations = combinations(self.agents_which_have_required_skills, r)
            for combination in new_combinations:
                combination_in_agent_combinations = combination in self.agent_combinations_which_solve_goal
                required_skills_are_covered = set(self.required_skills).issubset(set([skill for agent in combination for skill in agent.skills]))
                each_agent_has_skill_to_this_goal = all([self in agent.paths_and_costs_to_goals for agent in combination])


                if not combination_in_agent_combinations and required_skills_are_covered and each_agent_has_skill_to_this_goal:
                    combination_cost = 0
                    for agent in combination:
                        if self in agent.paths_and_costs_to_goals:
                            combination_cost += agent.paths_and_costs_to_goals[self][1]
                    self.agent_combinations_which_solve_goal[combination] = combination_cost
        

                
    

