from typing import List, Dict
from itertools import combinations, product
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal

import numpy as np

class Scheduler:
    def __init__(self, agents, goals):
        self.agents: List[Agent] = agents
        self.goals: List[Goal] = goals
        self.unclaimed_goals: List[Goal] = goals
        
        self.goal_assignments = {}

    def _get_agents_present_at_goal(self, goal: Goal):
        return [agent for agent in self.agents if agent.position == goal.position]
    
    def _get_skills_of_agents_present_at_goal(self, goal: Goal):
        agents = self._get_agents_present_at_goal(goal)
        if not agents:
            return []
        # print(f"Agents present at goal: {agents}")
        skills = []
        for agent in agents:
            skills.extend(agent.skills)
        return skills

    
    def _goal_can_be_claimed(self, goal: Goal):
        skills_of_agents_present = self._get_skills_of_agents_present_at_goal(goal)
        if not skills_of_agents_present:
            return False
        # print(f"Skills of agents present at goal: {skills_of_agents_present}")
        skills_required = goal.required_skills
        # print(f"Skills required for goal: {skills_required}, Skills of agents present: {skills_of_agents_present}")
        if set(skills_required).issubset(set(skills_of_agents_present)):
            return True
    
    def _update_goal_status(self, goal: Goal):
        amount_of_claimed_goals = 0
        if goal.claimed:
            return 0
        if self._goal_can_be_claimed(goal):
            # print(f"Goal at position {goal.position} can be claimed")
            goal.claim()
            amount_of_claimed_goals += 1
        self.unclaimed_goals = [goal for goal in self.goals if not goal.claimed]
        return amount_of_claimed_goals
    
    def reset(self):
        self.goal_assignments = {}
        self.unclaimed_goals = self.goals

    def assign_goal_to_agent(self, agent: Agent, goal: Goal):
        if agent in self.goal_assignments:
            self.goal_assignments[agent] = goal


    def get_normalized_claimed_goals(self):
        # 1 if claimed, 0 if not
        return [int(goal.claimed) for goal in self.goals]
    
    def get_allowed_goal_agent_pairs(self, goals: List[Goal], agents: List[Agent]):
        allowed_pairs = []
        for agent in agents:
            for goal in goals:
                if self.agent_has_one_or_more_required_skills_for_goal(agent, goal):
                    allowed_pairs.append((agent, goal))
        return allowed_pairs
    
    def all_goals_claimed(self):
        return len(self.unclaimed_goals) == 0
    
    def update_goal_statuses(self):
        amount_of_claimed_goals = 0
        for goal in self.goals:
            amount_of_claimed_goals += self._update_goal_status(goal)
        return amount_of_claimed_goals
    
    def is_goal_position(self, position):
        return any([goal.position == position for goal in self.goals])
    
    def get_goal_at_position(self, position):
        for goal in self.goals:
            if goal.position == position:
                return goal
        return None
    
    def agent_has_one_or_more_required_skills_for_goal(self, agent: Agent, goal: Goal):
        return any([skill in agent.skills for skill in goal.required_skills])

    def agent_combination_has_required_skills_for_goal(self, agents: List[Agent], goal: Goal):
        skills = []
        for agent in agents:
            skills.extend(agent.skills)
        return set(goal.required_skills).issubset(set(skills))
    
    def _get_all_goal_solutions(self, agent_goal_plans):
        goal_solutions: Dict[Goal,List[List[Agent]]] = {}
        for goal in self.unclaimed_goals:
            goal_solutions[goal] = []
            for r in range(1,len(self.agents)+1):
                combos = combinations(self.agents, r)
                for combination in combos:
                    invalid_combination = False
                    for agent in combination:
                        if (agent,goal) not in agent_goal_plans:
                            invalid_combination = True
                            break
                    if invalid_combination:
                        invalid_combination = False
                        continue
                    if self.agent_combination_has_required_skills_for_goal(agents=combination, goal=goal):
                        goal_solutions[goal].append(combination)
        return goal_solutions
    
    def _get_all_possible_solutions(self, goal_solutions):
        solution_lists = list(goal_solutions.values())
        combinations_of_agents_for_goals = product(*solution_lists)
        possible_solutions = []
        for combination_set in combinations_of_agents_for_goals:
            possible_solution = {}
            for i,combination in enumerate(combination_set):
                goal = list(goal_solutions.keys())[i]
                for agent in combination:
                    if agent not in possible_solution:
                        possible_solution[agent] = []
                    possible_solution[agent].append(goal)
            possible_solutions.append(possible_solution)
        return possible_solutions
    
    def _calculate_cost_of_solution(self, solution: Dict[Agent, List[Goal]], agent_goal_costs, goal_goal_costs) -> int:
        solution_cost = []
        for agent, goals in solution.items():
            if goals == []:
                continue
            first_goal = goals[0]
            path_cost = agent_goal_costs[(agent,first_goal)]
            for i in range(2,len(goals)):
                path_cost += goal_goal_costs[(goals[i-1],goals[i])]
            solution_cost.append(path_cost)
        return np.sum(solution_cost)
    

    def _get_cheapest_solution(self, possible_solutions, agent_goal_costs, goal_goal_costs):
        cheapest_solution = None
        cheapest_cost = np.inf
        for solution in possible_solutions:
            cost = self._calculate_cost_of_solution(solution=solution, agent_goal_costs=agent_goal_costs, goal_goal_costs=goal_goal_costs)
            if cost < cheapest_cost:
                cheapest_cost = cost
                cheapest_solution = solution
        return cheapest_solution