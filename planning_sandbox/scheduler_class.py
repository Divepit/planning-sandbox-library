from typing import List, Dict
from itertools import permutations, product, combinations
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal

import numpy as np

class Scheduler:
    def __init__(self, agents, goals):
        self.agents: List[Agent] = agents
        self.goals: List[Goal] = goals
        self.unclaimed_goals: List[Goal] = [goal for goal in goals if not goal.claimed]
        self.goal_assignments: Dict[Agent, Goal] = {}

    def _get_agents_present_at_goal(self, goal: Goal):
        return [agent for agent in self.agents if agent.position == goal.position]
    
    def _get_skills_of_agents_present_at_goal(self, goal: Goal):
        agents = self._get_agents_present_at_goal(goal)
        if not agents:
            return []
        skills = []
        for agent in agents:
            skills.extend(agent.skills)
        return skills
    
    def _goal_can_be_claimed(self, goal: Goal):
        skills_of_agents_present = self._get_skills_of_agents_present_at_goal(goal)
        if not skills_of_agents_present:
            return False
        skills_required = goal.required_skills
        if set(skills_required).issubset(set(skills_of_agents_present)):
            return True

    def _calculate_cost_of_solution(self, solution: Dict[Agent, List[Goal]], max_cost) -> int:
        solution_cost = 0
        for agent, goals in solution.items():
            cost, _ = self._calculate_cost_of_chain(agent, goals)
            solution_cost += cost
            if solution_cost > max_cost:
                return np.inf
        return solution_cost
    
    def _calculate_cost_of_chain(self, agent: Agent, chain: List[Goal]):
        cost = 0
        cost += agent.paths_and_costs_to_goals[chain[0]][1]
        length = len(agent.paths_and_costs_to_goals[chain[0]][0])
        for i in range(1,len(chain)):
            previous_goal = chain[i-1]
            current_goal = chain[i]
            cost += previous_goal.paths_and_costs_to_other_goals[current_goal][1]
            length += len(previous_goal.paths_and_costs_to_other_goals[current_goal][0])
        return cost, length
    
    def reset(self):
        self.goal_assignments = {}
        self.unclaimed_goals = [goal for goal in self.goals if not goal.claimed]

    def get_normalized_claimed_goals(self):
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
        claimed_a_goal = False
        for goal in self.goals:
            if goal.claimed:
                continue
            if self._goal_can_be_claimed(goal):
                goal.claimed = True
                claimed_a_goal = True
        self.unclaimed_goals = [goal for goal in self.goals if not goal.claimed]
        return claimed_a_goal
    
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

    # SOLVERS            
    
    def find_fast_solution(self):
        cheapest_combinations = {} # goal: (combination, cost)
        cheapest_solution: Dict[Agent, List[Goal]] = {} # agent: [goal]
        unaccounted_for_goals = set(self.unclaimed_goals)
        while len(cheapest_solution) != len(self.agents):
            for goal in unaccounted_for_goals:
                sorted_combinations = iter(sorted(goal.agent_combinations_which_solve_goal.items(), key=lambda combo_and_cost: combo_and_cost[1]))
                looking_for_goal_solution = True
                while looking_for_goal_solution:
                    try:
                        (cheapest_goal_combination,cost) = next(sorted_combinations)
                        looking_for_goal_solution = False
                    except StopIteration:
                        break
                    if any([agent in cheapest_solution for agent in cheapest_goal_combination]):
                        looking_for_goal_solution = True
                        continue
                if not looking_for_goal_solution:
                    cheapest_combinations[goal] = (cheapest_goal_combination,cost)
            
            cheapest_goals_sorted = iter(sorted(cheapest_combinations.items(), key=lambda goal_and_combo_and_cost: goal_and_combo_and_cost[1][1]))

            goal_available = True
            while True:
                try:
                    cheapest_goal, (cheapest_combination, cost) = next(cheapest_goals_sorted)
                except StopIteration:
                    goal_available = False
                    break

                if any([agent in cheapest_solution for agent in cheapest_combination]):
                    continue
                else:
                    break
            
            if not goal_available:
                cheapest_combinations.clear()
                break

            for agent in cheapest_combination:
                cheapest_solution[agent] = [cheapest_goal]

            unaccounted_for_goals.remove(cheapest_goal)
            cheapest_combinations.clear()

        return cheapest_solution
    
    def find_optimal_solution(self):
        
        cheapest_solution = None
        cheapest_cost = np.inf
        all_goal_orders = iter(permutations(self.unclaimed_goals, len(self.unclaimed_goals)))

        for goal_order in all_goal_orders:
            candidate_permutations =iter(product(*[goal.agent_combinations_which_solve_goal.keys() for goal in goal_order]))
            for candidate_permutation in candidate_permutations:
                proposed_solution = {}
                proposed_solution_cost = np.inf
                for i, agent_list in enumerate(candidate_permutation):
                    goal = goal_order[i]
                    for agent in agent_list:
                        if agent not in proposed_solution:
                            proposed_solution[agent] = []
                        proposed_solution[agent].append(goal)
                    proposed_solution_cost = self._calculate_cost_of_solution(solution=proposed_solution, max_cost=cheapest_cost)
                if cheapest_solution is None or proposed_solution_cost < cheapest_cost:
                    cheapest_solution = proposed_solution
                    cheapest_cost = proposed_solution_cost
        return cheapest_solution
    
    def find_linalg_solution(self):
        agent_combination_vector = []
        for r in range(1, len(self.agents)+1):
            agent_combination_vector.extend(combinations(self.agents, r))
        
        goal_permutation_vector = []
        for r in range(1, len(self.unclaimed_goals)+1):
            goal_permutation_vector.extend(permutations(self.unclaimed_goals, r))
        
        agent_combination_goal_permutation_matrix = np.zeros((len(agent_combination_vector), len(goal_permutation_vector)))

        for i,agent_combination in enumerate(agent_combination_vector):
            for j,goal_permutation in enumerate(goal_permutation_vector):
                if all([goal_permutation[0] in agent.paths_and_costs_to_goals.keys() for agent in agent_combination]) and all([self.agent_combination_has_required_skills_for_goal(agent_combination, goal) for goal in goal_permutation]):
                    agent_combination_goal_permutation_matrix[i,j] = np.sum([self._calculate_cost_of_chain(agent, goal_permutation) for agent in agent_combination])
                else:
                    agent_combination_goal_permutation_matrix[i,j] = np.inf

        sorted_indices = np.argsort(agent_combination_goal_permutation_matrix, axis=None)

        rows, cols = np.unravel_index(sorted_indices, agent_combination_goal_permutation_matrix.shape)

        covered_goals = set()
        busy_agents = set()
        solution = {}
        for r, c in zip(rows, cols):
            agent_combination = agent_combination_vector[r]
            goal_permutation = goal_permutation_vector[c]
            if any([goal in covered_goals for goal in goal_permutation]):
                continue
            for agent in agent_combination:
                busy_agents.add(agent)
                if agent not in solution:
                    solution[agent] = []
                solution[agent].extend(goal_permutation)
            for goal in goal_permutation_vector[c]:
                covered_goals.add(goal)
            if len(covered_goals) == len(self.unclaimed_goals):
                break
        return solution