from typing import List, Dict
import logging

from itertools import permutations, product, combinations

from planning_sandbox.grid_map_class import GridMap
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal
from planning_sandbox.scheduler_class import Scheduler
from planning_sandbox.benchmark_class import Benchmark

import numpy as np

class Environment:
    def __init__(self, size, num_agents, num_goals, num_skills, use_geo_data=False, solve_type="optimal",replan_on_goal_claim=False):
        self.size = size
        self.map_diagonal = np.sqrt(2 * (size ** 2))
        self.solve_type = solve_type
        self.num_skills = num_skills
        self.replan_on_goal_claim = replan_on_goal_claim
        self.deadlocked = False

        self.goals: List[Goal] = []
        self.agents: List[Agent] = []
        self.scheduler = Scheduler(agents=self.agents, goals=self.goals)
        self.full_solution = {}
        

        self._initial_num_agents = num_agents
        self._initial_num_goals = num_goals
        
        self.grid_map = GridMap(self.size, use_geo_data=use_geo_data)
        self._starting_position = self.grid_map.random_valid_position()

        self.initialised = False
        self.agents_goals_connected = False
        
        self._init()
        self._log_environment_info()

    def _init(self):
        assert not self.initialised, "Environment already initialised"
        self._initialize_goals()
        self._initialize_agents()

        while not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()
        
        self.scheduler.reset()
        
        self.initialised = True

    def _init_agent_goal_connections(self):
        assert self.initialised, "Environment not initialised"
        assert not self.agents_goals_connected, "Agents and goals already connected"
        self.connect_agents_and_goals()
        self.inform_goals_of_costs_to_other_goals()
        self.agents_goals_connected = True

    def _log_environment_info(self):
        logging.debug(f"=== Environment settings ===")
        logging.debug(f"Num agents: {len(self.agents)}")
        logging.debug(f"Num goals: {len(self.goals)}")
        logging.debug(f"Num skills: {self.num_skills}")
        logging.debug(f"Map size: {self.size}")
        logging.debug(f"=== === === === === === ===")
        
    def _initialize_agents(self):
        if self._starting_position is not None:
            start_pos = self._starting_position
        else:
            start_pos = self.grid_map.random_valid_position()
        for _ in range(self._initial_num_agents):
            agent = Agent(start_pos)
            self.agents.append(agent)

    def _initialize_goals(self):
        for _ in range(self._initial_num_goals):
            random_position = self.grid_map.random_valid_position()
            self._add_goal(random_position)
        self.new_goal_added = False

    def _add_goal(self, position):
        goal: Goal = Goal(position)
        self.goals.append(goal)
        self.new_goal_added = True
        return goal

    def _initialize_skills(self):
        if self.num_skills == 1:
            for goal in self.goals:
                if goal.required_skills == []:
                    goal.required_skills.append(0)
            for agent in self.agents:
                if agent.skills == []:
                    agent.skills.append(0)
            return

        for goal in self.goals:
            amount_of_skills = np.random.randint(1, min(3,self.num_skills+1))
            skills = []
            for _ in range(amount_of_skills):
                skill = np.random.randint(0, self.num_skills)
                while skill in skills:
                    skill = np.random.randint(0, self.num_skills)
                skills.append(skill)
            goal.required_skills = skills
                
        for agent in self.agents:
            if self.num_skills == 2:
                amount_of_skills = 1
            else:
                amount_of_skills = np.random.randint(1, max(1,min(3,self.num_skills)))
            skills = []
            for _ in range(amount_of_skills):
                skill = np.random.randint(0, self.num_skills)
                while skill in skills:
                    skill = np.random.randint(0, self.num_skills)
                skills.append(skill)
            agent.skills = skills

    def _all_skills_represented(self):
        all_skills = [0]*self.num_skills
        for agent in self.agents:
            for skill in agent.skills:
                all_skills[skill] = 1
        return all(all_skills)
    
    def _reset_skills(self):
        for agent in self.agents:
            agent.skills = []
        for goal in self.goals:
            goal.required_skills = []
        self._initialize_skills()  

    def connect_agents_and_goals(self):
        inform_goals_of_agents_bench = Benchmark("inform_goals_of_agents", start_now=True, silent=True)
        for goal in self.scheduler.unclaimed_goals:
            # goal.agents_which_have_required_skills.clear()
            # goal.agent_combinations_which_solve_goal.clear()
            # goal.cheapest_combination = (None,np.inf)
            for agent in self.agents:
                if any(skill in agent.skills for skill in goal.required_skills):
                    goal.add_agent_which_has_required_skills(agent)
        inform_goals_of_agents_bench.stop()

        inform_agents_of_costs_bench = Benchmark("inform_agents_of_costs_to_goals", start_now=True, silent=True)
        for goal in self.scheduler.unclaimed_goals:
            for agent in goal.agents_which_have_required_skills:
                    path = self.grid_map.generate_shortest_path_for_agent(agent, goal)
                    cost = self.grid_map.calculate_path_cost(path)
                    agent.add_path_to_goal(goal, path, cost)
        inform_agents_of_costs_bench.stop()

        for goal in self.scheduler.unclaimed_goals:
            goal.generate_agent_combinations_which_solve_goal()

    def inform_goals_of_costs_to_other_goals(self):
        inform_goals_of_costs_bench = Benchmark("inform_goals_of_costs_to_other_goals", start_now=True, silent=True)
        for goal in self.scheduler.unclaimed_goals:
            for other_goal in self.goals:
                if goal == other_goal:
                    continue

                if not other_goal in goal.paths_and_costs_to_other_goals:
                    path = self.grid_map.shortest_path(goal.position, other_goal.position)
                    cost = self.grid_map.calculate_path_cost(path)
                    goal.add_path_to_other_goal(other_goal, path, cost)

                if not goal in other_goal.paths_and_costs_to_other_goals:
                    path = self.grid_map.shortest_path(other_goal.position, goal.position)
                    cost = self.grid_map.calculate_path_cost(path)
                    other_goal.add_path_to_other_goal(goal, path, cost)


        inform_goals_of_costs_bench.stop()        

    def soft_reset(self):
        self.deadlocked = False
        self.grid_map.reset()
        for goal in self.goals:
            goal.soft_reset()

        for agent in self.agents:
            agent.soft_reset()

        self.scheduler.reset()

    def reset(self):
        self.initialised = False
        self.deadlocked = False
        self.agents_goals_connected = False
        self.grid_map.reset()
        self.full_solution = {}
        self._starting_position = self.grid_map.random_valid_position()
        self.goals.clear()
        self.agents.clear()
        self._init()
        
        



    def get_normalized_skill_vectors_for_all_agents(self):
        assert self.initialised, "Environment not initialised"
        all_skills_normalized = []
        for agent in self.agents:
            agent_skills_normalized = [0]*self.num_skills
            for skill in agent.skills:
                agent_skills_normalized[skill] = 1
            all_skills_normalized.extend(agent_skills_normalized)
        return all_skills_normalized
    
    def get_normalized_skill_vectors_for_all_goals(self):
        assert self.initialised, "Environment not initialised"
        all_skills_normalized = []
        for goal in self.goals:
            goal_skills_normalized = [0]*self.num_skills
            for skill in goal.required_skills:
                goal_skills_normalized[skill] = 1
            all_skills_normalized.extend(goal_skills_normalized)
        return all_skills_normalized        

    def find_numerical_solution(self, solve_type=None):
        if not self.agents_goals_connected:
            self._init_agent_goal_connections()
        if solve_type is not None:
            self.solve_type = solve_type
        if self.solve_type == "optimal":
            self.replan_on_goal_claim = False
            self.full_solution = self.find_optimal_solution()
        elif self.solve_type == "fast":
            self.replan_on_goal_claim = True
            if self.full_solution is None:
                self.full_solution = {}
            intermediate_solution = self.scheduler.find_fast_solution()
            for agent in intermediate_solution:
                if agent not in self.full_solution:
                    self.full_solution[agent] = []
                self.full_solution[agent].extend(intermediate_solution[agent])
        elif self.solve_type == "linalg":
            self.replan_on_goal_claim = False
            self.full_solution = self.find_linalg_solution()
        return self.full_solution

    def step_environment(self, fast=False):
        logging.debug("Stepping environment")
        for agent, goal_list in self.full_solution.items():
            for i, goal in enumerate(goal_list):
                if goal.claimed:
                    continue
                self.scheduler.goal_assignments[agent] = goal
                break # OH MY GOD DO NOT REMOVE THIS BREAK
        not_deadlocked = self.update(fast=fast)
        return not_deadlocked

    def solve_full_solution(self, fast=False):
        
        self.soft_reset()
        
        solving_bench: Benchmark = Benchmark("solve_full_solution", start_now=True, silent=True)
        while not (self.deadlocked or self.scheduler.all_goals_claimed()):
            self.step_environment(fast=fast)
        
        if self.deadlocked:
            logging.debug("Deadlocked")

        solve_time = solving_bench.stop()

        total_steps, steps_waited, total_cost = self.get_agent_benchmarks()
        amount_of_claimed_goals = len(self.goals) - len(self.scheduler.unclaimed_goals)

        return total_steps, steps_waited, total_cost, solve_time, amount_of_claimed_goals
    
    def get_observation_vector(self):
        goals_map = np.zeros((self.size, self.size), dtype=np.int16)
        for goal in self.goals:
            goals_map[goal.position[0], goal.position[1]] = 1
        agents_map = np.zeros((self.size, self.size), dtype=np.int16)
        for agent in self.agents:
            agents_map[agent.position[0], agent.position[1]] = 1

        flattened_map = self.grid_map.downscaled_data.flatten()
        min_value = np.min(flattened_map)
        max_value = np.max(flattened_map)
        normalized_map = 2*((flattened_map - min_value) / (max_value - min_value))-1

        observation_vector = {
            
            "claimed_goals": np.array([1 if goal.claimed else 0 for goal in self.goals], dtype=np.int16),
            "map_elevations": normalized_map.astype(np.int16),
            "goal_positions": goals_map.flatten(),
            "agent_positions": agents_map.flatten(),
            "goal_required_skills": np.array([[(1 if skill in goal.required_skills else 0) for skill in range(self.num_skills)] 
            for goal in self.goals], dtype=np.int16).flatten(),
            "agent_skills": np.array([[(1 if skill in agent.skills else 0) for skill in range(self.num_skills)] 
            for agent in self.agents], dtype=np.int16).flatten(),
        }
        return observation_vector
    
    def get_action_vector(self):
        action_vector = []
        for agent in self.agents:
            if agent in self.full_solution:
                goal_list = self.full_solution[agent]
                # Pad with -1 (or use `num_goals` as a dummy) if fewer than max_goals_per_agent
                padded_goal_list = [self.goals.index(goal) for goal in goal_list] + [-1] * (len(self.goals) - len(goal_list))
                action_vector.append(padded_goal_list)
            else:
                # No goals, append `-1` for all slots
                action_vector.append([-1] * len(self.goals))
        
        # Flatten the list for the MultiDiscrete action space
        flattened_action_vector = [goal for sublist in action_vector for goal in sublist]
        
        return flattened_action_vector

        
    
    def get_full_solution_from_action_vector(self, action_vector):
        full_solution = {}
        corrected_action = action_vector - 1
        for flat_index, selected_goal in enumerate(corrected_action):
            if selected_goal != -1:  # Only process if action is valid
                # Compute agent and goal indices from flat_index
                agent_index = flat_index // len(self.goals)
                goal_index = flat_index % len(self.goals)
                
                agent = self.agents[agent_index]
                goal = self.goals[goal_index]

                # Add goal to the agent's full solution
                if agent not in full_solution:
                    full_solution[agent] = []
                full_solution[agent].append(goal)
        return full_solution


    def update(self, fast=False):
        logging.debug("Updating environment")
        agent_positions_start = [agent.position for agent in self.agents]
        for agent,goal in self.scheduler.goal_assignments.items():
            if (not self.replan_on_goal_claim) and fast:
                logging.debug("Fast update")
                agent.move_to_position(goal.position)
            else:
                self.grid_map.assign_shortest_path_for_goal_to_agent(agent=agent, goal=goal)
                path = self.grid_map.paths[agent]
                cost = self.grid_map.calculate_path_cost(path)
                action, action_cost = self.grid_map.get_move_and_cost_to_reach_next_position(agent)
                agent.apply_action(action, action_cost)

        claimed_a_goal = self.scheduler.update_goal_statuses()
        if (claimed_a_goal and self.replan_on_goal_claim) or self.new_goal_added:
            self._replan()
        
        agent_positions_end = [agent.position for agent in self.agents]
        if not self.scheduler.all_goals_claimed():
            if agent_positions_start == agent_positions_end:
                self.deadlocked = True
            else:
                self.deadlocked = False
        logging.debug("Environment updated")
        return not self.deadlocked


    def _replan(self):
        logging.debug("Replanning")
        self.connect_agents_and_goals()
        new_goal = False
        for goal in self.goals:
            if  len(goal.paths_and_costs_to_other_goals) == 0:
                new_goal = True
        if new_goal:
            self.inform_goals_of_costs_to_other_goals()
        self.find_numerical_solution()
        self.new_goal_added = False

    
    def get_agent_benchmarks(self):
        total_steps = 0
        steps_waited = 0
        total_cost = 0
        for agent in self.agents:
            total_steps += agent.steps_moved
            steps_waited += agent.steps_waited
            total_cost += agent.accumulated_cost
        return total_steps, steps_waited, total_cost

    
    def _calculate_cost_of_closed_solution(self, solution: Dict[Agent, List[Goal]], max_cost=np.inf) -> int:
        solution_cost = 0
        for agent, goals in solution.items():
            cost, _ = self._calculate_cost_of_chain(agent, goals)
            solution_cost += cost
            if solution_cost > max_cost:
                return np.inf
        return solution_cost
    
    def _calculate_cost_of_chain(self, agent: Agent, chain: List[Goal]):
        cost = 0
        length = 0
        if not chain:
            return cost, length
        
        first_goal = chain[0]

        if self.agents_goals_connected and first_goal in agent.paths_and_costs_to_goals:
            cost = agent.paths_and_costs_to_goals[first_goal][1]
            length = len(agent.paths_and_costs_to_goals[first_goal][0])
        else:
            path = self.grid_map.generate_shortest_path_for_agent(agent, first_goal)
            cost = self.grid_map.calculate_path_cost(path)
            length = len(path)

        for i in range(1,len(chain)):
            previous_goal = chain[i-1]
            current_goal = chain[i]
            if previous_goal == current_goal:
                continue
            if self.agents_goals_connected:
                cost += previous_goal.paths_and_costs_to_other_goals[current_goal][1]
                length += len(previous_goal.paths_and_costs_to_other_goals[current_goal][0])
            else:
                path = self.grid_map.shortest_path(previous_goal.position, current_goal.position)
                cost += self.grid_map.calculate_path_cost(path)
                length += len(path)
        return cost, length
    
    def find_optimal_solution(self):
        assert self.agents_goals_connected, "Agents and goals not connected"
        logging.debug("Finding optimal solution")
        full_solution = None
        cheapest_cost = np.inf
        all_goal_orders = iter(permutations(self.scheduler.unclaimed_goals, len(self.scheduler.unclaimed_goals)))
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
                    proposed_solution_cost = self._calculate_cost_of_closed_solution(solution=proposed_solution, max_cost=cheapest_cost)
                if full_solution is None or proposed_solution_cost < cheapest_cost:
                    full_solution = proposed_solution
                    cheapest_cost = proposed_solution_cost
        return full_solution
    
    def find_linalg_solution(self):
        assert self.agents_goals_connected, "Environment not initialised"
        logging.debug("Finding optimal solution")
        agent_combination_vector = []
        for r in range(1, len(self.agents)+1):
            agent_combination_vector.extend(combinations(self.agents, r))
        
        goal_permutation_vector = []
        for r in range(1, len(self.scheduler.unclaimed_goals)+1):
            goal_permutation_vector.extend(permutations(self.scheduler.unclaimed_goals, r))
        
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
            if len(covered_goals) == len(self.scheduler.unclaimed_goals):
                break
        return solution