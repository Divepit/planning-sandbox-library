from typing import List

import copy
import logging

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

        self._init()
        
        self._log_environment_info()

    def _init(self):
        self._initialize_goals()
        self._initialize_agents()

        while not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()

        self.connect_agents_and_goals()
        self.inform_goals_of_costs_to_other_goals()

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
        self.deadlocked = False
        self.grid_map.reset()
        for goal in self.goals:
            new_position = self.grid_map.random_valid_position()
            goal.reset(new_position)

        self._starting_position = self.grid_map.random_valid_position()

        for agent in self.agents:
            agent.reset(self._starting_position)

        while not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()


        self.scheduler.reset()

        self.connect_agents_and_goals()
        self.inform_goals_of_costs_to_other_goals()

    def get_normalized_skill_vectors_for_all_agents(self):
        all_skills_normalized = []
        for agent in self.agents:
            agent_skills_normalized = [0]*self.num_skills
            for skill in agent.skills:
                agent_skills_normalized[skill] = 1
            all_skills_normalized.extend(agent_skills_normalized)
        return all_skills_normalized
    
    def get_normalized_skill_vectors_for_all_goals(self):
        all_skills_normalized = []
        for goal in self.goals:
            goal_skills_normalized = [0]*self.num_skills
            for skill in goal.required_skills:
                goal_skills_normalized[skill] = 1
            all_skills_normalized.extend(goal_skills_normalized)
        return all_skills_normalized        

    def find_numerical_solution(self):
        if self.solve_type == "optimal":
            self.replan_on_goal_claim = False
            self.full_solution = self.scheduler.find_optimal_solution()
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
            self.full_solution = self.scheduler.find_linalg_solution()

    def step_environment(self, fast=False):
        for agent, goal_list in self.full_solution.items():
            for i, goal in enumerate(goal_list):
                if goal.claimed:
                    continue
                self.scheduler.goal_assignments[agent] = goal
                break # OH MY GOD DO NOT REMOVE THIS BREAK
        self.update(fast=fast)

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
        observation_vector = {
            "map_elevations": self.grid_map.downscaled_data.flatten(),
            "goal_positions": np.array([[goal.position[0],goal.position[1]] for goal in self.goals]).flatten(),
            "goal_required_skills": np.array([[(1 if skill in goal.required_skills else 0) for skill in range(self.num_skills)] 
            for goal in self.goals]).flatten(),
            "agent_positions": np.array([[agent.position[0],agent.position[1]] for agent in self.agents]).flatten(),
            "agent_skills": np.array([[(1 if skill in agent.skills else 0) for skill in range(self.num_skills)] 
            for agent in self.agents]).flatten(),
            "claimed_goals": len(self.goals) - len(self.scheduler.unclaimed_goals)
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
    
    def update_agents_last_visited_goals(self):
        for agent in self.agents:
            for goal in self.goals:
                if agent.position == goal.position:
                    self.scheduler.last_visited_goals[agent] = goal


    def update(self, fast=False):
        agent_positions_start = [agent.position for agent in self.agents]
        for agent,goal in self.scheduler.goal_assignments.items():
            # if agent.position == agent.initial_position or (self.scheduler.last_visited_goals[agent] and not self.replan_on_goal_claim):
            #     path,cost = agent.paths_and_costs_to_goals[goal]
            #     self.grid_map.assign_path_to_agent(agent=agent, path=path)
            # elif self.scheduler.last_visited_goals[agent] and agent.position == self.scheduler.last_visited_goals[agent].position:
            #     path,cost = self.scheduler.last_visited_goals[agent].paths_and_costs_to_other_goals[goal]
            #     self.grid_map.assign_path_to_agent(agent=agent, path=path)
            # else:
            self.grid_map.assign_shortest_path_for_goal_to_agent(agent=agent, goal=goal)
            path = self.grid_map.paths[agent]
            cost = self.grid_map.calculate_path_cost(path)
            # if fast:
            #     agent.move_to_position(goal.position)
            #     agent.accumulated_cost += cost
            #     agent.steps_moved += len(path)
            # else:
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


    def _replan(self):
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
    
    def get_manhattan_distance_to_closest_unclaimed_goal(self, agent: Agent):
        distances = []
        for goal in self.scheduler.unclaimed_goals:
            distances.append(abs(agent.position[0] - goal.position[0]) + abs(agent.position[1] - goal.position[1]))
        return min(distances)
    
    def get_manhattan_distances_for_all_agents_to_all_goals(self):
        distances = []
        for agent in self.agents:
            agent_distances = []
            for goal in self.goals:
                agent_distances.append(abs(agent.position[0] - goal.position[0]) + abs(agent.position[1] - goal.position[1]))
            distances.extend(agent_distances)
        return distances