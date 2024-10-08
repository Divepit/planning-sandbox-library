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
    def __init__(self, size, num_agents, num_goals, num_skills, use_geo_data=False, solve_type="fast"):
        self.size = size
        self.map_diagonal = np.sqrt(2 * (size ** 2))
        self.solve_type = solve_type
        self.num_skills = num_skills
        self.replan_on_goal_claim = True

        self.goals: List[Goal] = []
        self.agents: List[Agent] = []
        self.scheduler = Scheduler(agents=self.agents, goals=self.goals)
        self.cheapest_solution = None

        self._initial_num_agents = num_agents
        self._initial_num_goals = num_goals
        self._initial_num_skills = num_skills
        
        self.grid_map = GridMap(self.size, use_geo_data=use_geo_data)
        self._starting_position = self.grid_map.random_valid_position()

        
        self._log_environment_info()
        
        self._init()

    def _init(self):
        self._initialize_goals()
        self._initialize_agents()

        while not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()

        self.connect_agents_and_goals()
        self.inform_goals_of_costs_to_other_goals()

    def _log_environment_info(self):
        logging.info(f"=== Environment settings ===")
        logging.info(f"Num agents: {len(self.agents)}")
        logging.info(f"Num goals: {len(self.goals)}")
        logging.info(f"Num skills: {self.num_skills}")
        logging.info(f"Map size: {self.size}")
        logging.info(f"=== === === === === === ===")
        
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
            goal.soft_reset()
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

    def reset(self, randomize_goal_positions=True, randomize_agent_positions=True, randomize_skills=True):
        self.num_skills = self._initial_num_skills


        self.grid_map.reset()
        self._starting_position = self.grid_map.random_valid_position()
        
        
        for goal in self.goals:
            new_position = None
            if randomize_goal_positions:
                new_position = self.grid_map.random_valid_position()
            goal.reset(position=new_position, randomize_skills=randomize_skills)

        for agent in self.agents:
            new_position = None
            if randomize_agent_positions:
                new_position = self.grid_map.random_valid_position()
            agent.reset(position=new_position, randomize_skills=randomize_skills)
        if randomize_skills:
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

    def add_random_goal(self, location=None):
        if location is not None:
            goal = self._add_goal(location)
        else:
            goal = self._add_goal(self.grid_map.random_valid_position())
        if self.num_skills == 1:
            if goal.required_skills == []:
                goal.required_skills.append(0)
            return
        amount_of_skills = np.random.randint(1, min(3,self.num_skills+1))
        skills = []
        for _ in range(amount_of_skills):
            skill = np.random.randint(0, self.num_skills)
            while skill in skills:
                skill = np.random.randint(0, self.num_skills)
            skills.append(skill)
        goal.required_skills = skills
        self.scheduler.reset()
        

    def find_numerical_solution(self):
        if self.solve_type == "optimal":
            self.replan_on_goal_claim = False
            self.cheapest_solution = self.scheduler.find_optimal_solution()
        elif self.solve_type == "fast":
            self.replan_on_goal_claim = True
            self.cheapest_solution = self.scheduler.find_fast_solution()
        elif self.solve_type == "linalg":
            self.replan_on_goal_claim = False
            self.cheapest_solution = self.scheduler.find_linalg_solution()

    def step_environment(self):
        for agent, goal_list in self.cheapest_solution.items():
            for i, goal in enumerate(goal_list):
                if goal.claimed:
                    continue
                self.assign_goal_to_agent(agent=agent, goal=goal_list[i])
                break
        self.update()

    def solve_cheapest_solution(self, max_iterations = None):
        
        self.reset(randomize_skills=False, randomize_goal_positions=False, randomize_agent_positions=False)
        
        if max_iterations is None:
            max_iterations = self.size**2
        
        solving_bench: Benchmark = Benchmark("solve_cheapest_solution", start_now=True, silent=True)
        for _ in range(max_iterations):
            self.step_environment()
            if self.scheduler.all_goals_claimed():
                break
        solving_bench.stop()

        total_steps_moved = sum([agent.steps_moved for agent in self.agents])
        total_steps_waited = sum([agent.steps_waited for agent in self.agents])
        total_cost = sum([agent.accumulated_cost for agent in self.agents])
        solve_time = solving_bench.elapsed_time
        claimed_goals = len(self.goals) - len(self.scheduler.unclaimed_goals)

        return total_steps_moved, total_steps_waited, total_cost, solve_time, claimed_goals
    
    def get_observation_vector(self):
        observation_vector = {
            "goal_positions": [goal.position for goal in self.goals],
            "goal_required_skills": [[(1 if skill in goal.required_skills else 0) for skill in range(self.num_skills)] 
            for goal in self.goals],
            "agent_positions": [agent.position for agent in self.agents],
            "agent_skills": [[(1 if skill in agent.skills else 0) for skill in range(self.num_skills)] 
            for agent in self.agents]
        }
        return observation_vector
    
    def get_action_vector(self):
        action_vector = []
        for agent in self.agents:
            if agent in self.cheapest_solution:
                goal_list = self.cheapest_solution[agent]
                # Pad with -1 (or use `num_goals` as a dummy) if fewer than max_goals_per_agent
                padded_goal_list = [self.goals.index(goal) for goal in goal_list] + [-1] * (len(self.goals) - len(goal_list))
                action_vector.append(padded_goal_list)
            else:
                # No goals, append `-1` for all slots
                action_vector.append([-1] * len(self.goals))

        flattened_action_vector = [goal for sublist in action_vector for goal in sublist]
        return flattened_action_vector
    
    def set_cheapest_solution_from_action_vector(self, action_vector):
        for i, agent in enumerate(self.agents):
            if i < len(action_vector):
                for goal_index in action_vector[i]:
                    self.cheapest_solution[agent] = self.goals[goal_index]

    def update(self):
        for agent in self.agents:
            if agent in self.scheduler.goal_assignments:
                goal = self.scheduler.goal_assignments[agent]
                self.grid_map.assign_shortest_path_for_goal_to_agent(agent=agent, goal=goal)
                action, action_cost = self.grid_map.get_move_and_cost_to_reach_next_position(agent)
                agent.apply_action(action, action_cost)

        claimed_a_goal = self.scheduler.update_goal_statuses()
        if (claimed_a_goal and self.replan_on_goal_claim) or self.new_goal_added:
            self._replan()

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
    
    def assign_goal_to_agent(self, agent: Agent, goal: Goal):
        self.scheduler.goal_assignments[agent] = goal