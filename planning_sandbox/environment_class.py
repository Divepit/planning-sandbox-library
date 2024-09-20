from typing import List, Dict, Tuple

import networkx as nx
import copy

from itertools import combinations, permutations, product
from planning_sandbox.grid_map_class import GridMap
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal
from planning_sandbox.controller_class import Controller
from planning_sandbox.scheduler_class import Scheduler
from planning_sandbox.planner_class import Planner
from planning_sandbox.benchmark_class import Benchmark

import numpy as np

class Environment:
    def __init__(self, size, num_agents, num_goals, num_obstacles, num_skills, use_geo_data=False):
        self.size = size

        self.num_agents = num_agents
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.num_skills = num_skills

        self.initial_num_agents = num_agents
        self.initial_num_goals = num_goals
        self.initial_num_obstacles = num_obstacles
        self.initial_num_skills = num_skills

        print( "=== Environment settings ===")
        print("Num agents: ", num_agents)
        print("Num goals: ", num_goals)
        print("Num skills: ", num_skills)
        print("Map size: ", size)
        print("=== === === === === === ===")


        self.grid_map = GridMap(self.size, num_obstacles, use_geo_data=use_geo_data)
        self.obstacles = self.grid_map.obstacles
        self.starting_position = self.grid_map.random_valid_position()
        
        
        self.goals = []
        self._initialize_goals()

        self.agents: List[Agent] = []
        self._initialize_agents()




        self.normalized_skill_map = {i: i / self.num_skills for i in range(self.num_skills)}
        
        while not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()


        self.controller = Controller(self.grid_map)
        self.planner = Planner(self.agents, self.grid_map)
        self.scheduler = Scheduler(self.agents, self.goals)

        self._inform_agents_of_costs_to_goals()
        self._inform_goals_of_agents()
        self._inform_goals_of_costs_to_other_goals()
        
    def _initialize_agents(self):
        if self.starting_position is not None:
            start_pos = self.starting_position
        else:
            start_pos = self.grid_map.random_valid_position()
        for _ in range(self.num_agents):
            agent = Agent(start_pos)
            self.agents.append(agent)

    def _initialize_goals(self):
        for _ in range(self.num_goals):
            random_position = self.grid_map.random_valid_position()
            goal = Goal(random_position)
            self.goals.append(goal)

    def _inform_goals_of_agents(self):
        inform_goals_of_agents_bench = Benchmark("inform_goals_of_agents", start_now=True)
        for goal in self.scheduler.unclaimed_goals:
            for agent in self.agents:
                if any(skill in agent.skills for skill in goal.required_skills) and agent not in goal.agents_which_have_required_skills:
                    goal.add_agent_which_has_required_skills(agent)
        inform_goals_of_agents_bench.stop()

    def _inform_goals_of_costs_to_other_goals(self):
        inform_goals_of_costs_bench = Benchmark("inform_goals_of_costs", start_now=True)
        for goal in self.scheduler.unclaimed_goals:
            for other_goal in self.goals:
                if goal == other_goal or other_goal in goal.cost_to_reach_other_goals:
                    continue
                path = self.grid_map.shortest_path(goal.position, other_goal.position)
                cost = self._calculate_path_cost(path)
                goal.add_cost_to_reach_other_goal(other_goal, cost)
        inform_goals_of_costs_bench.stop()

    def _inform_agents_of_costs_to_goals(self):
        inform_agents_of_costs_bench = Benchmark("inform_agents_of_costs", start_now=True)
        for agent in self.agents:
            for goal in self.goals:
                if goal not in agent.costs_to_reach_goals and any(skill in agent.skills for skill in goal.required_skills):
                    path = self.grid_map.shortest_path(agent.position, goal.position)
                    cost = self._calculate_path_cost(path)
                    agent.add_cost_to_reach_goal(goal, cost)
        inform_agents_of_costs_bench.stop()

    def reset(self):
        self.num_agents = self.initial_num_agents
        self.num_goals = self.initial_num_goals
        self.num_obstacles = self.initial_num_obstacles
        self.num_skills = self.initial_num_skills


        self.grid_map.reset()
        self.obstacles = self.grid_map.obstacles
        self.starting_position = self.grid_map.random_valid_position()
        
        
        self.goals.clear()
        self._initialize_goals()

        self.agents.clear()
        self._initialize_agents()

        self.normalized_skill_map = {i: i / self.num_skills for i in range(self.num_skills)}
        
        while not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()


        self.controller.reset()
        self.planner.reset()
        self.scheduler.reset()

        self._inform_agents_of_costs_to_goals()
        self._inform_goals_of_agents()
        self._inform_goals_of_costs_to_other_goals()

        # self.grid_map.reset()

        # for agent in self.agents:
        #     agent.reset(self.grid_map.random_valid_position())
        #     # self.grid_map.add_occupied_position(agent.position)

        # for goal in self.goals:
        #     goal.reset(self.grid_map.random_valid_position())
        #     # self.grid_map.add_occupied_position(goal.position)


        # self.planner.reset()
        # self.scheduler.reset()
        # self.controller.reset()

        # self._initialize_skills()
        # self._inform_agents_of_costs_to_goals()
        # self._inform_goals_of_agents()
        # self._inform_goals_of_costs_to_other_goals()

    def _initialize_skills(self):
        # assign 1 or 2 skills to each goal
        # assign 1 to num_skills skills to each agent
        # ensure that no goal has the same skill twice and no agent has the same skill twice
        if self.num_skills == 1:
            for goal in self.goals:
                if goal.required_skills == []:
                    goal.add_skill(0)
            for agent in self.agents:
                if agent.skills == []:
                    agent.add_skill(0)
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

    def _calculate_path_cost(self, path):
        return (nx.path_weight(self.grid_map.graph, path, weight="weight"))*(len(path)/self.size)    

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
            goal = Goal(location)
        else:
            goal = Goal(self.grid_map.random_valid_position())
        self.goals.append(goal)
        self.num_goals += 1
        if self.num_skills == 1:
            if goal.required_skills == []:
                goal.add_skill(0)
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
        self.planner.reset()

    def add_random_obstacle_close_to_position(self,position):
        pos = self.grid_map.random_valid_location_close_to_position(position, max_distance=1)
        for agent in self.agents:
            if pos == agent.position:
                return False
        for goal in self.goals:
            if pos == goal.position:
                return False
        grid_map_copy = copy.deepcopy(self.grid_map)
        grid_map_copy.add_obstacle(pos)
        if grid_map_copy.check_if_connected():
            self.grid_map.add_obstacle(pos)
            self.planner.reset()
            self.controller.reset()
            self.scheduler.reset()
            return True
        return False
    

    def find_numerical_solution(self):
        current_solution_cost = 0
        cheapest_solution = {}
        ordered_goals = []
        
        i = 0
        while i < len(self.scheduler.unclaimed_goals):
            print(i)
            for agent,assignment in self.scheduler.goal_assignments:
                self.cheapest_solution[agent] = assignment[0]
            
            if all([agent in cheapest_solution for agent in self.agents]):
                break
            
            cheapest_combinations = {}

            for goal in self.scheduler.unclaimed_goals:
                goals_cheapest_combinations_sorted = sorted(goal.agent_combinations_which_solve_goal.items(), key=lambda x: x[1])
                if i >= len(goals_cheapest_combinations_sorted):
                    continue
                next_cheapest_combination = goals_cheapest_combinations_sorted[i]
                cheapest_combinations[goal]=(next_cheapest_combination)

            cheapest_combinations_sorted = sorted(cheapest_combinations.items(), key=lambda x: x[1][1])
            

            for goal, (combination,cost) in cheapest_combinations_sorted:
                new_goal_added = False
                if goal in ordered_goals:
                    continue
                if combination is not None:
                    if any([agent in cheapest_solution or agent in self.scheduler.goal_assignments for agent in combination]):
                        continue
                    for agent in combination:
                        cheapest_solution[agent] = [goal]
                    ordered_goals.append(goal)
                    new_goal_added = True
            i += 1

                
        # cheapest_solution = {}
        # for goal in self.scheduler.unclaimed_goals:
        #     for combination,cost in goal.agent_combinations_which_solve_goal.items():
        #         combination_valid = True
        #         for agent in combination:
        #             if self.scheduler.goal_assignments.get(agent) is not None or agent in cheapest_solution:
        #                 combination_valid = False
        #                 break
        #         if combination_valid:
        #             for agent in combination:
        #                 if agent not in cheapest_solution:
        #                     cheapest_solution[agent] = []
        #                 cheapest_solution[agent].append(goal)
        #         agents_without_goal = False
        #         for agent in self.agents:
        #             if self.scheduler.goal_assignments.get(agent) is None and agent not in cheapest_solution:
        #                 agents_without_goal = True
        #         if not agents_without_goal:
        #             break
                    

        # cheapest_combinations = {goal:(goal.cheapest_combination[0],goal.cheapest_combination[1]) for goal in self.scheduler.unclaimed_goals}
        # cheapest_combinations_sorted = sorted(cheapest_combinations.items(), key=lambda x: x[1][1])


        # for goal, (combination,cost) in cheapest_combinations_sorted:
        #     if combination is not None:
        #         for agent in combination:
        #             if agent not in cheapest_solution:
        #                 cheapest_solution[agent] = []
        #             cheapest_solution[agent].append(goal)
        
        return cheapest_solution

    def update(self):
        return self.scheduler.update_goal_statuses()
    
    def get_normalized_grid(self):
        # use the graph
        # empty positions = 0
        # obstacle positions = 1
        # agent positions = 0.5
        # goal positions = 0.75

        grid = np.zeros((self.size, self.size))
        for obstacle in self.obstacles:
            grid[obstacle] = 1
        
        for agent in self.agents:
            grid[agent.position] = 0.5

        for goal in self.goals:
            grid[goal.position] = 0.75

        return grid