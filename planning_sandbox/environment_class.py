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

        self._connect_agents_and_goals()
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

    def _connect_agents_and_goals(self):
        inform_goals_of_agents_bench = Benchmark("inform_goals_of_agents", start_now=True)
        for goal in self.scheduler.unclaimed_goals:
            goal.soft_reset()
            for agent in self.agents:
                if any(skill in agent.skills for skill in goal.required_skills):
                    goal.add_agent_which_has_required_skills(agent)
        inform_goals_of_agents_bench.stop()

        inform_agents_of_costs_bench = Benchmark("inform_agents_of_costs_to_goals", start_now=True)
        for goal in self.scheduler.unclaimed_goals:
            for agent in goal.agents_which_have_required_skills:
                    path = self.planner.generate_shortest_path_for_agent(agent, goal)
                    cost = self._calculate_path_cost(path)
                    agent.add_cost_to_reach_goal(goal, cost)
        inform_agents_of_costs_bench.stop()

        for goal in self.scheduler.unclaimed_goals:
            goal.generate_agent_combinations_which_solve_goal()

    def _inform_goals_of_costs_to_other_goals(self):
        inform_goals_of_costs_bench = Benchmark("inform_goals_of_costs_to_other_goals", start_now=True)
        for goal in self.scheduler.unclaimed_goals:
            for other_goal in self.goals:
                if goal == other_goal:
                    continue

                if not other_goal in goal.cost_to_reach_other_goals:
                    path = self.grid_map.shortest_path(goal.position, other_goal.position)
                    cost = self._calculate_path_cost(path)
                    goal.add_cost_to_reach_other_goal(other_goal, cost)

                if not goal in other_goal.cost_to_reach_other_goals:
                    path = self.grid_map.shortest_path(other_goal.position, goal.position)
                    cost = self._calculate_path_cost(path)
                    other_goal.add_cost_to_reach_other_goal(goal, cost)


        inform_goals_of_costs_bench.stop()        

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

        self._connect_agents_and_goals()
        self._inform_goals_of_costs_to_other_goals()


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
        cheapest_combinations = {} # goal: (combination, cost)
        cheapest_solution = {} # agent: [goal]
        unaccounted_for_goals = set(self.scheduler.unclaimed_goals)
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
                print("No solution found")
                print("Cheapest combinations: ", cheapest_combinations)
                print("Cheapest solution: ", cheapest_solution)
                break

            for agent in cheapest_combination:
                cheapest_solution[agent] = [cheapest_goal]

            unaccounted_for_goals.remove(cheapest_goal)
            cheapest_combinations.clear()

        return cheapest_solution


    def update(self):
        return self.scheduler.update_goal_statuses()
    
    def get_normalized_grid(self):

        grid = np.zeros((self.size, self.size))
        for obstacle in self.obstacles:
            grid[obstacle] = 1
        
        for agent in self.agents:
            grid[agent.position] = 0.5

        for goal in self.goals:
            grid[goal.position] = 0.75

        return grid