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

        print( "=== Environment settings ===")
        print("Num agents: ", num_agents)
        print("Num goals: ", num_goals)
        print("Num skills: ", num_skills)
        print("Map size: ", size)
        print("=== === === === === === ===")


        self.grid_map = GridMap(self.size, num_obstacles, use_geo_data=use_geo_data)
        self.obstacles = self.grid_map.obstacles
        self.starting_position = self.grid_map.random_valid_position()

        self.agents = []
        self._initialize_agents()


        self.goals = []
        self._initialize_goals()

        self.normalized_skill_map = {i: i / self.num_skills for i in range(self.num_skills)}
        
        while not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()

        self.controller = Controller(self.grid_map)
        self.planner = Planner(self.agents, self.grid_map)
        self.scheduler = Scheduler(self.agents, self.goals)
        
    def _initialize_agents(self):
        if self.starting_position is not None:
            start_pos = self.starting_position
        else:
            start_pos = self.grid_map.random_valid_position()
        for _ in range(self.num_agents):
            agent = Agent(start_pos)
            # self.grid_map.add_occupied_position(agent.position)
            self.agents.append(agent)

    def _initialize_goals(self):
        for _ in range(self.num_goals):
            random_position = self.grid_map.random_valid_position()
            while random_position in [agent.position for agent in self.agents]:
                random_position = self.grid_map.random_valid_position()
            goal = Goal(random_position)
            # self.grid_map.add_occupied_position(goal.position)
            self.goals.append(goal)

    def reset(self):

        self.grid_map.reset()

        for agent in self.agents:
            agent.reset(self.grid_map.random_valid_position())
            # self.grid_map.add_occupied_position(agent.position)

        for goal in self.goals:
            goal.reset(self.grid_map.random_valid_position())
            # self.grid_map.add_occupied_position(goal.position)


        self.planner.reset()
        self.scheduler.reset()
        self.controller.reset()

        self._initialize_skills()

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

    def _get_all_agent_to_goal_plans(self):
        agent_goal_plans: Dict[Tuple[Agent, Goal], List[int]]  = {}
        agent_goal_costs: Dict[Tuple[Agent, Goal], int]  = {}
        allowed_goal_agent_pairs: List[Tuple[Goal, Agent]] = []
        allowed_goal_agent_pairs = self.scheduler.get_allowed_goal_agent_pairs(goals=self.scheduler.unclaimed_goals, agents=self.agents)
        for agent, goal in allowed_goal_agent_pairs:
            path = self.planner.generate_shortest_path_for_agent(agent=agent,goal=goal)
            agent_goal_plans[(agent,goal)] = path
            agent_goal_costs[(agent,goal)] = self._calculate_path_cost(path)
        return agent_goal_plans, agent_goal_costs
    

    def _get_all_goal_to_goal_plans(self):
        goal_goal_plans: Dict[Tuple[Goal, Goal], List[int]] = {}
        goal_goal_costs: Dict[Tuple[Goal, Goal], int] = {}
        goal_goal_combinations = permutations(self.scheduler.unclaimed_goals, 2)
        for combination in goal_goal_combinations:
            path = self.grid_map.shortest_path(start=combination[0].position, goal=combination[1].position)
            goal_goal_plans[combination] = path
            goal_goal_costs[combination] = self._calculate_path_cost(path)
        return goal_goal_plans, goal_goal_costs
    
    def _get_all_goal_solutions(self, agent_goal_plans):
        goal_solutions: Dict[Goal,List[List[Agent]]] = {}
        for goal in self.scheduler.unclaimed_goals:
            goal_solutions[goal] = []
            for r in range(1,self.num_agents+1):
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
                    if self.scheduler.agent_combination_has_required_skills_for_goal(agents=combination, goal=goal):
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

        agent_goal_bench = Benchmark("agent_to_goal", start_now=True)
        print("Calculating paths from all agents to all goals...")
        agent_goal_plans, agent_goal_costs = self._get_all_agent_to_goal_plans()
        agent_goal_bench.stop()

        goal_goal_bench = Benchmark("goal_to_goal", start_now=True)
        print("Calculating paths between all goals...")
        goal_goal_plans, goal_goal_costs = self._get_all_goal_to_goal_plans()
        goal_goal_bench.stop()

        goal_solutions_bench = Benchmark("goal_solutions", start_now=True)
        print("Figuring out which combinations of agents solve each goal...")
        goal_solutions = self._get_all_goal_solutions(agent_goal_plans=agent_goal_plans)
        goal_solutions_bench.stop()


        possible_solutions_bench = Benchmark("possible_solutions", start_now=True)
        print("Combining all single-goal solving agent-combinations to find all possible solutions to the full problem...")
        possible_solutions = self._get_all_possible_solutions(goal_solutions=goal_solutions)
        possible_solutions_bench.stop()

        cheapest_solution_bench = Benchmark("cheapest_solution", start_now=True)
        print("Finding cheapest solution...")
        cheapest_solution = self._get_cheapest_solution(possible_solutions=possible_solutions, agent_goal_costs=agent_goal_costs, goal_goal_costs=goal_goal_costs)    
        cheapest_solution_bench.stop()

        print("===================== Found solution =====================")

        return cheapest_solution, agent_goal_bench, goal_goal_bench, goal_solutions_bench, possible_solutions_bench, cheapest_solution_bench 
    
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