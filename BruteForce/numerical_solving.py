import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict

import numpy as np

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualiser_class import Visualizer
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal        

def run_sim(env: Environment, speed, cell_size=30):
    
    chance_of_adding_random_goal: float = 0.1
    chance_of_adding_random_obstacle: float = 0
    cell_size: int = cell_size

    max_goals = env.num_goals

    print("Initializing environment...")

    
    vis: Visualizer = Visualizer(env, cell_size=cell_size)

    cheapest_solution = env.find_numerical_solution()

    done: bool = False
    current_assignments: Dict[Goal, Agent] = {}

    while not done and cheapest_solution:
        if cheapest_solution is None:
            continue
        for agent, goal_list in cheapest_solution.items():
            for i, goal in enumerate(goal_list):
                if goal.claimed:
                    continue
                current_assignments[goal_list[i]] = agent
                vis.assignments[agent] = goal_list[i]
                env.scheduler.assign_goal_to_agent(agent=agent, goal=goal_list[i])
                env.planner.assign_shortest_path_for_goal_to_agent(agent=agent, goal=goal_list[i])
                break
            agent.apply_action(env.planner.get_move_to_reach_next_position(agent))

        env.update()

        # random obstacle
        if np.random.rand() < chance_of_adding_random_obstacle:
            if env.add_random_obstacle_close_to_position(position=np.random.choice(env.agents).position):
                cheapest_solution = env.find_numerical_solution()
        
        # random goal
        if np.random.rand() < chance_of_adding_random_goal:
            if len(env.scheduler.unclaimed_goals) < max_goals-1:
                location = env.grid_map.random_valid_location_close_to_position(position=np.random.choice(env.agents).position, max_distance=2000)
                env.add_random_goal(location=location)
                cheapest_solution = env.find_numerical_solution()

        vis.run_step(speed=speed)
        done = env.scheduler.all_goals_claimed()
        if done:
            print("All goals claimed!")
            break
        
while True:

    num_goals: int = 5
    num_agents: int = 4
    size: int = 200
    num_obstacles: int = 0
    num_skills: int = 3
    cell_size: int = int(1000/size)
    speed: int = 60

    print("Number of agents: ", num_agents)
    print("Number of goals: ", num_goals)
    print("Number of skills: ", num_skills)
    print("Map size (n x n), n = ", size)

    env: Environment = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, num_skills=num_skills, use_geo_data=True)

    run_sim(env=env, speed=speed, cell_size=cell_size)
    env.reset()