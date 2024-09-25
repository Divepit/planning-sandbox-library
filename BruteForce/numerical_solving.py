import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict

import numpy as np
import cProfile

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualiser_class import Visualizer
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal        
from planning_sandbox.benchmark_class import Benchmark

def run_sim(env: Environment, speed, cell_size=30):
    setup_bench = Benchmark('setup',start_now=True)
    chance_of_adding_random_goal: float = 0
    chance_of_adding_random_obstacle: float = 0
    cell_size: int = cell_size

    max_goals = env.num_goals

    print("Initializing environment...")

    
    vis: Visualizer = Visualizer(env, cell_size=cell_size)

    cheapest_solution = env.find_numerical_solution()

    done: bool = False
    current_assignments: Dict[Goal, Agent] = {}
    steps = 0
    setup_bench.stop()
    while not done:
        # print(f"Step {steps}")
        bench_step = Benchmark('step',start_now=True, silent=True)
        replan_required = False

        if cheapest_solution is None or not cheapest_solution:
            print("No solution found")
            bench_step.stop()
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
            steps += 1

        replan_required = env.update() > 0

        # random obstacle
        if np.random.rand() < chance_of_adding_random_obstacle:
            env.add_random_obstacle_close_to_position(position=np.random.choice(env.agents).position)
            replan_required = True
        
        # random goal
        new_goal = False
        if np.random.rand() < chance_of_adding_random_goal:
            if len(env.scheduler.unclaimed_goals) < max_goals-1:
                location = env.grid_map.random_valid_location_close_to_position(position=np.random.choice(env.agents).position, max_distance=20)
                env.add_random_goal(location=location)
                new_goal = True
                replan_required = True

        if replan_required:
            bench_replan = Benchmark('replanning',start_now=True, silent=True)
            env._connect_agents_and_goals()
            if new_goal:
                env._inform_goals_of_costs_to_other_goals()
            cheapest_solution = env.find_numerical_solution()
            bench_replan.stop()
        

        vis.run_step(speed=speed)
        done = env.scheduler.all_goals_claimed()
        if done:
            vis.close()
            print("All goals claimed!")
            bench_step.stop()
            break
        bench_step.stop()
        if replan_required:
            print(f"Replanning took {bench_step.elapsed_time:.2f} seconds")
    return steps

def main(iterations = np.inf):
    setup_bench = Benchmark('setup',start_now=True)
    num_goals: int = 15
    num_agents: int = 3
    size: int = 100
    num_obstacles: int = 0
    num_skills: int = 1
    cell_size: int = int(1000/size)
    velocity = 200 #m/s (max 200)
    speed = velocity*1/5

    env: Environment = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, num_skills=num_skills, use_geo_data=True, solve_type='optimal')

    runtimes = []
    all_steps = []
    i = 0

    setup_bench.stop()

    while i < iterations:
        i += 1
        bench = Benchmark('numerical_solving',start_now=True)
        steps = run_sim(env=env, speed=speed, cell_size=cell_size)
        runtimes.append(bench.stop())
        all_steps.append(steps)
        env.reset()
    print(f"Average runtime: {np.mean(runtimes)}")
    print(f"Average steps: {np.mean(all_steps)}")

if __name__ == "__main__":
    # cProfile.run('main()', sort='cumtime')
    main(iterations=10)