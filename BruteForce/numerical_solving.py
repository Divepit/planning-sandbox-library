import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict


# import cProfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualiser_class import Visualizer
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal        
from planning_sandbox.benchmark_class import Benchmark

from copy import deepcopy

global opened_3d_map
opened_3d_map = False

def run_sim(env: Environment, speed, cell_size=30, visualize=True):
    global opened_3d_map
    setup_bench = Benchmark('sim_setup',start_now=True, silent=True)
    chance_of_adding_random_goal: float = 0
    chance_of_adding_random_obstacle: float = 0
    cell_size: int = cell_size


    max_goals = len(env.goals)
    
    vis: Visualizer = Visualizer(env, cell_size=cell_size, visualize=visualize, show_3d_elevation=False)
    
    if not opened_3d_map:
        vis.display_3d_elevation()
        opened_3d_map = True

    cheapest_solution = env.find_numerical_solution()

    done: bool = False
    current_assignments: Dict[Goal, Agent] = {}
    steps = 0
    total_cost = 0
    setup_bench.stop()
    while not done:

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
            action, action_cost = env.planner.get_move_and_cost_to_reach_next_position(agent)
            agent.apply_action(action)
            total_cost += action_cost
            steps += 1

        replan_required = env.update() > 0 and env.solve_type == 'fast'

        # random obstacle
        if np.random.rand() < chance_of_adding_random_obstacle:
            random_agent: Agent = np.random.choice(env.agents)
            env.add_random_obstacle_close_to_position(position=random_agent.position)
            replan_required = True
        
        # random goal
        new_goal = False
        if np.random.rand() < chance_of_adding_random_goal:
            if len(env.scheduler.unclaimed_goals) < max_goals-1:
                random_agent: Agent = np.random.choice(env.agents)
                location = env.grid_map.random_valid_location_close_to_position(position=random_agent.position, max_distance=20)
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
            bench_step.stop()
            break
        bench_step.stop()
    return steps, total_cost

def clone_environments(env, num_clones):
    seed_envs = []
    print("Clone environments...")
    for _ in range(num_clones):
        print(f"Progress: {len(seed_envs)+1}/{num_clones}", end='\r')
        env.reset()
        seed_envs.append(deepcopy(env))
    return seed_envs

# Some plotting functions written by chatGPT (https://chatgpt.com/share/66f558b4-9844-8008-8442-3e9021b9bbbd)
def main(iterations = np.inf):
    
    visualize = True
    iterations = 3
    num_goals: int = 5
    num_agents: int = 3
    size: int = 100
    num_obstacles: int = 0
    num_skills: int = 2
    cell_size: int = int(1000/size)
    velocity = 200 #m/s (max 200)
    speed = velocity*1/5
    solve_types = ['fast','optimal']
    
    
    print("Setting up environment...")
    env = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_obstacles=num_obstacles, num_skills=num_skills, use_geo_data=True)
    
    seed_envs = clone_environments(env, iterations)
    
    results = {}
    for solve_type in solve_types:
        print(f"\nRunning simulations for solve type: {solve_type}")
        runtimes = []
        all_steps = []
        all_costs = []
        for env in seed_envs:
            if len(runtimes) > 0:
                time_left = round(np.mean(runtimes) * (iterations - len(runtimes)), 2)
            else:
                time_left = '...'
            print(f"Progress: {len(runtimes)+1}/{iterations} | Expected time left: {time_left} seconds", end='\r')
            bench = Benchmark('run_sim', start_now=True, silent=True)
            env_copy: Environment = deepcopy(env)
            env_copy.solve_type = solve_type
            steps, total_cost = run_sim(env=env_copy, speed=speed, cell_size=cell_size, visualize=visualize)
            runtime = bench.stop()
            runtimes.append(runtime)
            all_steps.append(steps)
            all_costs.append(total_cost)

        results[solve_type] = {
            'runtimes': np.array(runtimes),
            'steps': np.array(all_steps),
            'costs': np.array(all_costs),
        }

    sns.set_theme(style='whitegrid')
    metrics = ['runtimes', 'steps', 'costs']
    metric_names = ['Runtime (s)', 'Steps', 'Total Cost']
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 6))

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx] if num_metrics > 1 else axes
        data_means = []
        data_stds = []
        for solve_type in solve_types:
            data = results[solve_type][metric]
            data_means.append(np.mean(data))
            data_stds.append(np.std(data))

        # Create bar chart with error bars
        x_pos = np.arange(len(solve_types))
        colors = sns.color_palette('pastel')[0:len(solve_types)]
        bars = ax.bar(
            x_pos,
            data_means,
            yerr=data_stds,
            align='center',
            alpha=0.8,
            capsize=5,
            color=colors,
            edgecolor='black',
            linewidth=1.2,
            error_kw=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(solve_types, fontsize=12)
        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_title(f'Comparison of {metric_name}', fontsize=16)
        ax.tick_params(axis='y', labelsize=12)

        for bar, mean in zip(bars, data_means):
            height = bar.get_height()
            ax.annotate(
                f'{mean:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 10),  # Offset text by 10 points above bar
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=12
            )

    additional_info = 'Cost Function (at each step) = slope + 0.2 * distance_per_step'

    if results['optimal']['costs'].mean() < results['fast']['costs'].mean():
        word = 'cheaper'
        percentage = (results['fast']['costs'].mean() - results['optimal']['costs'].mean()) / results['fast']['costs'].mean() * 100
    else:
        word = 'more expensive'
        percentage = (results['optimal']['costs'].mean() - results['fast']['costs'].mean()) / results['optimal']['costs'].mean() * 100
    main_result = 'The optimal solver was {:.2f}% {} than the fast solver in terms of total cost.'.format(percentage, word)

    fig.suptitle(
        f'\nSimulation Results ({iterations} iterations for each solver)\n'
        f'\nAgents: {num_agents}, Goals: {num_goals}, Skills: {num_skills}, Grid Size: {size}x{size}\n'
        f'\n{additional_info}\n'
        f'\n{main_result}\n',
        fontsize=18,
        y=1.02
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show(block=False)

if __name__ == "__main__":
    # cProfile.run('main()', sort='cumtime')
    main(iterations=10)