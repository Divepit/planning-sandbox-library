# This benchmarking script has been created using ChatGPT in this conversation: https://chatgpt.com/share/66e938f6-3e28-8008-a465-5d041344fc47

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import Dict
from planning_sandbox.environment_class import Environment
from planning_sandbox.visualiser_class import Visualizer
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal


def run_sim(env: Environment, speed, cell_size=30):
    # Your existing run_sim function remains unchanged
    chance_of_adding_random_goal: float = 0
    chance_of_adding_random_obstacle: float = 0
    cell_size: int = cell_size
    max_goals = env.num_goals

    vis: Visualizer = Visualizer(env, cell_size=cell_size, visualize=False)

    # Collect benchmark data
    (cheapest_solution, agent_goal_bench, goal_goal_bench,
     goal_solutions_bench, possible_solutions_bench,
     cheapest_solution_bench) = env.find_numerical_solution()

    done: bool = False
    current_assignments: Dict[Goal, Agent] = {}

    while not done and cheapest_solution:
        if cheapest_solution is None:
            break
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

        # Random obstacle
        if np.random.rand() < chance_of_adding_random_obstacle:
            if env.add_random_obstacle_close_to_position(position=np.random.choice(env.agents).position):
                cheapest_solution = env.find_numerical_solution()[0]

        # Random goal
        if np.random.rand() < chance_of_adding_random_goal:
            if len(env.scheduler.unclaimed_goals) < max_goals - 1:
                location = env.grid_map.random_valid_location_close_to_position(
                    position=np.random.choice(env.agents).position, max_distance=30)
                env.add_random_goal(location=location)
                cheapest_solution = env.find_numerical_solution()[0]

        vis.run_step(speed=speed)
        done = env.scheduler.all_goals_claimed()
        if done:
            break

    return (agent_goal_bench, goal_goal_bench, goal_solutions_bench,
            possible_solutions_bench, cheapest_solution_bench)


def get_time_unit_scale(max_value):
    """
    Returns the appropriate time unit and scale factor based on the maximum value.
    """
    if max_value < 1e-6:
        return 'ns', 1e9  # Nanoseconds
    elif max_value < 1e-3:
        return 'µs', 1e6  # Microseconds
    elif max_value < 1:
        return 'ms', 1e3  # Milliseconds
    else:
        return 's', 1  # Seconds


def run_benchmarks_3d(num_agents_list, num_goals_list, fixed_params):
    results_list = []
    num_skills = fixed_params['num_skills']
    size = fixed_params['size']
    num_obstacles = 0
    cell_size = int(1000 / size)
    speed = 300
    num_runs = 3

    for num_agents, num_goals in product(num_agents_list, num_goals_list):
        avg_times = {k: 0 for k in ["agent_goal", "goal_goal", "goal_solutions",
                                    "possible_solutions", "cheapest_solution"]}
        for _ in range(num_runs):
            env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                              num_obstacles=num_obstacles, num_skills=num_skills, use_geo_data=True)
            benchmarks = run_sim(env=env, speed=speed, cell_size=cell_size)
            env.reset()
            for i, key in enumerate(avg_times.keys()):
                avg_times[key] += benchmarks[i].elapsed_time
        # Average the times
        for key in avg_times.keys():
            avg_times[key] /= num_runs
        # Compute total time
        total_time = sum(avg_times.values())
        avg_times['total_time'] = total_time
        # Add the results to the list
        run_config = {
            "num_agents": num_agents,
            "num_goals": num_goals,
            "num_skills": num_skills,
            "size": size,
        }
        results_list.append({**run_config, **avg_times})

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results_list)
    return results_df


def run_benchmarks_vs_param(param_list, param_name, fixed_params):
    # Your existing run_benchmarks_vs_param function remains unchanged
    results_list = []
    num_obstacles = 0
    speed = 300
    num_runs = 3

    for param_value in param_list:
        run_config = fixed_params.copy()
        run_config[param_name] = param_value
        cell_size = int(1000 / run_config.get('size', fixed_params.get('size', 100)))

        avg_times = {k: 0 for k in ["agent_goal", "goal_goal", "goal_solutions",
                                    "possible_solutions", "cheapest_solution"]}
        for _ in range(num_runs):
            env = Environment(size=run_config.get('size'),
                              num_agents=run_config.get('num_agents'),
                              num_goals=run_config.get('num_goals'),
                              num_obstacles=num_obstacles,
                              num_skills=run_config.get('num_skills'),
                              use_geo_data=True)
            benchmarks = run_sim(env=env, speed=speed, cell_size=cell_size)
            env.reset()
            for i, key in enumerate(avg_times.keys()):
                avg_times[key] += benchmarks[i].elapsed_time
        # Average the times
        for key in avg_times.keys():
            avg_times[key] /= num_runs
        # Compute total time
        total_time = sum(avg_times.values())
        avg_times['total_time'] = total_time
        # Add the results to the list
        results_list.append({**run_config, **avg_times})

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results_list)
    return results_df


def plot_results_3d_bar(results_df, x_param, y_param, benchmark_types, filter_params=None):
    # Your existing plot_results_3d_bar function remains unchanged
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
    import numpy as np

    if filter_params:
        for key, value in filter_params.items():
            results_df = results_df[results_df[key] == value]

    for benchmark in benchmark_types:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Pivot the data to create a grid
        pivot_df = results_df.pivot_table(index=y_param, columns=x_param,
                                          values=benchmark, aggfunc='mean')
        X_unique = pivot_df.columns.values.astype(float)
        Y_unique = pivot_df.index.values.astype(float)
        X, Y = np.meshgrid(X_unique, Y_unique)
        Z = pivot_df.values

        # Flatten the data for plotting
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        Z_flat = Z.ravel()

        if len(Z_flat) == 0:
            print(f"No data available for {benchmark} with the given parameters. Skipping plot.")
            continue

        dx = dy = 0.8  # Adjusted bar width for better spacing
        dz = Z_flat

        # Adjust z-axis labels
        z_unit, z_scale = get_time_unit_scale(dz.max())
        dz_scaled = dz * z_scale  # Scale the data for better readability

        # Create a colormap using plt.get_cmap
        cmap = plt.get_cmap('viridis')

        # Normalize colors based on z-values
        max_dz = dz_scaled.max()
        min_dz = dz_scaled.min()
        norm = plt.Normalize(min_dz, max_dz)
        colors = cmap(norm(dz_scaled))

        # Set alpha for transparency
        alpha = 0.7  # Adjust transparency (0.0 to 1.0)

        # Plot the bars with transparency
        ax.bar3d(X_flat - dx / 2, Y_flat - dy / 2, np.zeros_like(dz_scaled),
                 dx, dy, dz_scaled, color=colors, alpha=alpha, shade=False)

        # Set labels and ticks
        ax.set_xticks(X_unique)
        ax.set_xticklabels(pivot_df.columns.values)
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)

        ax.set_yticks(Y_unique)
        ax.set_yticklabels(pivot_df.index.values)
        ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)

        ax.set_zlabel(f'Time ({z_unit})', fontsize=12)
        title = f'{benchmark.replace("_", " ").title()} Benchmark'

        # Include fixed parameters in the title
        if filter_params:
            fixed_params_str = ', '.join(f'{k}={v}' for k, v in filter_params.items())
            title += f' ({fixed_params_str})'

        ax.set_title(title, fontsize=14)

        # Format z-axis tick labels to avoid scientific notation
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        # Rotate for better visualization
        ax.view_init(elev=20., azim=-35)

        # Adjust layout
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

        # Include color bar for reference
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(dz_scaled)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'Time ({z_unit})', fontsize=12)

        # Include filter parameters in filename for clarity
        filter_str = '_'.join([f'{k}{v}' for k, v in filter_params.items()]) if filter_params else ''
        plt.savefig(f'plots/{benchmark}_{x_param}_{y_param}_{filter_str}_3dbar.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("Benchmark results saved as 3D bar charts.")


def plot_benchmarks_vs_parameter(results_df, param, benchmark_types, fixed_params=None):
    # Your existing plot_benchmarks_vs_parameter function remains unchanged
    if fixed_params:
        for key, value in fixed_params.items():
            results_df = results_df[results_df[key] == value]

    if results_df.empty:
        print(f"No data available for {param} with fixed parameters {fixed_params}. Skipping plot.")
        return

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))

    for benchmark in benchmark_types:
        # Group data by the parameter and compute mean times
        grouped = results_df.groupby(param)[benchmark].mean().reset_index()

        if grouped.empty:
            print(f"No data available for {benchmark} vs {param}. Skipping plot.")
            continue

        # Adjust time units
        max_time = grouped[benchmark].max()
        time_unit, scale = get_time_unit_scale(max_time)
        grouped[benchmark] *= scale

        # Plot
        plt.plot(grouped[param], grouped[benchmark], marker='o', label=benchmark.replace("_", " ").title())

    plt.xlabel(param.replace('_', ' ').title())
    plt.ylabel(f'Time ({time_unit})')
    title = f'Benchmark Times vs {param.replace("_", " ").title()}'

    # Include fixed parameters in the title
    if fixed_params:
        fixed_params_str = ', '.join(f'{k}={v}' for k, v in fixed_params.items())
        title += f' ({fixed_params_str})'

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    param_str = '_'.join([f'{k}{v}' for k, v in fixed_params.items()]) if fixed_params else ''
    plt.savefig(f'plots/benchmark_times_vs_{param}_{param_str}.png', dpi=300)
    plt.close()

    print(f"Plot of benchmark times vs {param} saved.")


if __name__ == "__main__":
    # Update benchmark_types to include 'total_time'
    benchmark_types = ["agent_goal", "goal_goal", "goal_solutions", "possible_solutions", "cheapest_solution", "total_time"]

    # Run benchmarks for 3D bar charts
    num_agents_list = [2,3]
    num_goals_list = [4, 5, 6, 7]
    fixed_params_3d = {
        "num_skills": 1,
        "size": 200
    }
    results_df_3d = run_benchmarks_3d(num_agents_list, num_goals_list, fixed_params_3d)

    plot_results_3d_bar(results_df_3d, x_param="num_agents", y_param="num_goals",
                        benchmark_types=benchmark_types, filter_params=fixed_params_3d)

    # Run benchmarks for num_skills line plot
    # num_skills_list = [1, 2, 3, 4, 5]
    # fixed_params_skills = {
    #     "num_agents": 3,
    #     "num_goals": 8,
    #     "size": 200
    # }
    # results_df_skills = run_benchmarks_vs_param(num_skills_list, "num_skills", fixed_params_skills)

    # plot_benchmarks_vs_parameter(results_df_skills, param="num_skills",
    #                              benchmark_types=benchmark_types, fixed_params=fixed_params_skills)

    # Run benchmarks for size line plot
    size_list = [10, 20, 50, 100, 150, 200, 300, 500]
    fixed_params_size = {
        "num_agents": 3,
        "num_goals": 8,
        "num_skills": 2
    }
    results_df_size = run_benchmarks_vs_param(size_list, "size", fixed_params_size)

    plot_benchmarks_vs_parameter(results_df_size, param="size",
                                 benchmark_types=benchmark_types, fixed_params=fixed_params_size)

    print("Benchmark completed. Plots saved.")

    # Save all data to CSV
    results_df_3d.to_csv('datasets/benchmark_results_3d.csv', index=False)
    # results_df_skills.to_csv('datasets/benchmark_results_num_skills.csv', index=False)
    results_df_size.to_csv('datasets/benchmark_results_size.csv', index=False)
    print("Benchmark results saved as CSV files.")