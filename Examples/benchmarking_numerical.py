# import cProfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer
from planning_sandbox.benchmark_class import Benchmark



from copy import deepcopy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_sim(env: Environment, speed, visualize=True):
    
    compute_bench = Benchmark('compute',start_now=True, silent=True)
    env.find_numerical_solution()
    computation_time = compute_bench.stop()

    while not env.scheduler.all_goals_claimed():
        
        assert env.full_solution, "No solution found"

        # TODO: FIX BENCHING OF STEPS - NOT NECESSARY IN ALL CASES
        compute_bench = Benchmark('compute',start_now=True, silent=True)
        env.step_environment(fast=False)
        computation_time += compute_bench.stop()
    
    if visualize:
        visualizer: Visualizer = Visualizer(env, speed=speed)
        visualizer.visualise_full_solution(fast=False)

    return computation_time

def clone_environments(env: Environment, num_clones):
    seed_envs = []
    print("Clone environments...")
    for _ in range(num_clones):
        print(f"Progress: {len(seed_envs)+1}/{num_clones}", end='\r')
        env.reset()
        seed_envs.append(deepcopy(env))
    logging.debug("Done cloning environments.")
    return seed_envs

def plot_benchmarks(solve_types, results, num_agents, num_goals, num_skills, size):

    sns.set_theme(style='whitegrid')
    metrics = ['runtimes', 'steps', 'costs', 'wait_times']
    metric_names = ['Runtime (s)', 'Steps', 'Total Cost', 'Waited Steps']
    num_metrics = len(metrics)

    iterations = len(results[solve_types[0]]['runtimes'])

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

    additional_info = 'Cost Function (at each step) = slope + 0.1 * distance_per_step'

    if 'fast' in results and 'optimal' in results:
        if results['optimal']['costs'].mean() < results['fast']['costs'].mean():
            word = 'cheaper'
            percentage = (results['fast']['costs'].mean() - results['optimal']['costs'].mean()) / results['fast']['costs'].mean() * 100
        else:
            word = 'more expensive'
            percentage = (results['optimal']['costs'].mean() - results['fast']['costs'].mean()) / results['optimal']['costs'].mean() * 100
        main_result = 'The optimal solver was {:.2f}% {} than the fast solver in terms of total cost.'.format(percentage, word)
    else:
        main_result = ''

    fig.suptitle(
        f'\nSimulation Results ({iterations} iterations for each solver)\n'
        f'\nAgents: {num_agents}, Goals: {num_goals}, Skills: {num_skills}, Grid Size: {size}x{size}\n'
        f'\n{additional_info}\n'
        f'\n{main_result}\n',
        fontsize=18,
        y=1.02
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

def run_benchmarks_on_environments(env,solve_types, iterations, num_agents, num_goals, num_skills, size, speed, visualize):
    seed_envs = clone_environments(env, iterations)
    results = {}
    for solve_type in solve_types:
        print(f"\nRunning simulations for solve type: {solve_type}")
        runtimes = []
        all_steps = []
        all_costs = []
        all_wait_times = []
        for env in seed_envs:
            print(f"Progress: {len(runtimes)+1}/{iterations}", end='\r')
            env_copy: Environment = deepcopy(env)
            env_copy.solve_type = solve_type
            env_copy.replan_on_goal_claim = solve_type == 'fast'
            computation_time = run_sim(env=env_copy, speed=speed, visualize=visualize)
            runtimes.append(computation_time)
            total_steps, steps_waited, total_cost = env_copy.get_agent_benchmarks()
            all_steps.append(total_steps)
            all_costs.append(total_cost)
            all_wait_times.append(steps_waited)

        results[solve_type] = {
            'runtimes': np.array(runtimes),
            'steps': np.array(all_steps),
            'costs': np.array(all_costs),
            'wait_times': np.array(all_wait_times)
        }
    
    plot_benchmarks(solve_types, results, num_agents, num_goals, num_skills, size)

# Some plotting functions written by chatGPT (https://chatgpt.com/share/66f558b4-9844-8008-8442-3e9021b9bbbd)
def main():
    
    visualize = 1
    iterations = 1
    num_goals: int = 6
    num_agents: int = 3
    size: int = 100
    num_skills: int = 2
    velocity = 200 #m/s (max 200)
    speed = velocity*1/5

    solve_types = ['fast','optimal']
    
    
    logging.info("Setting up environment...")
    env = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=True)

    run_benchmarks_on_environments(env,solve_types, iterations, num_agents, num_goals, num_skills, size, speed, visualize)

if __name__ == "__main__":
    # cProfile.run('main()', sort='cumtime')
    main()