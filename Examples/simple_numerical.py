from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

num_agents = 3
num_goals = 5
num_skills = 2
size = 100
visualisation_speed = 100 # Max 200
solve_type = 'optimal' # 'optimal' or 'fast'
use_map = True

my_environment = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=use_map)
my_visualiser = Visualizer(my_environment, speed=visualisation_speed)

my_environment.find_numerical_solution(solve_type=solve_type)
my_solution = my_environment.full_solution

total_steps, steps_waited, total_cost, solve_time, amount_of_claimed_goals = my_environment.solve_full_solution()

my_visualiser.visualise_full_solution()

print(f"Total steps taken: {total_steps}")
print(f"Total steps waited: {steps_waited}")
print(f"Total cost: {int(total_cost)}")
print(f"Solve time: {int(solve_time*1000)} ms")
print(f"Amount of claimed goals: {amount_of_claimed_goals}")
