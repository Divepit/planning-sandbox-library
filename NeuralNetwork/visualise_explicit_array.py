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


action_vector = [3, 3, 2, 1, 0, 2, 2, 2, 0, 0, 2, 1, 1, 0, 0]

my_environment.full_solution = my_environment.get_full_solution_from_action_vector(action_vector)

my_visualiser = Visualizer(my_environment, speed=visualisation_speed)
my_visualiser.visualise_full_solution()
