from typing import List

from planning_sandbox.grid_map_class import GridMap
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal
from planning_sandbox.controller_class import Controller
from planning_sandbox.scheduler_class import Scheduler
from planning_sandbox.planner_class import Planner
# from visualiser_class import Visualizer

import numpy as np

class Environment:
    def __init__(self, width, height, num_agents, num_goals, num_obstacles, num_skills):
        self.width = width
        self.height = height

        self.num_agents = num_agents
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.num_skills = num_skills


        self.grid_map = GridMap(width, height, num_obstacles)
        self.obstacles = self.grid_map.obstacles

        self.agents = []
        self._initialize_agents()


        self.goals = []
        self._initialize_goals()

        self.normalized_skill_map = {i: i / self.num_skills for i in range(self.num_skills)}
        self._initialize_skills()

        self.controller = Controller(self.agents, self.grid_map)
        self.planner = Planner(self.agents, self.grid_map)
        self.scheduler = Scheduler(self.agents, self.goals)
        
        # self.visualizer = Visualizer(self.width, self.height, self.obstacles)

    def _initialize_agents(self):
        for _ in range(self.num_agents):
            agent = Agent(self.grid_map.random_valid_position())
            # self.grid_map.add_occupied_position(agent.position)
            self.agents.append(agent)

    def _initialize_goals(self):
        for _ in range(self.num_goals):
            goal = Goal(self.grid_map.random_valid_position())
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
                goal.add_skill(0)
            for agent in self.agents:
                agent.add_skill(0)
            return

        for goal in self.goals:
            amount_of_skills = np.random.randint(1, self.num_skills+1)
            skills = []
            for _ in range(amount_of_skills):
                skill = np.random.randint(0, self.num_skills)
                while skill in skills:
                    skill = np.random.randint(0, self.num_skills)
                skills.append(skill)
            goal.required_skills = skills
                
        for agent in self.agents:
            amount_of_skills = np.random.randint(1, self.num_skills+1)
            skills = []
            for _ in range(amount_of_skills):
                skill = np.random.randint(0, self.num_skills)
                while skill in skills:
                    skill = np.random.randint(0, self.num_skills)
                skills.append(skill)
            agent.skills = skills

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

    # def visualize_state(self, iterations=1):
    #     self.visualizer.run_step(self.agents, self.goals, iterations=iterations)

    def random_agent_actions(self):
        for agent in self.agents:
            agent.apply_action(self.controller.get_random_valid_action(agent))

    def assign_shortest_paths_for_scheduled_agents(self):
        for agent in self.agents:
            if agent in self.scheduler.goal_assignments:
                agent_path = self.planner.generate_shortest_path_for_agent(agent, self.scheduler.get_goal_for_agent(agent))
                self.planner.assign_path_to_agent(agent, agent_path)

    def advance_agents_on_paths(self):
        for agent in self.agents:
            if agent in self.planner.paths:
                agent.apply_action(self.planner.get_move_to_reach_next_position(agent))


# if __name__ == "__main__":
#     env = Environment(width=20, height=20, num_agents=3, num_goals=3, num_obstacles=10, num_skills=2)
#     env.scheduler.randomly_distribute_goals_to_agents()
#     env.visualizer.set_assignments(env.scheduler.goal_assignments)
#     env.assign_shortest_paths_for_scheduled_agents()

#     done = False
#     env.visualize_state()
#     iterations = 0
#     while not done:
#         iterations += 1
#         env.advance_agents_on_paths()
#         env.scheduler.update_goal_statuses()
#         env.visualize_state()
#         done = env.scheduler.all_goals_claimed()


