import numpy as np

class Scheduler:
    def __init__(self, agents, goals):
        self.agents = agents
        self.goals = goals
        
        self.goal_assignments = {}

    def _assign_goal_to_agent(self, agent, goal):
        self.goal_assignments[agent] = goal
    
    def _get_agent_for_goal(self, goal):
        for agent, assigned_goal in self.goal_assignments.items():
            if assigned_goal == goal:
                return agent
        return None
    
    def _get_unassigned_agents(self):
        return [agent for agent in self.agents if agent not in self.goal_assignments]
    
    def _get_unassigned_goals(self):
        return [goal for goal in self.goals if goal not in self.goal_assignments.values()]
    
    def _get_unclaimed_goals(self):
        return [goal for goal in self.goals if not goal.claimed]
    
    def _get_unassigned_unclaimed_goals(self):
        return [goal for goal in self._get_unclaimed_goals() if goal not in self.goal_assignments.values()]
    
    def _get_random_unassigned_unclaimed_goal(self):
        unassigned_goals = self._get_unassigned_unclaimed_goals()
        return np.random.choice(unassigned_goals)
    
    def _get_random_unassigned_agent(self):
        unassigned_agents = self._get_unassigned_agents()
        return np.random.choice(unassigned_agents)
    
    def _assign_random_unassigned_unclaimed_goal_to_random_unassigned_agent(self):
        agent = self._get_random_unassigned_agent()
        goal = self._get_random_unassigned_unclaimed_goal()
        self._assign_goal_to_agent(agent, goal)

    def _get_agents_present_at_goal(self, goal):
        return [agent for agent in self.agents if agent.position == goal.position]
    
    def _get_skills_of_agents_present_at_goal(self, goal):
        agents = self._get_agents_present_at_goal(goal)
        skills = []
        for agent in agents:
            skills.extend(agent.skills)
        return skills
    def randomly_distribute_goals_to_agents(self):
        for _ in self.agents:
            self._assign_random_unassigned_unclaimed_goal_to_random_unassigned_agent()
    
    def _goal_can_be_claimed(self, goal):
        skills_of_agents_present = self._get_skills_of_agents_present_at_goal(goal)
        skills_required = goal.required_skills
        return all([skill in skills_of_agents_present for skill in skills_required])
    
    def _update_goal_status(self, goal):
        if self._goal_can_be_claimed(goal):
            goal.claim()

    def _get_normalized_claimed_goals(self):
        # 1 if claimed, 0 if not
        return [int(goal.claimed) for goal in self.goals]
    
    def all_goals_claimed(self):
        return all([goal.claimed for goal in self.goals])

    def get_goal_for_agent(self, agent):
        return self.goal_assignments[agent]
    
    def update_goal_statuses(self):
        for goal in self.goals:
            self._update_goal_status(goal)
