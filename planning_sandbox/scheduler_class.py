import numpy as np

class Scheduler:
    def __init__(self, agents, goals):
        self.agents = agents
        self.goals = goals
        
        self.goal_assignments = {}

    def reset(self):
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
        if not agents:
            return []
        # print(f"Agents present at goal: {agents}")
        skills = []
        for agent in agents:
            skills.extend(agent.skills)
        return skills
    def randomly_distribute_goals_to_agents(self):
        for _ in self.agents:
            self._assign_random_unassigned_unclaimed_goal_to_random_unassigned_agent()
    
    def _goal_can_be_claimed(self, goal):
        skills_of_agents_present = self._get_skills_of_agents_present_at_goal(goal)
        if not skills_of_agents_present:
            return False
        # print(f"Skills of agents present at goal: {skills_of_agents_present}")
        skills_required = goal.required_skills
        # print(f"Skills required for goal: {skills_required}, Skills of agents present: {skills_of_agents_present}")
        if set(skills_required).issubset(set(skills_of_agents_present)):
            return True
    
    def _update_goal_status(self, goal):
        amount_of_claimed_goals = 0
        if goal.claimed:
            return 0
        if self._goal_can_be_claimed(goal):
            # print(f"Goal at position {goal.position} can be claimed")
            goal.claim()
            amount_of_claimed_goals += 1
        return amount_of_claimed_goals

    def _get_normalized_claimed_goals(self):
        # 1 if claimed, 0 if not
        return [int(goal.claimed) for goal in self.goals]
    
    def all_goals_claimed(self):
        return all([goal.claimed for goal in self.goals])

    def get_goal_for_agent(self, agent):
        return self.goal_assignments[agent]
    
    def update_goal_statuses(self):
        amount_of_claimed_goals = 0
        for goal in self.goals:
            amount_of_claimed_goals += self._update_goal_status(goal)
        return amount_of_claimed_goals
