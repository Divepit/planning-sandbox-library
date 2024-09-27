from typing import Dict, List
import pygame
import numpy as np

from planning_sandbox.environment_class import Environment
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import matplotlib.cm as cm
import matplotlib.colors as colors

# Colors
BACKGROUND = (240, 240, 240)
GRID_LINE = (100, 100, 100)
OBSTACLE = (100, 100, 100)
AGENT = (41, 128, 185)
GOAL_UNCLAIMED = (231, 76, 60)
GOAL_CLAIMED = (46, 204, 113)
ASSIGNMENT = (241, 196, 15)
PATH = (52, 152, 219)  # New color for the path
TEXT = (50, 50, 50)

# Cell size
CELL_SIZE = 5

class Visualizer:
    def __init__(self, sandboxEnv, cell_size=CELL_SIZE, visualize=True, show_3d_elevation=False):
        pygame.init()
        self.visualize = visualize
        self.show_3d_elevation = show_3d_elevation
        self.assignments: Dict[Agent, Goal] = {}
        self.sandboxEnv: Environment = sandboxEnv
        if self.visualize:
            self.cell_size = cell_size
            self.size = self.sandboxEnv.size * self.cell_size
            self.screen = pygame.display.set_mode((self.size, self.size))
            pygame.display.set_caption("Modern Grid World")
            self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
            self.debug_font = pygame.font.Font(pygame.font.get_default_font(), 12)

        if self.sandboxEnv.grid_map.use_geo_data:
            self.elevation_surface = None
            self.update_elevation_surface()

    @property
    def agents(self):
        return self.sandboxEnv.agents

    @property
    def goals(self):
        return self.sandboxEnv.goals

    @property
    def obstacles(self):
        return self.sandboxEnv.obstacles

    # Function for coloration and 3D map created by ChatGPT (https://chatgpt.com/share/66f66db2-2750-8008-a104-700c5c92cfa9)
    def update_elevation_surface(self):
        if not self.visualize:
            return

        elevations = self.sandboxEnv.grid_map.downscaled_data

        # Normalize elevations between 0 and 1 for colormap
        min_elevation = np.min(elevations)
        max_elevation = np.max(elevations)
        normalized_elevations = (elevations - min_elevation) / (max_elevation - min_elevation)

        # Get the 'terrain' colormap from Matplotlib
        colormap = cm.get_cmap('terrain')

        # Create a surface with the same size as the grid
        self.elevation_surface = pygame.Surface((self.size, self.size))

        for y in range(self.sandboxEnv.size):
            for x in range(self.sandboxEnv.size):
                elevation = normalized_elevations[y, x]
                # Get the RGBA color from the colormap
                rgba_color = colormap(elevation)
                # Convert RGBA to RGB tuple in 0-255 range
                rgb_color = tuple(int(255 * c) for c in rgba_color[:3])
                pygame.draw.rect(
                    self.elevation_surface,
                    rgb_color,
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

        if self.show_3d_elevation:
            self.display_3d_elevation()



    def display_3d_elevation(self):
        elevations = self.sandboxEnv.grid_map.downscaled_data
        size = elevations.shape[0]
        
        # Create coordinate grids
        x = np.arange(0, size)
        y = np.arange(0, size)
        x, y = np.meshgrid(x, y)
        
        # Create a figure and a 3D Axes
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Rotate 90 degrees around z
        ax.view_init(elev=40, azim=200)
        
        # Plot the surface with a colormap
        surface = ax.plot_surface(x, y, elevations, cmap='terrain', linewidth=0, antialiased=False)
        
        # Add a color bar to map colors to elevation values
        fig.colorbar(surface, shrink=0.5, aspect=10)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Elevation Map')
        
        # Show the plot
        plt.show(block=False)

    def draw_grid(self):
        if self.visualize == False:
            return
        self.screen.fill(BACKGROUND)
        if self.sandboxEnv.grid_map.use_geo_data:
            self.screen.blit(self.elevation_surface, (0, 0))
        for x in range(0, self.size, self.cell_size):
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, self.size))
        for y in range(0, self.size, self.cell_size):
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (self.size, y))

    def draw_obstacles(self):
        if self.visualize == False:
            return
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, OBSTACLE, (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size, self.cell_size, self.cell_size))

    def draw_agents(self):
        if self.visualize == False:
            return
        for i, agent in enumerate(self.agents):
            x, y = agent.position
            center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, AGENT, center, self.cell_size // 2)
            pygame.draw.circle(self.screen, BACKGROUND, center, self.cell_size // 2 - 2)
            text = self.font.render(f"{i}", True, TEXT)
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)
            skills_text = self.debug_font.render(f"{agent.skills}", True, TEXT)
            self.screen.blit(skills_text, (x * self.cell_size, y * self.cell_size + self.cell_size))

    def draw_goals(self):
        if self.visualize == False:
            return
        for i, goal in enumerate(self.goals):
            x, y = goal.position
            color = GOAL_CLAIMED if goal.claimed else GOAL_UNCLAIMED
            rect = pygame.Rect(x * self.cell_size + 1, y * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            text = self.font.render(f"{i}", True, TEXT)
            text_rect = text.get_rect(center=(x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2))
            self.screen.blit(text, text_rect)
            skills_text = self.debug_font.render(f"{[int(skill) for skill in goal.required_skills]}", True, TEXT)
            self.screen.blit(skills_text, (x * self.cell_size, y * self.cell_size + self.cell_size))

    def draw_assignments(self):
        if self.visualize == False:
            return
        for agent, goal in self.assignments.items():
            start = (agent.position[0] * self.cell_size + self.cell_size // 2,
                     agent.position[1] * self.cell_size + self.cell_size // 2)
            end = (goal.position[0] * self.cell_size + self.cell_size // 2,
                   goal.position[1] * self.cell_size + self.cell_size // 2)
            pygame.draw.line(self.screen, ASSIGNMENT, start, end, 2)

    def draw_paths(self):
        if self.visualize == False:
            return
        for agent in self.agents:
            if agent in self.sandboxEnv.planner.paths:
                path = self.sandboxEnv.planner.paths[agent]
                for i in range(len(path) - 1):
                    start = (path[i][0] * self.cell_size + self.cell_size // 2,
                             path[i][1] * self.cell_size + self.cell_size // 2)
                    end = (path[i+1][0] * self.cell_size + self.cell_size // 2,
                           path[i+1][1] * self.cell_size + self.cell_size // 2)
                    pygame.draw.line(self.screen, PATH, start, end, 2)

    def set_assignments(self, assignments):
        if self.visualize == False:
            return
        self.assignments = assignments

    def run_step(self, iterations=1, speed=20):
        if self.visualize == False:
            return
        done = False
        clock = pygame.time.Clock()
        i = 0
        while not done:
            if i >= iterations:
                break
            self.draw_grid()
            self.draw_obstacles()
            self.draw_paths()  # Draw paths before goals and agents
            self.draw_goals()
            self.draw_agents()
            self.draw_assignments()
            pygame.display.flip()
            clock.tick(speed)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            i += 1

    def close(self):
        if self.visualize == False:
            return
        pygame.quit()