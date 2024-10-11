import pygame
import logging
import numpy as np
import time

from planning_sandbox.environment_class import Environment

import matplotlib.pyplot as plt
import matplotlib.cm as cm

BACKGROUND = (240, 240, 240)
GRID_LINE = (100, 100, 100)
AGENT = (41, 128, 185)
GOAL_UNCLAIMED = (231, 76, 60)
GOAL_CLAIMED = (46, 204, 113)
ASSIGNMENT = (241, 196, 15)
PATH = (52, 152, 219)
TEXT = (50, 50, 50)
CELL_SIZE = 5

class Visualizer:
    def __init__(self, env: Environment, speed=200):
        self.env = env
        self.speed = speed
        self.setup_pygame()

    def setup_pygame(self):
        pygame.init()
        self.cell_size: int = int(1000/self.env.size)
        self.size = self.env.size * self.cell_size
        self.screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption("Modern Grid World")
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.debug_font = pygame.font.Font(pygame.font.get_default_font(), 12)

        if self.env.grid_map.use_geo_data:
            self.elevation_surface = None
            self.update_elevation_surface()

    @property
    def agents(self):
        return self.env.agents

    @property
    def goals(self):
        return self.env.goals

    # Function for coloration and 3D map created by ChatGPT (https://chatgpt.com/share/66f66db2-2750-8008-a104-700c5c92cfa9)
    def update_elevation_surface(self):
        elevations = self.env.grid_map.downscaled_data

        # Normalize elevations between 0 and 1 for colormap
        min_elevation = np.min(elevations)
        max_elevation = np.max(elevations)
        normalized_elevations = (elevations - min_elevation) / (max_elevation - min_elevation)

        # Get the 'terrain' colormap from Matplotlib
        colormap = cm.get_cmap('terrain')

        # Create a surface with the same size as the grid
        self.elevation_surface = pygame.Surface((self.size, self.size))

        for y in range(self.env.size):
            for x in range(self.env.size):
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

    def display_3d_elevation(self):
        elevations = self.env.grid_map.downscaled_data
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
        self.screen.fill(BACKGROUND)
        if self.env.grid_map.use_geo_data:
            self.screen.blit(self.elevation_surface, (0, 0))
        for x in range(0, self.size, self.cell_size):
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, self.size))
        for y in range(0, self.size, self.cell_size):
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (self.size, y))

    def draw_agents(self):
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
        for agent, goal in self.env.scheduler.goal_assignments.items():
            start = (agent.position[0] * self.cell_size + self.cell_size // 2,
                     agent.position[1] * self.cell_size + self.cell_size // 2)
            end = (goal.position[0] * self.cell_size + self.cell_size // 2,
                   goal.position[1] * self.cell_size + self.cell_size // 2)
            pygame.draw.line(self.screen, ASSIGNMENT, start, end, 2)

    def draw_paths(self):
        for agent in self.agents:
            if agent in self.env.grid_map.paths:
                path = self.env.grid_map.paths[agent]
                for i in range(len(path) - 1):
                    start = (path[i][0] * self.cell_size + self.cell_size // 2,
                             path[i][1] * self.cell_size + self.cell_size // 2)
                    end = (path[i+1][0] * self.cell_size + self.cell_size // 2,
                           path[i+1][1] * self.cell_size + self.cell_size // 2)
                    pygame.draw.line(self.screen, PATH, start, end, 2)

    def run_step(self):
        logging.debug("Running visualisation step")
        clock = pygame.time.Clock()
        self.draw_grid()
        self.draw_paths()  # Draw paths before goals and agents
        self.draw_goals()
        self.draw_agents()
        self.draw_assignments()
        pygame.display.flip()
        clock.tick(self.speed)

    def close(self):
        pygame.quit()

    def visualise_full_solution(self, max_iterations = None, fast=False):
        if fast:
            logging.debug("Visualising fast solution")
            
        self.env.soft_reset()
        self.setup_pygame()
        
        if max_iterations is None:
            max_iterations = self.size**2
        
        for _ in range(max_iterations):
            not_deadlocked = self.env.step_environment(fast=fast)
            self.run_step()
            if self.env.scheduler.all_goals_claimed() or not not_deadlocked:
                break
        if not not_deadlocked:
            logging.warning("Terminated visualisation due to deadlock")
            time.sleep(1)
        self.close()
    
