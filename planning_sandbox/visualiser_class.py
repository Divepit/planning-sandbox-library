import pygame

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Cell size
CELL_SIZE = 30

class Visualizer:
    def __init__(self, sandboxEnv):
            pygame.init()
            self.sandboxEnv = sandboxEnv
            self.width = sandboxEnv.width*CELL_SIZE
            self.height = sandboxEnv.height*CELL_SIZE
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Grid World")
            self.font = pygame.font.Font(None, 24)

            self.assignments = {}


    @property
    def agents(self):
        return self.sandboxEnv.agents

    @property
    def goals(self):
        return self.sandboxEnv.goals

    @property
    def obstacles(self):
        return self.sandboxEnv.obstacles

    def draw_grid(self):
        for x in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, self.height))
        for y in range(0, self.height, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (self.width, y))

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, BLACK, (obstacle[0] * CELL_SIZE, obstacle[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_agents(self):
        for i, agent in enumerate(self.agents):
            x, y = agent.position
            pygame.draw.circle(self.screen, BLUE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
            text = self.font.render(str(f"{i} {agent.skills}"), True, BLACK)
            self.screen.blit(text, (x * CELL_SIZE + CELL_SIZE // 3, y * CELL_SIZE + CELL_SIZE // 3))

    def draw_goals(self):
        for i, goal in enumerate(self.goals):
            x, y = goal.position
            color = RED if goal.claimed else GREEN
            pygame.draw.rect(self.screen, color, (x * CELL_SIZE + CELL_SIZE // 4, y * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
            text = self.font.render(str(f"{i} {goal.required_skills}"), True, BLACK)
            self.screen.blit(text, (x * CELL_SIZE + CELL_SIZE // 3, y * CELL_SIZE + CELL_SIZE // 3))

    # def draw_assignments(self):
    #     for agent in self.agents:
    #         if agent.goal:
    #             start = (agent.position[0] * CELL_SIZE + CELL_SIZE // 2, 
    #                      agent.position[1] * CELL_SIZE + CELL_SIZE // 2)
    #             end = (agent.goal.position[0] * CELL_SIZE + CELL_SIZE // 2, 
    #                    agent.goal.position[1] * CELL_SIZE + CELL_SIZE // 2)
    #             pygame.draw.line(self.screen, YELLOW, start, end, 2)

    def draw_assignments(self):
        for agent, goal in self.assignments.items():
            start = (agent.position[0] * CELL_SIZE + CELL_SIZE // 2, 
                     agent.position[1] * CELL_SIZE + CELL_SIZE // 2)
            end = (goal.position[0] * CELL_SIZE + CELL_SIZE // 2, 
                   goal.position[1] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.line(self.screen, YELLOW, start, end, 2)

    def set_assignments(self, assignments):
        self.assignments = assignments

    def run_step(self, iterations=1):

        done = False

        clock = pygame.time.Clock()

        i = 0
        while not done:
            if i >= iterations:
                break
            
            if iterations > 1:
                print("Iterations left: ", iterations - i)

            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_obstacles()
            self.draw_goals()
            self.draw_agents()
            self.draw_assignments()

            pygame.display.flip()
            clock.tick(5)


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            i += 1

    def close(self):
        pygame.quit()