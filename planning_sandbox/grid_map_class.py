import networkx as nx
import numpy as np

class GridMap:
    def __init__(self, width, height, num_obstacles):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.occupied_positions = []
        self.graph = None
        self.generate_connected_grid_with_obstalces(self.num_obstacles)


    def reset(self):
        self.obstacles.clear()
        self.occupied_positions.clear()
        self.graph = None
        self.generate_connected_grid_with_obstalces(self.num_obstacles)

    def add_obstacle(self, pos):
        self.obstacles.append(pos)
        self.graph.remove_node(pos)

    def add_occupied_position(self, pos):
        self.occupied_positions.append(pos)
    
    def is_valid_position(self, pos):
        # input: (x, y)
        
        return (0 <= pos[0] < self.width and 
                0 <= pos[1] < self.height and 
                pos not in self.obstacles and
                pos not in self.occupied_positions)
    
    def random_position(self):
        return (np.random.randint(0, self.width), np.random.randint(0, self.height))
    
    def random_valid_position(self):
        pos = self.random_position()
        while not self.is_valid_position(pos):
            pos = self.random_position()
        return pos
    
    def shortest_path(self, start, goal):
        path = nx.shortest_path(self.graph, start, goal)
        return path
        
    def generate_random_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            pos = self.random_valid_position()
            self.add_obstacle(pos)

    def generate_connected_grid_with_obstalces(self, num_obstacles):
        while True:
            self.graph = nx.grid_2d_graph(self.width, self.height)
            self.obstacles.clear()
            self.generate_random_obstacles(num_obstacles)
            if nx.is_connected(self.graph):
                print("Obstacles: ", self.obstacles)
                break

    def get_grid(self):
        grid = np.zeros((self.width, self.height))
        for obstacle in self.obstacles:
            grid[obstacle] = 1
        return grid
    
    def get_normalized_position(self, pos):
        return pos[0] / self.width, pos[1] / self.height
    
    def get_normalized_positions(self, positions):
        return [(pos[0] / self.width, pos[1] / self.height) for pos in positions]
    
    def get_normalized_obstacles(self):
        return self.get_normalized_positions(self.obstacles)