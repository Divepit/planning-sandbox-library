import networkx as nx
import numpy as np
import time
from PIL import Image
from skimage.transform import resize

TIF = '/Users/marco/Programming/PlanningEnvironmentLibrary/planning_sandbox/maps/shoemaker_ele_5mpp.tif'
MPP = 5
WINDOW_SIZE = 4000
X_OFFSET = 0
Y_OFFSET = 0
ORIGINAL_TOP_LEFT = (X_OFFSET, Y_OFFSET)
ORIGINAL_BOTTOM_RIGHT = (X_OFFSET + WINDOW_SIZE, Y_OFFSET + WINDOW_SIZE)

class GridMap:
    def __init__(self, size, num_obstacles, use_geo_data=False, flat_map_for_testing=False, downhill_slope_max=np.inf, uphill_slope_max=np.inf, uphill_factor=1):
        
        if flat_map_for_testing:
            print("========= ATTENTION =========")
            print("Using flat map for testing")
            print("========= ATTENTION =========")
            time.sleep(5)
        
        self.use_geo_data = use_geo_data
        self.flat_map_for_testing = flat_map_for_testing
        
        self.downhill_slope_max = downhill_slope_max
        self.uphill_slope_max = uphill_slope_max
        self.uphill_factor = uphill_factor

        self.size = size
        self.graph = None
        self.is_connected = False

        self.num_obstacles = num_obstacles
        self.obstacles = []
        
        if self.use_geo_data:
            self.data = self._extract_data_from_tif()
            self.downscaled_data, self.pixel_size = self._downscale_data()
            self.create_directed_graph(data=self.downscaled_data, pixel_size=self.pixel_size, uphill_factor=uphill_factor, downhill_slope_max=downhill_slope_max, uphill_slope_max=uphill_slope_max)
        else:
            self._generate_connected_grid_with_obstalces(self.num_obstacles)

    def _random_position(self):
        return (np.random.randint(0, self.size), np.random.randint(0, self.size))
    
    def _generate_random_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            pos = self.random_valid_position()
            self.add_obstacle(pos)

    def _generate_connected_grid_with_obstalces(self, num_obstacles):
        i = 0
        while True:
            i += 1
            if i % 5 == 0:
                print(f"Reducing obstacle count by 10%")
                num_obstacles = int(num_obstacles*0.9)
                print(f"Current obstacle count: {num_obstacles}")

            self.graph = nx.grid_2d_graph(self.size, self.size)
            self.obstacles.clear()
            self._generate_random_obstacles(num_obstacles)
            if nx.is_connected(self.graph):
                self.is_connected = True
                break
    
    def reset(self):
        self.obstacles.clear()
        self.graph = None
        self.is_connected = False
        if self.use_geo_data:
            self.create_directed_graph(data=self.downscaled_data, pixel_size=self.pixel_size, uphill_factor=self.uphill_factor, downhill_slope_max=self.downhill_slope_max, uphill_slope_max=self.uphill_slope_max)
        else:
            self._generate_connected_grid_with_obstalces(self.num_obstacles)


    def add_obstacle(self, pos):
        self.obstacles.append(pos)
        self.graph.remove_node(pos)
    
    def is_valid_position(self, pos):
        return (0 <= pos[0] < self.size and 
                0 <= pos[1] < self.size and 
                pos not in self.obstacles)
    
    def random_valid_position(self):
        pos = self._random_position()
        while not self.is_valid_position(pos):
            pos = self._random_position()
        return pos
    
    def shortest_path(self, start, goal):
        if self.use_geo_data:
            path = nx.astar_path(G=self.graph, source=start, target=goal, weight="weight")
        else:
            path = nx.astar_path(G=self.graph, source=start, target=goal)
        return path
    
    def get_normalized_positions(self, positions):
        return [(pos[0] / self.size, pos[1] / self.size) for pos in positions]
    
    def random_valid_location_close_to_position(self,position, max_distance):
        x,y = position
        x += np.random.randint(-max_distance,max_distance+1)
        y += np.random.randint(-max_distance,max_distance+1)
        x = np.clip(x,0,self.size-1)
        y = np.clip(y,0,self.size-1)
        if not self.is_valid_position((x,y)):
            return self.random_valid_location_close_to_position(position,max_distance)
        return (x,y)
    
    
    def _print_tif_info(self,file_path):
        print("TIFF File Information:")
        print("-----------------------")
        
        with Image.open(file_path) as img:
            width, height = img.size
            print(f"Dimensions: {width} x {height}")
            data = np.array(img)
            min_val = np.min(data)
            max_val = np.max(data)
            print(f"Min value: {min_val:.2f}")
            print(f"Max value: {max_val:.2f}")
            
        print("-----------------------")

    
    def _extract_data_from_tif(self):
        self._print_tif_info(TIF)
        with Image.open(TIF) as img:
            cropped = img.crop((ORIGINAL_TOP_LEFT[0], ORIGINAL_TOP_LEFT[1], ORIGINAL_BOTTOM_RIGHT[0], ORIGINAL_BOTTOM_RIGHT[1]))
            data = np.array(cropped)
        return data
    
    def _downscale_data(self):

        current_height, current_width = self.data.shape
        current_pixel_size = MPP * max(current_height, current_width) / max(self.size, self.size)

        scale_factor = current_pixel_size / MPP

        new_height = int(current_height / scale_factor)
        new_width = int(current_width / scale_factor)

        downscaled_data = resize(self.data, (new_height, new_width), order=1, mode='reflect', anti_aliasing=True)

        new_pixel_size = MPP * max(current_height, current_width) / max(new_height, new_width)

        return downscaled_data, new_pixel_size

    
    def check_if_connected(self):
        if self.use_geo_data or self.graph is None:
            return False
        if nx.is_connected(self.graph):
            return True
        
    def get_cost_for_move(self, start, goal):
        if start == goal:
            return 0
        if self.use_geo_data:
            return self.graph.get_edge_data(start, goal)['weight']
        else:
            return MPP
    
    def create_directed_graph(self, data, pixel_size, uphill_factor, downhill_slope_max, uphill_slope_max):
        np.set_printoptions(linewidth=100000)
        height, width = data.shape
        G = nx.DiGraph()
        if self.flat_map_for_testing:
            data = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                node = (i, j)
                node_elevation = data[i, j]
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_node = (ni, nj)
                        neighbor_elevation = data[ni, nj]
                        elevation_diff = neighbor_elevation - node_elevation
                        slope = elevation_diff / pixel_size
                        weight = self.calculate_weight(slope)
                        
                        G.add_edge(node, neighbor_node, weight=weight)
        for i in range(height):
            for j in range(width):
                node = (i, j)
                node_elevation = data[i, j]
                G.nodes[node]["elevation"] = node_elevation

        self.graph = G

    def calculate_weight(self, slope):
        if self.flat_map_for_testing:
            return MPP
        
        if slope > 0:
            if slope > self.uphill_slope_max:
                weight = np.inf
            else:
                weight = slope * self.uphill_factor
        else:
            if abs(slope) > self.downhill_slope_max:
                weight = np.inf
            else:
                weight = abs(slope)
        
        weight += 0.1 * MPP
        return weight