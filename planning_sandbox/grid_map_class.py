import networkx as nx
import numpy as np
from PIL import Image
from skimage.transform import resize

TIF = './maps/shoemaker_ele_5mpp.tif'
MPP = 5
WINDOW_SIZE = 4000
X_OFFSET = 0  # Change this to move the window horizontally
Y_OFFSET = 0  # Change this to move the window vertically
ORIGINAL_TOP_LEFT = (X_OFFSET, Y_OFFSET)
ORIGINAL_BOTTOM_RIGHT = (X_OFFSET + WINDOW_SIZE, Y_OFFSET + WINDOW_SIZE)
TARGET_SIZE = 4000 # SQUARE MAPS ONLY
IS_SLOPE_DATA = False


UPHILL_FACTOR = 1
UPHILL_SLOPE_MAX = np.inf
DOWNHILL_SLOPE_MAX = np.inf

class GridMap:
    def __init__(self, size, num_obstacles, use_geo_data=False):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.graph = None
        self.is_connected = False
        self.use_geo_data = use_geo_data
        if self.use_geo_data:
            self.data = self._extract_data_from_tif()
            self.downscaled_data, self.pixel_size = self._downscale_data()
            self.create_directed_graph(data=self.downscaled_data, is_slope_data=IS_SLOPE_DATA, pixel_size=self.pixel_size, uphill_factor=UPHILL_FACTOR, downhill_slope_max=DOWNHILL_SLOPE_MAX, uphill_slope_max=UPHILL_SLOPE_MAX)
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
        if not self.use_geo_data:
            self.graph = None
            self._generate_connected_grid_with_obstalces(self.num_obstacles)

    def add_obstacle(self, pos):
        self.obstacles.append(pos)
        self.graph.remove_node(pos)

    def remove_obstacle(self, pos):
        self.obstacles.remove(pos)
        self.graph.add_node(pos)
    
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
            # path = nx.shortest_path(self.graph, start, goal, weight="weight")
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
        # Calculate the current dimensions and pixel size
        current_height, current_width = self.data.shape
        current_pixel_size = MPP * max(current_height, current_width) / max(self.size, self.size)

        # Calculate the scaling factor
        scale_factor = current_pixel_size / MPP

        # Calculate the new dimensions
        new_height = int(current_height / scale_factor)
        new_width = int(current_width / scale_factor)

        # Resize the data using skimage
        downscaled_data = resize(self.data, (new_height, new_width), order=1, mode='reflect', anti_aliasing=True)

        # Calculate the new pixel size
        new_pixel_size = MPP * max(current_height, current_width) / max(new_height, new_width)

        return downscaled_data, new_pixel_size

    
    def check_if_connected(self):
        if self.use_geo_data or self.graph is None:
            return False
        if nx.is_connected(self.graph):
            return True
    
    def create_directed_graph(self,data, is_slope_data, pixel_size, uphill_factor, downhill_slope_max, uphill_slope_max):
        np.set_printoptions(linewidth=100000)  # Adjust the value as needed    
        height, width = data.shape
        G = nx.DiGraph()
        for i in range(height):
            for j in range(width):
                node = (i, j)
                node_elevation = data[i, j]
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                if is_slope_data:
                    weight = 0
                    G.add_node(node, weight=node_elevation)
                    for ni, nj in neighbors:
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_node = (ni, nj)
                            neighbor_elevation = data[ni, nj]
                            G.add_edge(node, neighbor_node, weight=neighbor_elevation)
                else:
                    for ni, nj in neighbors:
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_node = (ni, nj)
                            neighbor_elevation = data[ni, nj]
                            elevation_diff = neighbor_elevation - node_elevation
                            slope = elevation_diff / pixel_size
                            if elevation_diff > 0:  # Uphill
                                if slope > uphill_slope_max:
                                    weight = np.inf
                                else:
                                    weight = slope * uphill_factor
                            else:  # Downhill or flat
                                if abs(slope) > downhill_slope_max:
                                    weight = np.inf
                                else:
                                    weight = abs(slope)
                            # Add edge to graph
                            G.add_edge(node, neighbor_node, weight=weight)
                            # G.nodes[neighbor_node]["elevation"] = neighbor_elevation
        for i in range(height):
            for j in range(width):
                node = (i, j)
                node_elevation = data[i, j]
                G.nodes[node]["elevation"] = node_elevation

        self.graph = G