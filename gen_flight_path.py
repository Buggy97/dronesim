import json
import random
import math
import utils as utils
from scipy.special import erfc
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import heapq
import csv
import cities
import pandas as pd

# Constants
CITY_SIZE = (1000, 1000, 1000)
GRID_SIZE = (100, 100)
NUM_DRONES = 10
NUM_SPOTS = 80
NUM_PARKS = GRID_SIZE[0]*GRID_SIZE[1]
PARK_PROBABILITY = 0.3
VOXEL_SIZE = 20  # Adjusted voxel size
MIN_ALTITUDE = 60
MAX_ALTITUDE = 120
MAX_START_DELAY = 3000 # in ms
STEP_SIZE = VOXEL_SIZE
MIN_BUILD_HEIGHT = 20
MAX_BUILD_HEIGHT = 100
MIN_BUILD_WIDTH = 15
MAX_BUILD_WIDTH = 50
MIN_BUILD_DEPTH = 15
MAX_BUILD_DEPTH = 50
GRACE_PERIOD = 1  # Grace period for entry and exit times
NUM_WIFI_ANTENNAS= 5
TELE_RADIUS= 250
MIN_TELE_ANTENNA_HEIGHT=10
MAX_TELE_ANTENNA_HEIGHT=20
WIFI_ANTENNA_EFFICIENCY_DIST=100
WIFI_ANTENNA_TX_POWER = 20
TELE_ANTENNA_TX_POWER = 74
VOXEL_SIMULATION_CEILING=6
WIFI_BUILDING_ATTENUATION=20
WIFI_FLOOR_NOISE=-100
WIFI_FRAME_LENGTH=200 #Remote ID message length
SENSING_ERROR_MAX = 20
TELE_ATTENUATION = 100
TELE_FLOOR_NOISE = -100
TELE_FRAME_LENGTH = 200
SENSING_PERTURBATION_LIMIT = 5
AVERAGE_VELOCITY = 16
VELOCITY_STANDARD_DEVIATION = 5
HIGHEST_VELOCITY = 24
LOWEST_VELOCITY = 8
LATERAL_GPS_RNP = 30
VERTICAL_GPS_RNP = 50

# Initialize the voxel time schedule
voxel_time_schedule = defaultdict(list)

#Given a building, picks a point close to its perimeter
def random_point_on_top_surface(building, city_size, voxel_size, preference=None):
    x, y, z = building["x"], building["y"], building["z"]
    width, depth, height = building["width"], building["depth"], building["height"]
    
    # Choose a random face of the building
    choices = ["front", "back", "left", "right", "top"]
    face = ''
    if preference and preference in choices:
        face = preference
    else:
        face = random.choice(choices)
    #print(face)
    max_x, max_y, max_z = city_size[0]-voxel_size, city_size[1]-voxel_size, city_size[2]-voxel_size
    if face == "front":
        # Front face
        random_x = min(random.uniform(x, x + width), max_x)
        random_y = min(y + height + 1, max_y)
        random_z = min(z + 2, max_z)
    elif face == "back":
        # Back face
        random_x = min(random.uniform(x, x + width), max_x)
        random_y = min(y + height + 1, max_y)
        random_z = min(z + depth + 2, max_z)
    elif face == "left":
        # Left face
        random_x = min(x + 2, max_x)
        random_y = min(y + height + 1, max_y)
        random_z = min(random.uniform(z, z + depth), max_z)
    elif face == "right":
        # Right face
        random_x = min(x + width + 2, max_x)
        random_y = min(y + height + 1, max_y)
        random_z = min(random.uniform(z, z + depth), max_z)
    elif face == "top":
        # Top face
        random_x = min(random.uniform(x, x + width) + 1, max_x)
        random_y = min(y + height + 1, max_y)
        random_z = min(random.uniform(z, z + depth) + 1, max_z)
    
    return random_x, random_y, random_z

def calculate_optimal_points(plane_size, radius):
    h_dist = 2 * radius
    v_dist = np.sqrt(3) * radius
    points = []
    y = 0
    while y < plane_size:
        x_offset = radius if (y // v_dist) % 2 == 1 else 0
        x = x_offset
        while x < plane_size:
            points.append((x, y))
            x += h_dist
        y += v_dist
    return points


def compute_wifi_probabilities_profiles(wifi_antennas, city_size, buildings, voxel_size, display_height_limit, voxel_simulation_ceiling, attenuation, floor_noise, frame_length):
    if not voxel_simulation_ceiling:
        voxel_simulation_ceiling = (city_size[1]//voxel_size) - int((city_size[1]//voxel_size)*0.3)
        print(f'Limiting simulation at {voxel_simulation_ceiling}')
    voxel_city = create_voxel_city(city_size, buildings, voxel_size)
    profiles = get_antennas_power_profiles(wifi_antennas, voxel_city, voxel_size, voxel_simulation_ceiling, attenuation, floor_noise)
    matrix = pickup_probability_matrix(profiles, floor_noise, frame_length, 0.01)
    return matrix

def compute_tele_probabilities_profiles(tele_antennas, city_size, buildings, voxel_size, display_height_limit, voxel_simulation_ceiling, attenuation, floor_noise, frame_length):
    voxel_city = create_voxel_city(city_size, buildings, voxel_size)
    profiles = get_antennas_power_profiles(tele_antennas, voxel_city, voxel_size, voxel_simulation_ceiling, attenuation, floor_noise)
    matrix = pickup_probability_matrix(profiles, floor_noise, frame_length)
    return matrix

def get_antennas_power_profiles(antennas, voxel_city, voxel_size, height_limit, attenuation, floor_noise):
    profiles = []
    for antenna in antennas:
        vx, vy, vz = antenna['position'][0] // voxel_size, antenna['position'][1] // voxel_size, antenna['position'][2] // voxel_size
        profiles.append(simulate_radio_antenna(voxel_city, antenna['name'], (vx, vy, vz), antenna['frequency'], antenna['txPower'], voxel_size, height_limit, attenuation, floor_noise))
    return profiles

def initialize_antennas(buildings, city_size, voxel_size, num_wifi_atennas=20, wifi_antenna_positions=[], desired_tele_radius=500, min_tele_altitude=40, max_tele_altitude=60):
    if not wifi_antenna_positions:
        wifi_antenna_positions = []
    wifi_antennas = initialize_wifi_antennas(buildings, city_size, voxel_size, num_wifi_atennas, wifi_antenna_positions)
    tele_antennas = initialize_tele_antennas(city_size, desired_tele_radius, min_tele_altitude, max_tele_altitude)
    return (wifi_antennas, tele_antennas)

def initialize_wifi_antennas(buildings, city_size, voxel_size, num_antennas=20, positions=[]):
    wifi_antennas = []
    build_counter = 0
    for position in positions:
        if num_antennas<=0:
            return
        add_wifi_antenna(buildings, city_size, voxel_size, wifi_antennas, f'wifi_antenna_{build_counter}', position)
        build_counter+=1
        num_antennas-=1
    for _ in range(num_antennas):
        add_wifi_antenna(buildings, city_size, voxel_size, wifi_antennas, f'wifi_antenna_{build_counter}')
        build_counter+=1
    return wifi_antennas

def initialize_tele_antennas(city_size, desired_radius, min_antenna_altitude, max_antenna_altitude):
    tele_antennas = []
    add_tele_antennas(tele_antennas, city_size, desired_radius, min_antenna_altitude, max_antenna_altitude)
    return tele_antennas

def is_point_in_sphere(p, p2, r):
    distance_squared = sum((p[i] - p2[i]) ** 2 for i in range(3))    
    return distance_squared <= r ** 2

def add_wifi_antenna(buildings, city_size, voxel_size, array, name, position=None, eff_dist=WIFI_ANTENNA_EFFICIENCY_DIST):
    print(f'Adding wifi antenna')
    def efficiency_distance(position, other_antennas):
        for antenna in other_antennas:
            if(is_point_in_sphere(position, antenna['position'], eff_dist)):
                return False
        return True

    if not position:
        building = random.choice(buildings)
        position = random_point_on_top_surface(building, city_size, voxel_size)
        while not efficiency_distance(position, array):
            building = random.choice(buildings)
            position = random_point_on_top_surface(building, city_size, voxel_size)
    new_height = math.ceil(position[1] / voxel_size)*voxel_size #Aligns the antenna position voxel wise
    position = (position[0], new_height, position[2])
    array.append({ 'name' : name, 'position': position, 'type' : 'W', 'frequency': 2.4e9, 'floor_noise': -100, 'txPower' : WIFI_ANTENNA_TX_POWER})

def add_tele_antennas(array, city_size, desired_radius=None, min_altitude=None, max_altitude=None):
    if not desired_radius:
        desired_radius = 100
    if not min_altitude:
        min_altitude = city_size[1]/20
    if not max_altitude:
        max_altitude = min_altitude
    if max_altitude < min_altitude:
        t = max_altitude
        max_altitude = min_altitude
        min_altitude = t
    points = calculate_optimal_points(city_size[0], int(desired_radius*0.9))
    for i, (x,z) in enumerate(points):
        y = random.randint(min_altitude, max_altitude)
        array.append({'name' : f'tele_antenna_{i}', 'position': (x,y,z), 'type' : 'T', 'frequency': 34e9, 'floor_noise': -100, 'txPower' : TELE_ANTENNA_TX_POWER})

def occupied_voxels(building, voxel_size):
    # Calculate voxel ranges
    x_start = int(building["x"] // voxel_size)
    x_end = int((building["x"] + building["width"]) // voxel_size)
    y_start = int(building["y"] // voxel_size)
    y_end = int((building["y"] + building["height"]) // voxel_size)
    z_start = int(building["z"] // voxel_size)
    z_end = int((building["z"] + building["depth"]) // voxel_size)
    # Generate voxel coordinates
    voxel_coordinates = [(x, y, z) 
                        for x in range(x_start, x_end-1)
                        for y in range(y_start, y_end-1) 
                        for z in range(z_start, z_end-1)]
    return voxel_coordinates

def create_voxel_city(city_size, buildings, voxel_size):

    #Create a 3d model where obstacles are modeled after a city
    refined_buildings = [building for building in buildings if building['height']>1]
    a = math.ceil(city_size[0]/voxel_size)
    b = math.ceil(city_size[1]/voxel_size)
    c = math.ceil(city_size[2]/voxel_size)
    voxel_city = [[[0 for _ in range(a)] for _ in range(b)] for _ in range(c)]
    for building in refined_buildings:
        building_voxel = occupied_voxels(building, voxel_size)
        for ov in building_voxel:
            #print(ov)
            voxel_city[ov[0]][ov[1]][ov[2]] = 1
    return voxel_city

def bresenham_3d(start, end):
    points = []
    x, y, z = x0, y0, z0 = start
    x1, y1, z1 = end
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    step_x = 1 if x1 > x0 else -1
    step_y = 1 if y1 > y0 else -1
    step_z = 1 if z1 > z0 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x != x1:
            points.append((x, y, z))
            x += step_x
            if p1 >= 0:
                y += step_y
                p1 -= 2 * dx
            if p2 >= 0:
                z += step_z
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y != y1:
            points.append((x, y, z))
            y += step_y
            if p1 >= 0:
                x += step_x
                p1 -= 2 * dy
            if p2 >= 0:
                z += step_z
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z != z1:
            points.append((x, y, z))
            z += step_z
            if p1 >= 0:
                y += step_y
                p1 -= 2 * dz
            if p2 >= 0:
                x += step_x
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    points.append((x,y,z))  # Include the end point
    return points

def simulate_radio_antenna(voxel_city, profile_name, source, frequency, initial_power, voxel_size=20, max_heigth=6, attenuation=50, sensitivity=-90):
    print(f'Simulating antenna for (profile_name, source, frequency, initial_power, voxel_size, max_heigth, attenuation, sensitivity){profile_name, source, frequency, initial_power, voxel_size, max_heigth, attenuation, sensitivity}')
    profile = {}
    c = 3e8  # Speed of light in m/s
    f = frequency  # Frequency in Hz
    def path_loss(distance, loss_exponent=3):
        # Free Space Path Loss (FSPL) formula
        if distance == 0:
            return initial_power
        fspl = 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)
        path_loss_value = fspl + 10 * loss_exponent * np.log10(distance)
        #print(initial_power - path_loss_value)
        return initial_power - path_loss_value

    source_x, source_y, source_z = source
    comp_min = min(max_heigth+1, len(voxel_city[1]))
    for x in range(0,len(voxel_city)):
        for y in range(0,comp_min):
            for z in range(0,len(voxel_city)):
                #print(f'Processing {x,y,z}')
                distance = np.linalg.norm(source - np.array((x, y, z))) * voxel_size
                power = path_loss(distance)
                if power > sensitivity:
                    # Apply attenuation due to obstacles
                    #print(f'brese {(source_x, source_y, source_z), (x, y, z)}')
                    line_of_sight = bresenham_3d((source_x, source_y, source_z), (x, y, z))
                    for point in line_of_sight:
                        p_x, p_y, p_z = point
                        p_x, p_y, p_z = int(p_x), int(p_y), int(p_z)
                        power -= voxel_city[p_x][p_y][p_z]*attenuation
                        if power < sensitivity:
                            continue
                if power > sensitivity:
                    #print('Added {power}')
                    profile[(x,y,z)] = power
    return profile

def pickup_probability_matrix(power_profiles, noise_floor=-100, frame_length_bits=376, probability_tolerance=0.01):
    def overall_pickup_probability(current, added):
        probabilities = np.array([current, added])
        overall_failure_prob = np.prod(1 - probabilities)
        return 1 - overall_failure_prob

    # Q-function approximation
    def Q(x):
        return 0.5 * erfc(x / math.sqrt(2))

    # Calculate SNR in dB and linear scale
    def calculate_snr(rss, noise_floor):
        snr_db = rss - noise_floor
        snr_linear = 10 ** (snr_db / 10)
        return snr_db, snr_linear

    # Calculate BER for BPSK
    def calculate_ber(snr_linear):
        return Q(math.sqrt(2 * snr_linear)/2)

    # Calculate PER
    def calculate_per(ber, frame_length):
        return 1 - (1 - ber) ** frame_length

    # Calculate Probability of Successful Reception
    def calculate_success_probability(per):
        return 1 - per
    
    probability_matrix = {}
    for profile in power_profiles:
        for pos, power in profile.items():
            current_probability = probability_matrix.get(pos, 0)
            _, snr_linear = calculate_snr(power, noise_floor)
            ber = calculate_ber(snr_linear)
            per = calculate_per(ber, frame_length_bits)
            success_probability = calculate_success_probability(per)
            probability_tolerance
            overall_probability = overall_pickup_probability(current_probability, success_probability)
            if(overall_probability > probability_tolerance):
                probability_matrix[pos] = overall_probability
    
    return probability_matrix

def cuboids_intersect(cuboid1, cuboid2):
    def intervals_overlap(a_min, a_max, b_min, b_max):
        return max(a_min, b_min) <= min(a_max, b_max)
    
    (x1, y1, z1), cube_size = cuboid1
    x1_min, x1_max = x1, x1 + cube_size
    y1_min, y1_max = y1, y1 + cube_size
    z1_min, z1_max = z1, z1 + cube_size
    
    (x2, y2, z2), (width, height, depth) = cuboid2
    x2_min, x2_max = x2, x2 + width
    y2_min, y2_max = y2, y2 + height
    z2_min, z2_max = z2, z2 + depth
    
    x_overlap = intervals_overlap(x1_min, x1_max, x2_min, x2_max)
    y_overlap = intervals_overlap(y1_min, y1_max, y2_min, y2_max)
    z_overlap = intervals_overlap(z1_min, z1_max, z2_min, z2_max)
    
    return x_overlap and y_overlap and z_overlap

def building_intersects(cuboid1, cuboid2):
    def intervals_overlap(a_min, a_max, b_min, b_max):
        return max(a_min, b_min) <= min(a_max, b_max)
    
    (x1, y1, z1), (width0, height0, depth0) = cuboid1
    x1_min, x1_max = x1, x1 + width0
    y1_min, y1_max = y1, y1 + height0
    z1_min, z1_max = z1 , z1 + depth0
    
    (x2, y2, z2), (width, height, depth) = cuboid2
    x2_min, x2_max = x2, x2 + width
    y2_min, y2_max = y2, y2 + height
    z2_min, z2_max = z2, z2 + depth
    
    x_overlap = intervals_overlap(x1_min, x1_max, x2_min, x2_max)
    y_overlap = intervals_overlap(y1_min, y1_max, y2_min, y2_max)
    z_overlap = intervals_overlap(z1_min, z1_max, z2_min, z2_max)
    
    return x_overlap and y_overlap and z_overlap


def random_point(buildings, ground=True):
    while True:
        if ground:
            x = random.randint(0, CITY_SIZE[0])
            y = 0
            z = random.randint(0, CITY_SIZE[2])
        else:
            building = random.choice(buildings)
            x = int(building["x"] + random.uniform(VOXEL_SIZE, building["width"]-VOXEL_SIZE))
            y = int(building["height"])
            z = int(building["z"] + random.uniform(VOXEL_SIZE, building["depth"]-VOXEL_SIZE))
            return x // VOXEL_SIZE, y // VOXEL_SIZE, z // VOXEL_SIZE
        if not is_collision((x, y, z), buildings):
            return x // VOXEL_SIZE, y // VOXEL_SIZE, z // VOXEL_SIZE

def random_air_point(buildings, min_alt=MIN_ALTITUDE):
    while True:
        x = random.randint(0, CITY_SIZE[0] // VOXEL_SIZE )
        y = 0
        z = random.randint(0, CITY_SIZE[2] // VOXEL_SIZE )
        if not is_collision((x, y, z), buildings):
            return x, y, z

def random_building():
    width = random.uniform(MIN_BUILD_WIDTH, MAX_BUILD_WIDTH)
    depth = random.uniform(MIN_BUILD_DEPTH, MAX_BUILD_DEPTH)
    height = random.uniform(MIN_BUILD_HEIGHT, MAX_BUILD_HEIGHT)
    x = random.uniform(0, CITY_SIZE[0] - width)
    y = 0
    z = random.uniform(0, CITY_SIZE[2] - depth)
    return {"x": x, "y": y, "z": z, "width": width, "depth": depth, "height": height}

def generate_city_layout():
    buildings = []
    cell_width = CITY_SIZE[0] / GRID_SIZE[0]
    cell_depth = CITY_SIZE[2] / GRID_SIZE[1]
    
    building_count = 0
    park_count = 0

    while building_count < NUM_SPOTS:
        #print(f'buildings: {building_count}/{NUM_SPOTS}   parks: {park_count}/{NUM_PARKS}')
        i = random.randint(0, GRID_SIZE[0] - 1)
        j = random.randint(0, GRID_SIZE[1] - 1)
        cell_x = i * cell_width
        cell_z = j * cell_depth

        if random.random() < PARK_PROBABILITY and park_count < NUM_PARKS:
            check_pass = True
            for b in buildings:
                building_cuboid = [[b['x'], 0, b['z']], [b["width"], b["height"], b["depth"]]]
                park_cuboid = [[cell_x, 0, cell_z], [cell_width, 1, cell_depth]]
                if building_intersects(building_cuboid, park_cuboid):
                    check_pass = False
                else:
                    pass#print(f'PARK: {park_cuboid} intersects with {building_cuboid}')
            if check_pass:
                buildings.append({"x": cell_x, "y": 0, "z": cell_z, "width": cell_width, "depth": cell_depth, "height": 1})
                park_count+=1
                continue

        if i % 2 == 0 or j % 2 == 0:
            continue

        if building_count < NUM_SPOTS:
            building = random_building()
            building["x"] = cell_x + random.uniform(0, cell_width - building["width"])
            building["z"] = cell_z + random.uniform(0, cell_depth - building["depth"])
            check_pass = True
            for b in buildings:
                building_cuboid = [[b['x'], 0, b['z']], [b["width"], b["height"], b["depth"]]]
                candidate_building_cuboid = [[building['x'], 0, building['z']], [building["width"], building["height"], building["depth"]]]
                if building_intersects(building_cuboid, candidate_building_cuboid):
                    check_pass = False
                else:
                    pass#print(f'BUILDING: {candidate_building_cuboid} intersects with {building_cuboid}')
            if check_pass:
                buildings.append(building)
                building_count += 1
                continue
        else:
            continue

    return buildings

def is_collision(node, buildings):
    node_cuboid = [node, VOXEL_SIZE]
    for building in buildings:
        building_cuboid = [[building['x'], 0, building['z']], [building["width"], building["height"],building["depth"]]]
        if cuboids_intersect(node_cuboid, building_cuboid):
            #print(f'Node {node} colliding with {building_cuboid}')
            return True
    return False
def is_time_conflict(node, entry_time, exit_time):
    if node not in voxel_time_schedule:
        return False
    for (scheduled_entry, scheduled_exit) in voxel_time_schedule[node]:
        if not (exit_time <= scheduled_entry or entry_time >= scheduled_exit):
            return True
    return False

def nearest_node(tree, random_point):
    nearest = tree[0]
    min_dist = np.linalg.norm(np.array(nearest) - np.array(random_point))
    for node in tree:
        dist = np.linalg.norm(np.array(node) - np.array(random_point))
        if dist < min_dist:
            nearest = node
            min_dist = dist
    return nearest

def steer(from_node, to_node, step_size):
    direction = np.array(to_node) - np.array(from_node)
    length = np.linalg.norm(direction)
    if length == 0:
        return from_node
    direction = direction / length
    step = direction * min(step_size, length)
    new_node = tuple(np.array(from_node) + step)
    return tuple(int(round(coord)) for coord in new_node)

def rrt(start, goal, buildings, start_time, velocity, step_size=STEP_SIZE, max_iter=100):
    tree = [start]
    parent_map = {start: None}
    node_times = {start: (start_time, start_time + GRACE_PERIOD)}
    
    for _ in range(max_iter):
        rand_point = random_air_point(buildings)
        nearest = nearest_node(tree, rand_point)
        new_node = steer(nearest, rand_point, step_size)
        
        if is_collision(new_node, buildings):
            continue
        
        travel_time = math.ceil((VOXEL_SIZE / velocity)*1000)
        entry_time = node_times[nearest][1]  # exit time of nearest becomes entry time for new node
        exit_time = entry_time + travel_time + GRACE_PERIOD
        
        if is_time_conflict(new_node, entry_time, exit_time):
            continue
        
        if new_node not in parent_map:
            tree.append(new_node)
            parent_map[new_node] = nearest
            node_times[new_node] = (entry_time, exit_time)
            voxel_time_schedule[new_node].append((entry_time, exit_time))
            
            if np.linalg.norm(np.array(new_node) - np.array(goal)) <= step_size:
                parent_map[goal] = new_node
                node_times[goal] = (exit_time, exit_time + travel_time + GRACE_PERIOD)
                tree.append(goal)
                break

    path = []
    current = goal
    while current:
        if current not in node_times:
            print(f'Warning! Node {current} not found in times! A flight path might be missing!')
            break
        path.append((current, node_times[current]))
        current = parent_map[current]
    path.reverse()
    return path

def anytime_d_star(start, goal, buildings, start_time, exit_time, velocity, max_iterations=2000):
    params = {'start': start, 'goal':goal, 'start_time':start_time, 'exit_time':exit_time, 'velocity':velocity}
    print(f'Starting AD* for {params}')

    def heuristic(node, goal):
        return np.exp2(np.linalg.norm(np.array(node) - np.array(goal)))*10000000

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    node_times = {start: (start_time, exit_time)}
    best_path = None
    best_cost = float('inf')
    for _ in range(max_iterations):
        if not open_list:
            break
        
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            path = []
            while current != start:
                path.append((current, node_times[current]))
                current = came_from[current]
            path.append((start, node_times[start]))
            path.reverse()
            path_cost = sum(node_times[node][1] - node_times[node][0] for node, _ in path)
            if path_cost < best_cost:
                best_cost = path_cost
                best_path = path
            continue
        
        for neighbor in get_neighbors(current, buildings):
            travel_time = math.ceil((VOXEL_SIZE / velocity)*1000)
            new_cost = cost_so_far[current] + travel_time + GRACE_PERIOD
            entry_time = node_times[current][1]  # exit time of current becomes entry time for neighbor
            exit_time = math.ceil(entry_time + travel_time + GRACE_PERIOD)
            #print('Node has these entry times: {node_times[current]}')
            #print('Current neighbour has these entry times: {node_times[neighbor]}')
            #print('Checking if i can ')
            if is_time_conflict(neighbor, entry_time, exit_time):
                continue
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current
                node_times[neighbor] = (entry_time, exit_time)
                voxel_time_schedule[neighbor].append((entry_time, exit_time))

    return best_path

def get_neighbors(node, buildings):
    neighbors = []
    directions = [
        (-1, 0, 0), (1, 0, 0),  # x-axis neighbors
        (0, -1, 0), (0, 1, 0),  # y-axis neighbors
        (0, 0, -1), (0, 0, 1)   # z-axis neighbors
    ]
    for dx, dy, dz in directions:
        neighbor = (node[0] + dx, node[1] + dy, node[2] + dz)
        real_size_neighbor = (neighbor[0]*VOXEL_SIZE, neighbor[1]*VOXEL_SIZE, neighbor[2]*VOXEL_SIZE)
        if (0 <= neighbor[0] < (CITY_SIZE[0] // VOXEL_SIZE) and
                0 <= neighbor[1] < (CITY_SIZE[1] // VOXEL_SIZE) and
                0 <= neighbor[2] < (CITY_SIZE[2] // VOXEL_SIZE) and
                not is_collision(real_size_neighbor, buildings)):
            #print(f'Appending neighbour {neighbor} for {node}')
                    neighbors.append(neighbor)
    return neighbors

def generate_flight_plan(args):
    start, end, buildings, start_time, velocity = args
    print(f'Creating flight plan for: {start} -> {end} start time: {start_time}')
    start_ground = start[1] == 0
    end_ground = end[1] == 0

    takeoff_path = []

    if start[1] < (MIN_ALTITUDE // VOXEL_SIZE):
        time_exit_prev = start_time
        for i in range(int(((MIN_ALTITUDE // VOXEL_SIZE) - start[1]))+1):
            node = (start[0], start[1] + i, start[2])
            time_enter = time_exit_prev
            time_exit = int(time_enter + math.ceil((VOXEL_SIZE / velocity)*1000) + GRACE_PERIOD)
            time_exit_prev = time_exit
            voxel_time_schedule[node].append((time_enter, time_exit))
            takeoff_path.append((node, (time_enter, time_exit)))
        print(f'Created takeoff path')

    print(f'Start time before: {start_time}')
    main_start_time = takeoff_path[-1][1][0] if takeoff_path else start_time
    main_exit_time = takeoff_path[-1][1][1] if takeoff_path else (start_time + math.ceil((VOXEL_SIZE / velocity)*1000) + GRACE_PERIOD)
    print(f'Start time after: {main_start_time}')

    air_end_point = end
    air_end_point = (air_end_point[0], (air_end_point[1] + max(((MIN_ALTITUDE // VOXEL_SIZE) - air_end_point[1]), 0)), air_end_point[2])
    air_start_point = takeoff_path[-1][0] if takeoff_path else start

    d_star_path = anytime_d_star(air_start_point, air_end_point, buildings, main_start_time, main_exit_time, velocity)
    if d_star_path is not None:
        print(f'Computed AD* path for {start} -> {end}')
        main_path = d_star_path
    else:
        print(f'AD* failed.')
        return [],-1
    landing_path = []
    if(end[1] < main_path[-1][0][1]):
        time_exit_prev = main_path[-1][1][1]
        for i in range(0, int(main_path[-1][0][1] - end[1])+1):
            descent_step = main_path[-1][0][1] - i
            if descent_step < 0:
                descent_step = 0
            node = (main_path[-1][0][0], descent_step, main_path[-1][0][2])
            time_enter =  time_exit_prev
            time_exit = int(time_enter + math.ceil((VOXEL_SIZE / velocity)*1000) + GRACE_PERIOD)
            time_exit_prev = time_exit
            voxel_time_schedule[node].append((time_enter, time_exit))
            landing_path.append((node, (time_enter, time_exit)))
    print(f'Created landing path\n')
    if(takeoff_path):
        takeoff_path.pop() #Remove last element of the takeoff path that was fed as a starting point to the path finder

    if(len(main_path)<=1):
        return [], -1
    
    path = takeoff_path + main_path + landing_path

    if path:
        times = []
        for node, time in path:
            if node not in voxel_time_schedule:
                voxel_time_schedule[node] = []
            voxel_time_schedule[node].append(time)
            times.append(time)
        return list(zip([p[0] for p in path], times)), velocity
    return [],-1


def in_voxel(voxel_position, voxel_size, point):
    x, y, z = voxel_position
    x = x * voxel_size
    y = y * voxel_size
    z = z * voxel_size
    px, py, pz = point
    if (x <= px <= x + voxel_size) and (y <= py <= y + voxel_size) and (z <= pz <= z + voxel_size):
        return True
    return False


def generate_RID_dataset(trajectories, probability_matrix, voxel_size):
        # Function to check if a point is in the voxel and save based on probability
        saved_points = {}
        def perturb_position_rnp(position, lateral_rnp=LATERAL_GPS_RNP, vertical_rnp=VERTICAL_GPS_RNP):
            latitude, altitude, longitude = position
            
            lateral_std = lateral_rnp / 1.96  # Lateral RNP corresponds to 95% CI, so divide by 1.96
            vertical_std = vertical_rnp / 1.96  # Vertical RNP corresponds to 95% CI, so divide by 1.96
            
            horizontal_error_magnitude = np.random.normal(0, lateral_std)
            horizontal_error_angle = np.random.uniform(0, 2 * np.pi)
            
            delta_latitude = horizontal_error_magnitude * np.cos(horizontal_error_angle)
            delta_longitude = horizontal_error_magnitude * np.sin(horizontal_error_angle)
            
            delta_altitude = np.random.normal(0, vertical_std)
            
            perturbed_latitude = latitude + delta_latitude
            perturbed_longitude = longitude + delta_longitude
            perturbed_altitude = altitude + delta_altitude
        
            return (float(perturbed_latitude), float(perturbed_altitude), float(perturbed_longitude))
        
        for drone, trajectory, in trajectories.items():
            print(f'Generating RID data for {drone}')
            #print(f'Eval matrix: {len(probability_matrix), len(trajectory)/100}')
            saved_points[drone] = []
            if(len(trajectory)==0):
                continue
            timestamp = int(trajectory[0]['timestamp'])
            #BYPASS
            #saved_points[drone].append({'position' : perturb_position_rnp(trajectory[0]['position']), 'timestamp' : timestamp})
            #
            #Evaluate the first element in the trajectory always
            for (x,y,z), probability in probability_matrix.items():
                position = trajectory[0]['position']
                if in_voxel((x,y,z), voxel_size, position):
                    #print('Ciao')
                    prob = random.random()
                    if (prob < probability):
                       #Before appending we add a perturbations according to simulate a GPS precision error
                        saved_points[drone].append({'position' : perturb_position_rnp(position), 'timestamp' : timestamp})
                        break
                    else:
                        pass
            ##
            for i in range(1,len(trajectory),100):
                curr_timestamp = int(trajectory[i]['timestamp'])
                if curr_timestamp <= timestamp + 1000:
                    continue
                else:
                    timestamp = curr_timestamp
                #BYPASS to get all the possible for rid messages
                #saved_points[drone].append({'position' : perturb_position_rnp(trajectory[i]['position']), 'timestamp' : timestamp})
                #continue
                ####
                #print(f'Evaluating point {trajectory[i]}')
                for (x,y,z), probability in probability_matrix.items():
                    position = trajectory[i]['position']
                    if in_voxel((x,y,z), voxel_size, position):
                        #print('Ciao')
                        prob = random.random()
                        if (prob < probability):
                            #print('Appending')
                            #Before appending we add a perturbations according to simulate a GPS precision error
                            saved_points[drone].append({'position' : perturb_position_rnp(position), 'timestamp' : timestamp})
                            break
                        else:
                            pass
                            #print(f'{prob} < {probability}')
        #Trick to have a a random rid point for ech drone
        #for d in saved_points.keys():
        #    if len(saved_points[d])==0:
        #        continue
        #    rc = random.choice(saved_points[d])
        #    print(f'Random choice for {d} is {rc}')
        #    saved_points[d] = [rc]
        print(f'Saved {len(saved_points)}')                                    
        with open('RID_dataset.json', 'w') as file:
            json.dump(saved_points, file, indent=4)

def generate_sensing_dataset(trajectories, probability_matrix, voxel_size):
    print(f'Generating sensing dataset for {len(trajectories)} trajectories')
    def value_perturbation(min_value, max_value, average, reliability):
        if not (0 <= reliability <= 1):
            raise ValueError("Reliability must be a float value between 0 and 1.")
        if not (min_value <= average <= max_value):
            raise ValueError("Average must be between min_value and max_value.")
        random_value = random.uniform(min_value, max_value)
        value = reliability * average + (1 - reliability) * random_value
        return value

    def mutated_point(point, factor):
        x,y,z = point
        x = value_perturbation(x-SENSING_PERTURBATION_LIMIT, x+SENSING_PERTURBATION_LIMIT, x, factor)
        y = value_perturbation(y-SENSING_PERTURBATION_LIMIT, y+SENSING_PERTURBATION_LIMIT, y, factor)
        z = value_perturbation(z-SENSING_PERTURBATION_LIMIT, z+SENSING_PERTURBATION_LIMIT, z, factor)
        return (float(x),float(y),float(z))

# Function to check if a point is in the voxel and save based on probability
    saved_points = {}
    timesteps = []
    not_empty_trajectory_count=0
    for drone, trajectory, in trajectories.items():
        print(f'Processing sensing data for drone {drone}')
        processed_points_count = 0
        if(len(trajectory)==0):
            continue
        timestamp = int(trajectory[0]['timestamp']) - 100
        for i in range(0,len(trajectory),100):
            curr_timestamp = int(trajectory[i]['timestamp'])
            if curr_timestamp <= timestamp + 100:
                continue
            else:
                timestamp = curr_timestamp
            saved_points[timestamp] = []
            timesteps.append(timestamp)
            #Trick to have fully accurate sensing data
            #saved_points[timestamp].append((float(trajectory[i]['position'][0]), float(trajectory[i]['position'][1]), float(trajectory[i]['position'][2])))
            #processed_points_count+=1
            #print(f'Evaluating point {trajectory[i]}')
            for (x,y,z), probability in probability_matrix.items():
                position = trajectory[i]['position']
                if in_voxel((x,y,z), voxel_size, position):
                    saved_points[timestamp].append(mutated_point(position, probability))
                    processed_points_count+=1
                    break
        print(f'Processed {processed_points_count} points')
        if(processed_points_count>0):
            not_empty_trajectory_count+=1
    print(f'Processed {not_empty_trajectory_count} not empty trajectories')
    csv_data = []
    sorted(timesteps)
    for timestep in timesteps:
        row = [timestep]
        for position in saved_points[timestep]:
            row.append(position)
        if len(row) > 1:
            csv_data.append(row)

    with open('sensing_dataset.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)


def format_flight_plans_to_json(flight_plans, buildings, voxel_size, city_size, wifi_antennas=[], tele_antennas=[], drone_points = [], start_times = [], velocities = []):
    data = {
        "min_altitude" : MIN_ALTITUDE,
        "max_altitude" : MAX_ALTITUDE,
        "voxel_size": voxel_size,
        "city_size": city_size,
        "buildings": buildings,
        "wifi_antennas" : wifi_antennas,
        "tele_antennas" : tele_antennas,
        "flight_plans": {},
        "drone_points": drone_points,
        "start_times" : start_times,
        "velocities" : velocities
    }
    for i, (flight_plan, velocity) in enumerate(flight_plans):
        data["flight_plans"][f"drone_{i + 1}"] = [{"velocity": velocity, "position": [coord * voxel_size for coord in node], "entry_time": time[0], "exit_time": time[1]} for node, time in flight_plan]
    return data

def compose_flight_plans_to_be_saved(flight_plans, buildings, voxel_size, city_size, wifi_antennas=[], tele_antennas=[], drone_points = [], start_times = [], velocities = [], wifi_coverage=0, tele_coverage=0, gnss_rnp = [0,0]):
    data = {
        "min_altitude" : MIN_ALTITUDE,
        "max_altitude" : MAX_ALTITUDE,
        "voxel_size": voxel_size,
        "city_size": city_size,
        "buildings": buildings,
        "wifi_antennas" : wifi_antennas,
        "tele_antennas" : tele_antennas,
        "flight_plans": flight_plans,
        "drone_points": drone_points,
        "start_times" : start_times,
        "velocities" : velocities,
        "wifi_coverage" : wifi_coverage, 
        "tele_coverage" : tele_coverage,
        "gnss_rnp" : gnss_rnp
    }
    return data

def save_flight_trajectories_to_json(trajectories, buildings, voxel_size, city_size, precision, filename='trajectories.json'):
    data = {
        "min_altitude" : MIN_ALTITUDE,
        "max_altitude" : MAX_ALTITUDE,
        "voxel_size": voxel_size,
        "city_size": city_size,
        "buildings": buildings,
        "precision": precision,
        "trajectories": {}
    }
    for element in trajectories:
        if not element:
            continue
        (name,trajectory) = element
        data["trajectories"][name] = [{"position": [x,y,z], "timestamp" : timestamp} for (x,y,z,timestamp) in trajectory]

    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return data

def save_flight_plans_to_csv(flight_plans, filename='flight_plans.csv'):
    timesteps = sorted({time[0] for (flight_plan,_) in flight_plans for _, time in flight_plan})
    
    csv_data = []
    
    for timestep in timesteps:
        row = [timestep]
        for (flight_plan, _) in flight_plans:
            position = next((position for position, time in flight_plan if time[0] == timestep), None)
            if position:
                row.append((position[0] * VOXEL_SIZE, position[1] * VOXEL_SIZE, position[2] * VOXEL_SIZE))
        csv_data.append(row)
    
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)

def save_matrix_to_file(matrix, filename):
    str_key_dict = {str(key): value for key, value in matrix.items()}
    with open(filename, 'w') as file:
        json.dump(str_key_dict, file)

if __name__ == '__main__':
    """
        A lot of the aspects that are generated randomly can be hardcoded here. What i did is that i let the script first generate these values for me and if i liked them i just saved them in the
        cities.py file by copy and pasting the values from the generated flight_plans.json file. This also allows to for example fix the city layout and uav starting/ending points while variating other aspects
        such as velocity or antenna placement. The trajectories are saved in a separate file because the dimensions of it can reach many gigabytes depending on how long the simulation is and how many
        UAVs are in the simulation. I advice to just save the flight plans and let the script compute the trajectories every time. The antennas placements in the flight_plans.json can be edited using the utility
        antenna_configurator.py which allows to graphically make changes and assess the placements. This allows us to test different type of antenna placements while keeping all of the other aspects of the simulation intact.
        Unfortunately, in case of antenna configuration changes, they need be simulated again. We can do this by fixing all the values for buildings, drone_points, start_times, velocities, flight plans and let the script generate
        trajectories and antennas simulations again. This is because the program will have to also generate the sensing data and remote id data from scratch again. 
    """
    override_city = cities.FULL_COVERAGE_REAL_RID_SENSING_GPS

    print(f'Creating city')
    buildings = generate_city_layout()
    #Trick to skip city generation and reuse a premade one
    #buildings = override_city['buildings']#cities.city_1000_urban_medium_dense['buildings']

    print(f'Creating drone points')
    drone_points = [(random_point(buildings, ground=False), random_point(buildings, ground=False)) for _ in range(NUM_DRONES)]
    #Trick to use the default drone starting/ending points made for the specific city
    #drone_points = override_city['drone_points']#cities.city_1000_urban_medium_dense['drone_points']

    print(f'Creating start times')
    start_times = [random.randint(0, MAX_START_DELAY) for _ in range(NUM_DRONES)]
    #Trick to use the default drone starting/ending points made for the specific city
    #start_times = override_city['start_times']#cities.city_1000_urban_medium_dense['start_times']

    print(f'Creating velocities')
    velocities = [max(min(random.gauss(AVERAGE_VELOCITY, VELOCITY_STANDARD_DEVIATION), HIGHEST_VELOCITY), LOWEST_VELOCITY) for _ in range(NUM_DRONES)]
    #Trick to use the default drone speeds made for the specific city
    #velocities = override_city['velocities']#cities.city_1000_urban_medium_dense['velocities']

    print(f'Creating flight plans')
    flight_plans = []
    drone_args = [(start, end, buildings, start_time, velocity) for (start, end), start_time, velocity in zip(drone_points, start_times, velocities)]
    with Pool(cpu_count()) as pool:
        flight_plans = pool.map(generate_flight_plan, drone_args)


    #print(f'Saving flight plans as CSV')
    formatted_flight_plans = format_flight_plans_to_json(flight_plans, buildings, VOXEL_SIZE, CITY_SIZE, None, None)
    save_flight_plans_to_csv(flight_plans)

    print(f'Creating flight trajectories')
    trajectories = []
    precision = 100
    final_flight_plans = formatted_flight_plans['flight_plans']
    #Trick to use the default flight plans made for the specific city and drone configuration, path planning is deterministic for drones given that their startin points and velocities remain the same
    #But this trick allows us to not wait the iterations of the path planning algorithm which can become lenghty if there are a lot of drones
    #And the upper part of this code needs to be commented
    #final_flight_plans = override_city['flight_plans']
    pu_args = [(flight_plan_id, flight_plan, VOXEL_SIZE, precision) for (flight_plan_id, flight_plan) in final_flight_plans.items()]
    with Pool(cpu_count()) as pool:
        trajectories = pool.map(utils.compute_timestamped_trajectory, pu_args)

    print(f'Saving flight trajectories')
    trajectories = save_flight_trajectories_to_json(trajectories, buildings, VOXEL_SIZE, CITY_SIZE, precision)

    print(f'Creating wifi and tele antennas')
    preferred_wifi_antenna_positions = []
    wifi_antennas, tele_antennas = initialize_antennas(buildings, CITY_SIZE, VOXEL_SIZE, NUM_WIFI_ANTENNAS, preferred_wifi_antenna_positions, TELE_RADIUS, MIN_TELE_ANTENNA_HEIGHT, MAX_TELE_ANTENNA_HEIGHT)
    #Trick to use the default antenna confgurations for the specific city
    #wifi_antennas = override_city['wifi_antennas']
    #wifi_antennas, tele_antennas = override_city['wifi_antennas'],override_city['tele_antennas']


    print(f'Creating RemoteID receiver heatmaps')
    wifi_matrix = compute_wifi_probabilities_profiles(wifi_antennas, CITY_SIZE, buildings, VOXEL_SIZE, 0, VOXEL_SIMULATION_CEILING, WIFI_BUILDING_ATTENUATION, WIFI_FLOOR_NOISE, WIFI_FRAME_LENGTH)
    print(f'Saving RemoteID data')
    generate_RID_dataset(trajectories['trajectories'], wifi_matrix, VOXEL_SIZE)
    print(f'Creating sensing heatmaps')
    tele_matrix = compute_tele_probabilities_profiles(tele_antennas, CITY_SIZE, buildings, VOXEL_SIZE, 0, VOXEL_SIMULATION_CEILING, TELE_ATTENUATION, TELE_FLOOR_NOISE, TELE_FRAME_LENGTH)
    print(f'Size tele matrix {len(tele_matrix)}')
    print(f'Size wifi matrix {len(wifi_matrix)}')

    #Generate after RID because trajectories elements get mutated
    print(f'Saving sensing data')
    generate_sensing_dataset(trajectories['trajectories'], tele_matrix, VOXEL_SIZE)

    print(f'Saving simulation')
    composed_flight_plans = compose_flight_plans_to_be_saved(final_flight_plans, buildings, VOXEL_SIZE, CITY_SIZE, wifi_antennas, tele_antennas, drone_points, start_times, velocities, len(wifi_matrix), len(tele_matrix), [LATERAL_GPS_RNP, VERTICAL_GPS_RNP])
    #print(wifi_matrix)
    if wifi_matrix and len(wifi_matrix)>0:
        save_matrix_to_file(wifi_matrix, 'wifi_matrix.csv')
    if tele_matrix and len(tele_matrix)>0:
        save_matrix_to_file(tele_matrix, 'tele_matrix.csv')

    with open('flight_plans.json', 'w') as json_file:
        json.dump(composed_flight_plans, json_file, indent=4)
