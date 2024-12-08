#Dependency used by the gen_flight_path script, do not remove this

import numpy as np
from scipy.interpolate import make_interp_spline, CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy.special import comb
from scipy.special import erfc
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import math
import random
    
# Step 1: Parse the input data
#velocities = np.array([point["velocity"] for point in path])
#entry_times = np.array([point["entry_time"] for point in path])
#exit_times = np.array([point["exit_time"] for point in path])

# Step 3: Trajectory Pruning using Douglas-Peucker Algorithm (Optional)
def douglas_peucker(points, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points)
    for i in range(1, end - 1):
        d = np.linalg.norm(np.cross(points[end-1] - points[0], points[0] - points[i])) / np.linalg.norm(points[end-1] - points[0])
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        rec_results1 = douglas_peucker(points[:index + 1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)

        # Build the result list
        result = np.vstack((rec_results1[:-1], rec_results2))
    else:
        result = np.vstack((points[0], points[end-1]))

    return result

def linear_interpolation(path, n_points):
    t = np.linspace(0, 1, len(path))
    linear_interp = interp1d(t, path, axis=0)
    t_smooth = np.linspace(0, 1, n_points)
    smooth_path = linear_interp(t_smooth)
    return smooth_path

def bezier_curve(path, nPoints=300):
    def bernstein_poly(i, n, t):
        return comb(n, i) * (t**(n-i)) * ((1 - t)**i)

    n = len(path) - 1
    t = np.linspace(0, 1, nPoints)
    polynomial_array = np.array([bernstein_poly(i, n, t) for i in range(0, n+1)]).T

    xvals = np.dot(polynomial_array, path[:, 0])
    yvals = np.dot(polynomial_array, path[:, 1])
    zvals = np.dot(polynomial_array, path[:, 2])

    return np.vstack((xvals, yvals, zvals)).T

def calculate_path_length(path):
    distances = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
    total_length = np.sum(distances)
    return total_length

def generate_timed_trajectory(trajectory, t_start, t_end):
    timestamps = linear_interpolation([t_start, t_end], len(trajectory))
    timed_trajectory =  np.column_stack((trajectory, timestamps[::-1]))
    return timed_trajectory

#This is what  a function shoul call
#Precision to 100 mean that the trajectory generated will have a precision to 1cm, 1000 is 1mm and so on
#Higher precision requires more time for computation, so try to avoid.
def compute_timestamped_trajectory(args):
    name, path, voxel_size, precision = args
    print(f'Computing trajectory for {name}')
    if not path: 
        return (name, [])
    positions = np.array([[point["position"][0]+voxel_size/2, point["position"][1]+voxel_size/2, point["position"][2]+voxel_size/2] for point in path])
    pruned_path = douglas_peucker(positions, voxel_size-(voxel_size/4))#np.ceil(voxel_size/np.sqrt(3)))
    rich_initial_path = linear_interpolation(pruned_path, int(len(path)*voxel_size/2))
    smooth_path = bezier_curve(rich_initial_path, nPoints=(len(positions)*voxel_size*precision))
    #plot_paths(smooth_path, pruned_path, voxel_size, positions) For testing
    t_start = path[0]['entry_time']
    t_end = path[-1]['exit_time']
    timestamped_trajectory = generate_timed_trajectory(smooth_path, t_start, t_end)
    return (name, timestamped_trajectory[::-1])

#print(smooth_path)
#path_length = calculate_path_length(smooth_path)
#print(f'Length of flight plan {len(positions)}, Length of trajectories {len(positions)*voxel_size*100}, Length of trajectories {len(smooth_path)}')
#print(f'Trajectory length: {path_length}')

def plot_paths(smooth_path, pruned_path, voxel_size, positions):
    # Function to draw a voxel
    def draw_voxel(ax, center, size, color='k', alpha=0.1):
        half_size = size / 2
        corners = np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, -1], [-1, -1, 1]]) * half_size
        corners += center
        vertices = [[corners[j] for j in [0, 1, 2, 3]], [corners[j] for j in [4, 5, 6, 7]], [corners[j] for j in [0, 3, 7, 4]], [corners[j] for j in [1, 2, 6, 5]], [corners[j] for j in [0, 1, 5, 4]], [corners[j] for j in [2, 3, 7, 6]]]
        poly = Poly3DCollection(vertices, facecolors=color, edgecolors='k', alpha=alpha)
        ax.add_collection3d(poly)

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2], 'magenta', label='Smooth Path Bezier')
    ax.plot(pruned_path[:, 0], pruned_path[:, 1], pruned_path[:, 2], 'g-o', label='Pruned Path')
    for point in positions:
        draw_voxel(ax, point, voxel_size, color='b', alpha=0.1)

    ax.legend()
    plt.show()

#compute_timestamped_trajectory(('ciao', path, 20, 100)) #For thesting


def random_point_on_top_surface(building, preference=None):
        x, y, z = building["x"], building["y"], building["z"]
        width, depth, height = building["width"], building["depth"], building["height"]
        
        # Choose a random face of the building
        choices = ["front", "back", "left", "right", "top"]
        face = ''
        if preference and preference in choices:
            face = preference
        else:
            face = random.choice(choices)
        print(face)
        if face == "front":
            # Front face
            random_x = random.uniform(x, x + width)
            random_y = y + height + 1
            random_z = z + 1
        elif face == "back":
            # Back face
            random_x = random.uniform(x, x + width)
            random_y = y + height + 1
            random_z = z + depth + 1
        elif face == "left":
            # Left face
            random_x = x + 1
            random_y = y + height + 1
            random_z = random.uniform(z, z + depth)
        elif face == "right":
            # Right face
            random_x = x + width + 1
            random_y = y + height + 1
            random_z = random.uniform(z, z + depth)
        elif face == "top":
            # Top face
            random_x = random.uniform(x, x + width)
            random_y = y + height + 1
            random_z = random.uniform(z, z + depth)
        
        return random_x, random_y, random_z

def calculate_optimal_points(plane_size, radius):
    """
    Calculate the optimal number of points to cover a plane of given size with circles of a given radius.
    
    Parameters:
    plane_size (int): The size of the plane (assumed to be square).
    radius (float): The radius of the circles.
    
    Returns:
    List of tuples: The points.
    """
    # Calculate the distance between points in a hexagonal grid
    h_dist = 2 * radius
    v_dist = np.sqrt(3) * radius

    # Generate points in a hexagonal grid
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
                        for x in range(x_start, x_end+1)
                        for y in range(y_start, y_end+1) 
                        for z in range(z_start, z_end+1)]
    return voxel_coordinates

def create_voxel_city(city_size, buildings, voxel_size):

    #Create a 3d model where obstacles are modeled after a city
    refined_buildings = [building for building in buildings if building['height']>1]
    print(refined_buildings)
    a = math.ceil(city_size[0]/voxel_size)
    b = math.ceil(city_size[1]/voxel_size)
    c = math.ceil(city_size[2]/voxel_size)
    voxel_city = [[[0 for _ in range(a)] for _ in range(b)] for _ in range(c)]
    for building in refined_buildings:
        occupied_voxels = occupied_voxels(building, voxel_size)
        for ov in occupied_voxels:
            print(f'OV: {ov}')
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

def simulate_radio_antenna(voxel_city, profile, source, frequency, initial_power, voxel_size=20, max_heigth=10, attenuation=20, sensitivity=-90):
    profile = {}
    c = 3e8  # Speed of light in m/s
    f = frequency  # Frequency in Hz
    def path_loss(distance, loss_exponent=2):
        # Free Space Path Loss (FSPL) formula
        if distance == 0:
            return initial_power
        return initial_power - (20 * np.log10(distance) + 20 * np.log10(f) + 20 * np.log10(4 * np.pi / c) + loss_exponent * 10 * np.log10(distance / 1.0))
    
    source_x, source_y, source_z = source
    comp_min = min(max_heigth+1, len(voxel_city[1]))
    for x in range(0,len(voxel_city)):
        for y in range(0,comp_min):
            for z in range(0,len(voxel_city)):
                print(f'Processing {x,y,z}')
                distance = np.linalg.norm(source - np.array((x, y, z))) * voxel_size
                power = path_loss(distance)
                if power > sensitivity:
                    # Apply attenuation due to obstacles
                    line_of_sight = bresenham_3d((source_x, source_y, source_z), (x, y, z))
                    for point in line_of_sight:
                        p_x, p_y, p_z = point
                        p_x, p_y, p_z = int(p_x), int(p_y), int(p_z)
                        power -= voxel_city[p_x][p_y][p_z]*attenuation
                        if power <= 0:
                            break
                if power > sensitivity:
                    print('Added')
                    profile[(x,y,z)] = power
    return profile


def pickup_probability_matrix(power_profiles, noise_floor=-100, frame_length_bits=200, probability_matrix = {}):

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
        return Q(math.sqrt(2 * snr_linear))

    # Calculate PER
    def calculate_per(ber, frame_length):
        return 1 - (1 - ber) ** frame_length

    # Calculate Probability of Successful Reception
    def calculate_success_probability(per):
        return 1 - per
    
    for profile in power_profiles:
        for pos, power in profile.items():
            current_probability = probability_matrix.get(pos, 0)
            _, snr_linear = calculate_snr(power, noise_floor)
            ber = calculate_ber(snr_linear)
            per = calculate_per(ber, frame_length_bits)
            success_probability = calculate_success_probability(per)
            probability_matrix[pos] = overall_pickup_probability(current_probability, success_probability)
    
    return probability_matrix
