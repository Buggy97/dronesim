#Dependency used by the display_statistics script, do not remove this


import numpy as np
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

def normalize_trajectory(traj, num_points):
    """ Normalize a trajectory to have a fixed number of points """
    traj = np.array(traj, dtype=np.float64)
    length = len(traj)
    x = np.linspace(0, length - 1, num=length, dtype=np.float64)
    f = interp1d(x, traj, axis=0, kind='linear')
    x_new = np.linspace(0, length - 1, num=num_points, dtype=np.float64)
    return f(x_new)

def trajectory_match_rate_dtw(traj1, traj2, threshold=25.0):
    """ Calculate the match rate of two trajectories using DTW and return non-matching points """
    traj1 = [point['position'] for point in traj1]
    traj2 = [point['position'] for point in traj2]
    
    # Normalize trajectories to have the same number of points
    num_points = max(len(traj1), len(traj2))
    traj1_normalized = normalize_trajectory(traj1, num_points)
    traj2_normalized = normalize_trajectory(traj2, num_points)
    
    # Calculate DTW distance and path
    distance, path = fastdtw(traj1_normalized, traj2_normalized, dist=euclidean)
    
    # Calculate match rate
    num_matches = 0
    non_matching_points = []
    
    for (i, j) in path:
        dist = np.linalg.norm(traj1_normalized[i] - traj2_normalized[j])
        if dist <= threshold:
            num_matches += 1
        else:
            non_matching_points.append({
                'index1': i,
                'index2': j,
                'point1': traj1_normalized[i].tolist(),
                'point2': traj2_normalized[j].tolist(),
                'distance': dist
            })
    
    match_rate = num_matches / len(path)
    
    return match_rate, non_matching_points

def get_diff_points(traj1, traj2):
    traj1 = [point['position'] for point in traj1]
    traj2 = [point['position'] for point in traj2]
    trajB = traj1
    trajS = traj2
    diff_points = []
    if(len(traj1) >= len(traj2)):
        trajB = traj1
        trajS = traj2
    else:
        trajB = traj2
        trajS = traj1
    for point in trajB:
        if(point not in trajS):
            diff_points.append(point)
    return diff_points

def trajectories_start_match_rate(traj1, traj2, threshold=1.0):
    position1 = traj1[0]['position']
    position2 = traj2[0]['position']

    # Ensure the positions are numpy arrays
    position1 = np.array(position1, dtype=np.float64)
    position2 = np.array(position2, dtype=np.float64)
    distance = np.linalg.norm(position1 - position2)
    if (distance <= threshold): 
        return 1 
    else: 
        return 0
    
def trajectories_end_match_rate(traj1, traj2, threshold=1.0):
    position1 = traj1[-1]['position']
    position2 = traj2[-1]['position']

    # Ensure the positions are numpy arrays
    position1 = np.array(position1, dtype=np.float64)
    position2 = np.array(position2, dtype=np.float64)
    distance = np.linalg.norm(position1 - position2)
    if (distance <= threshold): 
        return 1 
    else: 
        return 0

