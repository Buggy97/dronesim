import csv
import ast
import numpy as np
import json
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.measures import Mahalanobis, Euclidean
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.reader.base import DetectionReader
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
import scipy.stats as stats
from scipy.spatial.distance import directed_hausdorff, cityblock, euclidean, cosine, minkowski
from fastdtw import fastdtw
from frechetdist import frdist
from math import inf

TIME_BASE = datetime.now()
INTERPOLATED_POINTS = 100

# Parse the CSV data
def parse_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            timestamp = int(row[0])
            detections = [ast.literal_eval(detection) for detection in row[1:]]
            data.append((timestamp, detections))
    return data

# Convert parsed data to StoneSoup detections
class CSVDetectionReader(DetectionReader):
    data = Property(list, doc="List of timestamped detections")

    def __init__(self, data, *args, **kwargs):
        self.start_time = TIME_BASE
        super().__init__(data=data, *args, **kwargs)

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for timestamp, detection_list in self.data:
            detection_time = self.start_time + timedelta(milliseconds=timestamp)
            detections = {Detection(np.array(detection[:3]), timestamp=detection_time) for detection in detection_list}
            yield detection_time, detections

# Main tracking function using StoneSoup
def track_objects_with_stonesoup(data):
    # Convert to StoneSoup detections
    reader = CSVDetectionReader(data=data)

    # Define models
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(10), ConstantVelocity(10), ConstantVelocity(10)])
    measurement_model = LinearGaussian(ndim_state=6, mapping=(0, 2, 4), noise_covar=np.diag([1, 1, 1]))

    # Define predictor and updater
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Define data associator and hypothesiser for the initiator
    measure_initiator = Euclidean()
    hypothesiser_initiator = DistanceHypothesiser(predictor, updater, measure=measure_initiator, missed_distance=5)
    data_associator_initiator = GNNWith2DAssignment(hypothesiser_initiator)

    # Define initiator with high initial uncertainty
    initial_state_covariance = np.diag([10, 10, 10, 10, 10, 10])
    prior_state = GaussianState(np.zeros(6), initial_state_covariance)
    
    # Define a deleter for the initiator
    measurement_deleter = UpdateTimeStepsDeleter(time_steps_since_update=10)#CovarianceBasedDeleter(1)

    initiator = MultiMeasurementInitiator(
        prior_state=prior_state,
        measurement_model=measurement_model,
        deleter=measurement_deleter,
        data_associator=data_associator_initiator,
        updater=updater,
        min_points = 5
    )

    # Define deleter for the tracker
    track_deleter = UpdateTimeStepsDeleter(time_steps_since_update=10)

    # Define data associator and hypothesiser for the tracker
    measure_tracker = Euclidean()  # Use Mahalanobis distance
    hypothesiser_tracker = DistanceHypothesiser(predictor, updater, measure=measure_tracker, missed_distance=10)
    data_associator_tracker = GNNWith2DAssignment(hypothesiser_tracker)

    # Define tracker
    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=track_deleter,
        detector=reader.detections_gen(),
        data_associator=data_associator_tracker,
        updater=updater
    )

    return tracker

# Apply Gaussian smoothing to the tracked paths and discard initial points
def smooth_tracked_paths(tracker, sigma=1, discard_initial=1):
    smoothed_paths = {}
    for time, ctracks in tracker:
        for track in ctracks:
            state = track.state
            track_id = track.id
            position = state.mean[[0, 2, 4]].flatten().tolist()  # Extract and flatten x, y, z positions
            timestamp = state.timestamp
            drone_id = f"computed_path_{track_id}"
            if drone_id not in smoothed_paths:
                smoothed_paths[drone_id] = []
            smoothed_paths[drone_id].append((position, timestamp))

    for drone_id, path in smoothed_paths.items():
        positions = np.array([pos for pos, time in path])
        times = [time for pos, time in path]
        
        if len(positions) > discard_initial:
            positions = positions[discard_initial:]
            times = times[discard_initial:]
        
        if positions.shape[0] > 1:  # Only apply smoothing if there are enough points
            smoothed_positions = gaussian_filter1d(positions, sigma=sigma, axis=0)
            smoothed_paths[drone_id] = [(pos.tolist(), time) for pos, time in zip(smoothed_positions, times)]

    return smoothed_paths

# Output tracked paths to JSON
def format_tracked_paths_json(smoothed_paths):
    flight_plans = {}
    for drone_id, path in smoothed_paths.items():
        flight_plans[drone_id] = [{"position": pos, "timestamp": int((time - TIME_BASE).total_seconds() * 1000)} for pos, time in path]
    return flight_plans

def output_paths_JSON(paths, output_filename='tracked_paths.json'):
    with open(output_filename, 'w') as file:
        json.dump({"flight_plans": paths}, file, indent=4)
    return paths

def parse_ground_truth(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        return data['trajectories']
    
def parse_RID_trajectories():
    with open('RID_dataset.json', 'r') as json_file:
        data = json.load(json_file)
        return data
    
def remove_outliers_from_paths(smoothed_paths, threshold):
    cleaned_paths = {}

    for drone_id, path in smoothed_paths.items():
        cleaned_path = []
        previous_position = None

        for position, timestamp in path:
            if previous_position is None:
                # Add the first point without any check
                cleaned_path.append((position, timestamp))
                previous_position = np.array(position)
            else:
                current_position = np.array(position)
                distance = np.linalg.norm(current_position - previous_position)

                if distance <= threshold:
                    cleaned_path.append((position, timestamp))
                    previous_position = current_position

        cleaned_paths[drone_id] = cleaned_path

    return cleaned_paths

def is_point_equal(p1, p2, position_tolerance, time_tolerance):
    #print(f'Comparing {p1, p2}')
    time1 = p1['timestamp']
    time2 = p2['timestamp']
    x1, y1, z1 = p1['position']
    x2, y2, z2 = p2['position']
    return (((time2 - time_tolerance) <= time1 <= (time2 + time_tolerance)) and ((x2 - position_tolerance) <= x1 <= (x2 + position_tolerance)) and ((y2 - position_tolerance) <= y1 <= (y2 + position_tolerance)) and ((z2 - position_tolerance) <= z1 <= (z2 + position_tolerance)))

def interpolate_trajectory(points, num_points):
    points = np.array(points)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points[:, :3], axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    interpolator = interp1d(distance, points, kind='linear', axis=0)
    uniform_distances = np.linspace(0, 1, num_points)
    interpolated_points = interpolator(uniform_distances)
    return interpolated_points

def calculate_rmse(coords1, coords2):
    return np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))

def calculate_max_error(coords1, coords2):
    return np.max(np.sqrt(np.sum((coords1 - coords2)**2, axis=1)))

def calculate_hausdorff_distance(coords1, coords2):
    return max(directed_hausdorff(coords1, coords2)[0], directed_hausdorff(coords2, coords1)[0])

def calculate_cosine_similarity(coords1, coords2):
    tangents1 = np.diff(coords1, axis=0)
    tangents2 = np.diff(coords2, axis=0)
    dot_products = np.sum(tangents1 * tangents2, axis=1)
    magnitudes = np.sqrt(np.sum(tangents1**2, axis=1)) * np.sqrt(np.sum(tangents2**2, axis=1))
    cosine_similarities = dot_products / magnitudes
    return np.mean(cosine_similarities)

def calculate_dtw(coords1, coords2):
    distance, _ = fastdtw(coords1, coords2, dist=lambda x, y: minkowski(x, y, p=1.5))
    return distance

def calculate_edr(coords1, coords2, epsilon_space=0.1, epsilon_time=0.1):
    len1, len2 = len(coords1), len(coords2)
    dp = np.zeros((len1+1, len2+1))

    for i in range(1, len1+1):
        dp[i][0] = i
    for j in range(1, len2+1):
        dp[0][j] = j
    
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            spatial_dist = np.linalg.norm(coords1[i-1][:3] - coords2[j-1][:3])
            temporal_dist = abs(coords1[i-1][3] - coords2[j-1][3]) if coords1.shape[1] == 4 else 0
            if spatial_dist < epsilon_space and temporal_dist < epsilon_time:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    
    return dp[len1][len2]

def calculate_lcss(coords1, coords2, epsilon_space=0.1, epsilon_time=0.1):
    len1, len2 = len(coords1), len(coords2)
    dp = np.zeros((len1+1, len2+1))

    for i in range(1, len1+1):
        for j in range(1, len2+1):
            spatial_dist = np.linalg.norm(coords1[i-1][:3] - coords2[j-1][:3])
            temporal_dist = abs(coords1[i-1][3] - coords2[j-1][3]) if coords1.shape[1] == 4 else 0
            if spatial_dist < epsilon_space and temporal_dist < epsilon_time:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return len1 + len2 - 2 * dp[len1][len2]

def calculate_discrete_frechet_distance(coords1, coords2):
    return frdist(coords1, coords2)


def get_minimum_distance(point, track):
    point = np.array(point)
    track = np.array(track)
    squared_differences = (track - point) ** 2
    rmse_values = np.sqrt(np.mean(squared_differences, axis=1))

    # Find the smallest RMSE
    min_rmse = np.min(rmse_values)
    return min_rmse


def processed_and_interpolated(short, long):
    a = [(*point["position"],point["timestamp"]) for point in short]
    b = [(*point["position"],point["timestamp"]) for point in long]
    trajectory1_interpolated = interpolate_trajectory(a, INTERPOLATED_POINTS)
    trajectory2_interpolated = interpolate_trajectory(b, INTERPOLATED_POINTS)
    return trajectory1_interpolated, trajectory2_interpolated

#Given the long track and short track compares how many points % of the short track are in the long track considering spatio-temporal datapoints
def compute_error_rate(short, long):
    trajectory1_interpolated, trajectory2_interpolated = processed_and_interpolated(short, long)
    
    # Spatial and temporal metrics
    rmse_spatiotemporal = calculate_rmse(trajectory1_interpolated, trajectory2_interpolated)    
    return rmse_spatiotemporal

def rid_compute_error_rate(short, long):
    #We put centiseconds for rid because we want to prioritize the distance more than the timedifference, and even small timestamp difference of 1 second can make the error too big
    a = [(*point["position"],point["timestamp"]/100) for point in short]
    b = [(*point["position"],point["timestamp"]/100) for point in long]
    trajectory1_interpolated = interpolate_trajectory(a, INTERPOLATED_POINTS)
    trajectory2_interpolated = interpolate_trajectory(b, INTERPOLATED_POINTS)
    
    # Spatial and temporal metrics
    rmse_spatiotemporal = calculate_rmse(trajectory1_interpolated, trajectory2_interpolated)    
    return rmse_spatiotemporal

def match_rate_matrix(tracks, ground_truth):
    matrix = {}
    for id1, track1 in tracks.items():
        for id2, track2 in ground_truth.items():
            if(len(track1)<=1 or len(track2)<=1):
                continue
            #print(f'Examining {id1, id2} {len(track1), len(track2)}')
            matrix[(id2, id1)] = compute_error_rate(track2, track1)
    return matrix

def rid_match_rate_matrix(tracks, rid_tracks):
    matrix = {}
    for id1, track1 in tracks.items():
        for id2, track2 in rid_tracks.items():
            if(len(track1)>=2 and len(track2)>=2):
                matrix[(id2, id1)] = rid_compute_error_rate(track2, track1)
            else:
                a = [(*point["position"],point["timestamp"]/100) for point in track1]
                b = [(*point["position"],point["timestamp"]/100) for point in track2]
                if(len(track1)==0 or len(track2)==0):
                    continue
                elif(len(track1)==1 and len(track2)==1):
                    matrix[(id2, id1)] = get_minimum_distance(a[0], [b[0]])
                elif(len(track1)==1):
                    matrix[(id2, id1)] = get_minimum_distance(a[0], b)
                elif(len(track2)==1):
                    matrix[(id2, id1)] = get_minimum_distance(b[0], [a[0]])
            #print(f'Examining {id1, id2} {len(track1), len(track2)}')
    return matrix


def fittest_computed_track_for_truetrack(rmse_matrix, true_track_id):
    lowest_error = 99999999999999999999999999999999999999999999999999999
    fittest_computed_track = ''
    for (ttid, ctid), rmse in rmse_matrix.items():
        if ttid==true_track_id:
            if rmse < lowest_error:
                lowest_error = rmse
                fittest_computed_track = ctid
    return fittest_computed_track

def fittest_true_track_for_computedtrack(rmse_matrix, computed_track_id):
    lowest_error = 99999999999999999999999999999999999999999999999999999
    fittest_true_track = ''
    for (ttid, ctid), rmse in rmse_matrix.items():
        if ctid==computed_track_id:
            if rmse < lowest_error:
                lowest_error = rmse
                fittest_true_track = ttid
    return fittest_true_track

def get_track_pairings(rmse_matrix, maximum_average_rmse = 10):
    track_associations = {}
    faulty_associations = {}
    uniques_ttid = set([pair[0] for pair in rmse_matrix.keys()])
    for ttid in uniques_ttid:
        fittest_computedtrack = fittest_computed_track_for_truetrack(rmse_matrix, ttid)
        if fittest_computedtrack=='':
            continue
        if (float(rmse_matrix[ttid, fittest_computedtrack]) < (maximum_average_rmse*INTERPOLATED_POINTS)):
            track_associations[ttid, fittest_computedtrack] = float(rmse_matrix[ttid, fittest_computedtrack])
        else:
            faulty_associations[ttid, fittest_computedtrack] = float(rmse_matrix[ttid, fittest_computedtrack])
    return track_associations, faulty_associations

def print_simulation_info_for(filename):
    retmap = {}
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        name = data.get("name", "")
        description = data.get("description", "")
        total_buildings = len(data.get("buildings", []))
        total_rid_receivers = len(data.get("wifi_antennas", []))
        total_base_stations = len(data.get("tele_antennas", []))
        simulation_flight_plans = data.get("flight_plans", [])
        total_UAVS = len([simulation_flight_plan for simulation_flight_plan in simulation_flight_plans.values() if len(simulation_flight_plan)>0])
        simulation_voxel_size = data.get("voxel_size", 20)
        simulation_city_size_array = data.get("city_size", [])
        simulation_city_size = f'{simulation_city_size_array[0]} X {simulation_city_size_array[1]} X {simulation_city_size_array[2]} (m)'
        total_airspace_voxels = (simulation_city_size_array[0] // simulation_voxel_size) * (simulation_city_size_array[2] // simulation_voxel_size) * (140 // simulation_voxel_size)
        rid_receiver_coverage = data.get("wifi_coverage", 0)
        rid_receiver_coverage =  rid_receiver_coverage / total_airspace_voxels
        sensing_coverage = data.get("tele_coverage", 0)
        sensing_coverage =  sensing_coverage / total_airspace_voxels
        gnss_rnp = data.get("gnss_rnp", [0,0])

        print(f'Info about the simulation: {name}')
        print(f'General description: {description}')
        print(f'There are {total_rid_receivers} RID receivers')
        print(f'There are {total_base_stations} ISAC enabled base stations')
        print(f'There are {total_UAVS} UAVs flying')
        print(f'Scenario size is: {simulation_city_size}')
        print(f'GNSS RNP is {gnss_rnp[0]}m lateral and {gnss_rnp[1]}m vertical')
        print(f'RID receiver cover ~{int(rid_receiver_coverage*100)}% of the UAV airspace')
        print(f'ISAC base stations cover ~{int(sensing_coverage*100)}% of the UAV airspace')

        retmap['total_UAVS'] = total_UAVS

    return retmap
    

# Main execution
filename = 'sensing_dataset.csv'
params = print_simulation_info_for('flight_plans.json')
data = parse_csv(filename)
tracker = track_objects_with_stonesoup(data)
smoothed_paths = smooth_tracked_paths(tracker)
threshold_distance = 20
cleaned_paths = remove_outliers_from_paths(smoothed_paths, threshold_distance)
formatted_paths = format_tracked_paths_json(cleaned_paths)
output_paths_JSON(formatted_paths)
ground_truth = parse_ground_truth('trajectories.json')
rid_data_set = parse_RID_trajectories()

#Compute RMSE for each pair of ground truth and cleaned_paths
rmse_matrix = match_rate_matrix(formatted_paths, ground_truth)
#For each ground truh track associate it to fittest cleaned_path
true_to_computed_associations, ttcfaulty = get_track_pairings(rmse_matrix, 1000)
print(f'Correctly predicted {len(true_to_computed_associations)}/{params["total_UAVS"]} trajectories')
print(f'{len(ttcfaulty)} faulty trajectories (average RMSE > 1000)')
#print(true_to_computed_associations)
#print(rmse_matrix.keys())
#For each pair computer:
#Distance between A[0], B[0], distance between A[-1],B[-1], get the higest distance -> Initial and final position accuracy precision

precisions = []
rmses = []
dtws = []
normalized_dtws = []
normalized_rmses = []
for (ttid, ctid), rmse in true_to_computed_associations.items():
    true_track = ground_truth[ttid]
    computed_track = formatted_paths[ctid]
    trajectory1_interpolated, trajectory2_interpolated = processed_and_interpolated(true_track, computed_track)

    #Precision calc
    ttstart, ttend = np.array(true_track[0]["position"]), np.array(true_track[-1]["position"])
    ctstart, ctend = np.array(computed_track[0]["position"]), np.array(computed_track[-1]["position"])
    if(np.linalg.norm(ttstart-ttend)<=0):
        continue
    precision = max(np.linalg.norm(ttstart-ctstart), np.linalg.norm(ttend-ctend))
    dtw = calculate_dtw(trajectory1_interpolated[:, :3], trajectory2_interpolated[:, :3])
    precisions.append(precision)
    rmses.append(rmse)
    dtws.append(dtw)
    normalized_dtws.append(dtw/INTERPOLATED_POINTS)
    normalized_rmses.append(rmse/INTERPOLATED_POINTS)
    #print(f'TT: {ttstart, ttend}, CT: {ctstart, ctend}, PRECISON: {precision}')

def print_measurements_statistics(measurements):
    print(f'\tmean: {np.mean(measurements)}m')
    print(f'\tmedian: {np.median(measurements)}m')
    print(f'\tstd_dev: {np.std(measurements)}m')
    print(f'\tvariance: {np.var(measurements)}m')
    print(f'\trange: {np.min(measurements)}, {np.max(measurements)}m')
    print('\tpercentile: ')
    print(f'\t\t10: {np.percentile(measurements, 10)}m')
    print(f'\t\t25: {np.percentile(measurements, 25)}m') 
    print(f'\t\t50: {np.percentile(measurements, 50)}m') 
    print(f'\t\t75: {np.percentile(measurements, 75)}m') 
    print(f'\t\t90: {np.percentile(measurements, 90)}m') 
    print(f'\t\t95: {np.percentile(measurements, 95)}m') 
    print(f'\t\t99: {np.percentile(measurements, 99)}m')

#percentiles = np.percentile(measurements, [10, 25, 50, 75, 90])
print('\n#################################################################')
print('Statistics for computed to ground truth associations')
print('Precision for start and end detection (worse):')
print_measurements_statistics(precisions)
print('RMSE between predicted and ground truth:')
print_measurements_statistics(rmses)
print('Normalized RMSE between predicted and ground truth:')
print_measurements_statistics(normalized_rmses)
print('DTW distance:')
print_measurements_statistics(dtws)
print('Normalized DTW distance:')
print_measurements_statistics(normalized_dtws)
print('#################################################################\n')

#RMSE (get get from the first computation) -> Tracking accuracy metric
#DTW score -> Tracking accuracy metric

#Computer RMSE for each pair of rid_paths and cleaned_paths
rid_rmse_matrix = rid_match_rate_matrix(formatted_paths, rid_data_set)
rid_to_computed_associations, rtcfaulty = get_track_pairings(rid_rmse_matrix, 100000)#get_track_pairings(rid_rmse_matrix)
#print(rid_to_computed_associations)

#For each cleaned_path associate it to the fittest rid_path -> RID-Sensing Association
#For each (cleaned_path, rid_path) check if (ground_truth, cleaned_path) in the previous pairings is that ground_truth==rid_path count this -> Number of correct RID associations
correct_rid_associtations = 0
for (ttid, ctid), rmse in true_to_computed_associations.items():
    for (rtid, ctid2), rmse2 in rid_to_computed_associations.items():
        if(ctid==ctid2):
            #print(f'Computer track {ctid} is associated to RID track {rtid} and to true track {ttid}')
            if ttid==rtid:
                correct_rid_associtations+=1
print(f'Correct rid associations: {correct_rid_associtations}/{params["total_UAVS"]}')
print(f'{len(ttcfaulty)} faulty associations (average RMSE > 100000)')
#How much does a calculated path match with the original one
#mat = match_rate_matrix(saved_paths, ground_truth, 5, 10)
#mat2 = match_rate_matrix(saved_paths, rid_data_set, 5, 1000) #Time tolerance set to 1000ms=1s because RID is generated every 1 second and sensing very 0.1s

#Get top len()

#print(mat)
#print(mat2)

output_paths_JSON(formatted_paths)
print(f'Created {len(formatted_paths)} tracks')
print(f'Ground truth tracks {len([track for _,track in ground_truth.items() if (len(track)>0)])}')