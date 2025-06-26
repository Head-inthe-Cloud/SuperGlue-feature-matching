import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from utils.utils import *
import time
import matplotlib.cm as cm
from matplotlib.widgets import Button
from scipy.spatial.distance import cdist


def draw_segments(planes):
    if isinstance(planes[0], Plane):
        planes = [plane.to_2D() for plane in planes]
    
    # Create the plot
    for seg in planes:
        plt.plot(*zip(*seg), color='black')
    plt.axis('equal')
    plt.show()

def visualize(segments_1=None, 
              segments_2=None, 
              segments_3=None, 
              l1=None, 
              l2=None, 
              L1=None, 
              L2=None, 
              T=None, 
              points_1=None, 
              points_2=None,
              points_group=None,
              point_1=None,
              point_2=None,
              heatmap_1 = None,
              heatmap_2 = None,
              img_1 = None,
              img_2 = None,
              title=None, 
              output_path=None,
              block=True):
    # x_values = np.array([x for seg in map_data for x, _ in seg])
    # y_values = np.array([y for seg in map_data for _, y in seg])

    # # Calculate the center of the data
    # center_x = sum(x_values) / len(x_values)
    # center_y = sum(y_values) / len(y_values)

    # Create the plot
    if segments_1 is not None:
        for seg in segments_1:
            plt.plot(*zip(*seg), color='black')

    if segments_2 is not None:
        for seg in segments_2:
            plt.plot(*zip(*seg), color='purple')

    if segments_3 is not None:
        for seg in segments_3:
            plt.plot(*zip(*seg), color='orange')

    if l1 is not None and l2 is not None:
        if T is not None: 
            for seg in apply_transformation_to_segments(np.array([l1, l2]), T):
                plt.plot(*zip(*seg), color='red')
        else: 
            for seg in [l1, l2]:
                plt.plot(*zip(*seg), color='red')


    if L1 is not None and L2 is not None:
        for seg in [L1, L2]:
            plt.plot(*zip(*seg), color='green')

    if l1 is not None and L1 is not None:
        plt.plot(*zip(*l1), color='red')
        plt.plot(*zip(*L1), color='green')
    
    if points_1 is not None:
        plt.plot(*zip(*points_1), 'ro', markersize=1)
    
    if points_2 is not None:
        plt.plot(*zip(*points_2), 'go', markersize=1)

    if points_group is not None:
        assert len(points_group) <= 4, "We can only handle 4 groups right now"
        colors=['orange', 'blue', 'gray', 'purple']
        for i in range(len(points_group)):
            plt.plot(*zip(*points_group[i]), 'o', color=colors[i], markersize=1)

    
    # if points_1 is not None and points_2 is not None:
    #     for p1, p2 in zip(points_1, points_2):
    #         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green')

    if point_1 is not None:
        plt.plot(point_1[0], point_1[1], 'bo', markersize=2)

    if point_2 is not None:
        plt.plot(point_2[0], point_2[1], 'o', color='orange', markersize=2)

    if img_1 is not None:
        plt.imshow(img_1, origin='lower')
    
    if img_2 is not None:
        plt.imshow(img_2, origin='lower')

    if heatmap_1 is not None:
        im = plt.imshow(heatmap_1, cmap='plasma', origin='lower', alpha=0.5)
        plt.colorbar(im, label="Heatmap Intensity")
    
    if heatmap_2 is not None:
        im = plt.imshow(heatmap_2, cmap='plasma', origin='lower', alpha=0.5)
        plt.colorbar(im, label="Heatmap Intensity")

    if title is not None:
        plt.title = title

    plt.axis('equal')
    if output_path is not None:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show(block=block)


def to_binary(segments):
    # Method from PF code, turns map into binary
    #################################
    x_values = np.array([x for seg in segments for x, _ in seg])
    y_values = np.array([y for seg in segments for _, y in seg])

    cell_size = 0.1
    width = np.max(x_values) - np.min(x_values)
    height = np.max(y_values) - np.min(y_values)
    
    num_x_cell = int(np.ceil(width/cell_size))
    num_y_cell = int(np.ceil(height/cell_size))

    numpy_map = np.zeros((num_y_cell, num_x_cell))

    BH = Bresenham(cell_size)

    for seg in segments:
        x_list, y_list = BH.seg(seg[0][0], seg[0][1], seg[1][0], seg[1][1])
        for x, y in zip(x_list, y_list):
            numpy_map[int(y), int(x)] = 1

    plt.plot(numpy_map)
    plt.show()
    #################################


def visualize_array(data, bins=20):
    # Create a histogram
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Data')
    plt.grid(True)

    # Show the plot
    plt.show()


# Visualizes the input orientations as line segments
# Input: 
#       orientations: list of orientations, -pi/2 ~ pi/2
#       center: a pair of x, y coordinate for the center of the returned line segments
#       length: the length of the line segments
# Return: a numpy array of line segments that can be used with the other visualization codes, shape (n x 2 x 2)
def visualize_orientations(orientations, center=[0, 0], length=10):
    segments = []
    for ori in orientations:
        x = np.cos(ori)
        y = np.sin(ori)
        x1 = x * length + center[0]
        y1 = y * length + center[1]
        x2 = -x * length + center[0]
        y2 = -y * length + center[1]
        segments.append([[x1, y1], [x2, y2]])

    return np.array(segments)


def distance_vs_dispersion():
    """
    Plot the proportion of mean points that are within a certain distance threshold from the gt, when different dispersion 'cut off's are used
    """
    exp_path = './results/ARKit vs. RoNIN/num_particle_750/ARKit'

    distance_threshold = 1

    dispersions = []
    distances = []
    
    run_paths = [os.path.join(exp_path, run_name) for run_name in os.listdir(exp_path)]
    run_paths = [run_path for run_path in run_paths if os.path.isdir(run_path)]
    for run_path in run_paths:
        ori_paths = [os.path.join(run_path, ori_name) for ori_name in os.listdir(run_path)]
        ori_paths = [ori_path for ori_path in ori_paths if os.path.isdir(ori_path)]
        for ori_path in ori_paths:
            json_path = os.path.join(ori_path, 'dispersion_vs_distance.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as file:
                data = json.load(file)
                dispersions += data['dispersion']
                distances += data['distance']

    assert len(dispersions) == len(distances), "Error, lengths of dispersion data and distance data don't match"
    dispersions = np.array(dispersions)
    distances = np.array(distances)

    # plt.scatter(dispersions, distances)
    # plt.title("Particle Dispersions vs. Error")
    # plt.xlabel("particle dispersion")
    # plt.ylabel("distance from mean location to gt. (m)")
    # plt.show()

    x_data = np.arange(0, np.max(dispersions), 0.1)

    y_data = []
    for dispersion_threshold in x_data:
        indices = np.where(dispersions < dispersion_threshold)
        thresholded_distances = distances[indices]
        if len(thresholded_distances) == 0:
            proportion = 0
        else:
            # print(len(thresholded_distances), np.count_nonzero(thresholded_distances < distance_threshold), np.max(thresholded_distances))
            proportion = np.count_nonzero(thresholded_distances < distance_threshold) / len(thresholded_distances)
        y_data.append(proportion)
    
    y_data = np.array(y_data)

    max_y_idx = np.argmax(y_data)
    max_y = y_data[max_y_idx]
    vert_line_x = x_data[max_y_idx]

    plt.plot(x_data, y_data)

    # plt.axvline(x=vert_line_x, ymax=max_y, color='r', linestyle='--', linewidth=1, label=f'x = {vert_line_x}')
    # plt.text(vert_line_x + 0.2, 0, f'x = {vert_line_x:.2f}', color='r', verticalalignment='bottom')
    # plt.axhline(y=max_y, xmin=0, xmax=vert_line_x, color='r', linestyle='--', linewidth=1, label=f'y = {max_y}')

    plt.plot(vert_line_x, max_y, 'ro')
    plt.text(vert_line_x+0.5, max_y, f'({vert_line_x:.2f}, {max_y:.2f})', color='r')

    plt.title(f"Proportion of Prediction Error < {distance_threshold}m under Different Dispersion Thresholds")
    plt.xlabel("Dispersion threshold")
    plt.ylabel(f"proportion of prediction error < {distance_threshold}m")
    plt.show()


def avg_error_vs_convergence_criteria():
    """
    For different dispersion thresholds used as convergence criteria, measure the average error (distance) after that threshold
    """
    exp_paths = ['./results/ARKit vs. RoNIN/num_particle_500/RoNIN',
                 './results/ARKit vs. RoNIN/num_particle_750/RoNIN',
                 './results/ARKit vs. RoNIN/num_particle_1000/RoNIN']

    for exp_path in exp_paths:
        dispersions = []
        distances = []
        
        run_paths = [os.path.join(exp_path, run_name) for run_name in os.listdir(exp_path)]
        run_paths = [run_path for run_path in run_paths if os.path.isdir(run_path)]
        for run_path in run_paths:
            ori_paths = [os.path.join(run_path, ori_name) for ori_name in os.listdir(run_path)]
            ori_paths = [ori_path for ori_path in ori_paths if os.path.isdir(ori_path)]
            for ori_path in ori_paths:
                json_path = os.path.join(ori_path, 'dispersion_vs_distance.json')
                if not os.path.exists(json_path):
                    continue
                with open(json_path, 'r') as file:
                    data = json.load(file)
                    dispersions += data['dispersion']
                    distances += data['distance']

        assert len(dispersions) == len(distances), "Error, lengths of dispersion data and distance data don't match"
        dispersions = np.array(dispersions)
        distances = np.array(distances)

        # plt.scatter(dispersions, distances)
        # plt.title("Particle Dispersions vs. Error")
        # plt.xlabel("particle dispersion")
        # plt.ylabel("distance from mean location to gt. (m)")
        # plt.show()

        x_data = np.arange(0, 20, 0.01)

        y_data = []
        for dispersion_threshold in x_data:
            avg_error = np.inf
            convergence_index = None
            for i in range(len(dispersions)):
                if dispersions[i] <= dispersion_threshold:
                    convergence_index = i
                    break
            if convergence_index is not None:
                n = len(distances) - convergence_index
                avg_error = np.sum(distances[convergence_index:]) / n

            y_data.append(avg_error)
        
        y_data = np.array(y_data)

        plt.plot(x_data, y_data, label=f"num particles = {exp_path.split('/')[-2].split('_')[-1]}")

    # plt.axvline(x=vert_line_x, ymax=max_y, color='r', linestyle='--', linewidth=1, label=f'x = {vert_line_x}')
    # plt.text(vert_line_x + 0.2, 0, f'x = {vert_line_x:.2f}', color='r', verticalalignment='bottom')
    # plt.axhline(y=max_y, xmin=0, xmax=vert_line_x, color='r', linestyle='--', linewidth=1, label=f'y = {max_y}')


    plt.title(f"Average Error after convergence (RoNIN)")
    plt.xlabel("Dispersion threshold for convergence")
    plt.ylabel(f"Averge Error")
    plt.legend()
    plt.show()


def avg_error_vs_convergence_criteria_single_ori():
    """
    For different dispersion thresholds used as convergence criteria, measure the average error (distance) after that threshold
    """
    ori_path = './results/ARKit vs. RoNIN/num_particle_500/ARKit/BE_2/Ori 1'
    json_path = os.path.join(ori_path, 'dispersion_vs_distance.json')


    dispersions = []
    distances = []

    with open(json_path, 'r') as file:
        data = json.load(file)
        dispersions += data['dispersion']
        distances += data['distance']

    assert len(dispersions) == len(distances), "Error, lengths of dispersion data and distance data don't match"
    dispersions = np.array(dispersions)
    distances = np.array(distances)
    x_data = np.arange(0, np.max(dispersions), 0.1)

    y_data = []
    z_data = []
    for dispersion_threshold in x_data:
        avg_error = np.inf
        max_error = np.inf
        convergence_index = None
        for i in range(len(dispersions)):
            if dispersions[i] <= dispersion_threshold:
                convergence_index = i
                break
        if convergence_index is not None:
            n = len(distances) - convergence_index
            avg_error = np.sum(distances[convergence_index:]) / n
            max_error = np.max(distances[convergence_index:])

        y_data.append(avg_error)
        z_data.append(max_error)
    
    y_data = np.array(y_data)

    plt.plot(x_data, y_data, label="Avg error")
    # plt.plot(x_data, z_data, label="Max error")

    plt.title(f"Average Error after convergence for {ori_path.split('/')[-2:]}")
    plt.xlabel("Dispersion threshold for convergence")
    plt.ylabel(f"Averge Error")
    plt.legend()
    plt.show()


def error_vs_at_dispersion_threshold():
    """
    For different dispersion thresholds used as convergence criteria, measure the average error (distance) after that threshold
    """
    exp_path = './results/06.06/Uniform + Ori/exp1'

    dispersion_threshold = 0.1

    dispersions = []
    distances = []
    
    run_paths = [os.path.join(exp_path, run_name) for run_name in os.listdir(exp_path)]
    run_paths = [run_path for run_path in run_paths if os.path.isdir(run_path)]
    for run_path in run_paths:
        ori_paths = [os.path.join(run_path, ori_name) for ori_name in os.listdir(run_path)]
        ori_paths = [ori_path for ori_path in ori_paths if os.path.isdir(ori_path)]
        for ori_path in ori_paths:
            json_path = os.path.join(ori_path, 'dispersion_vs_distance.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as file:
                data = json.load(file)
                dispersions += data['dispersion']
                distances += data['distance']

    assert len(dispersions) == len(distances), "Error, lengths of dispersion data and distance data don't match"
    dispersions = np.array(dispersions)
    distances = np.array(distances)

    # plt.scatter(dispersions, distances)
    # plt.title("Particle Dispersions vs. Error")
    # plt.xlabel("particle dispersion")
    # plt.ylabel("distance from mean location to gt. (m)")
    # plt.show()

    x_data = np.arange(0, 20, 1)

    y_data = []

    convergence_index = None
    for i in range(len(dispersions)):
        if dispersions[i] <= dispersion_threshold:
            convergence_index = i
            break

    errors_after_convergence = distances[convergence_index:]
    n = len(errors_after_convergence)
    for meter in x_data:
        proportion = len(errors_after_convergence[np.where(errors_after_convergence <= meter)]) / n
        y_data.append(proportion)
    
    y_data = np.array(y_data)

    plt.plot(x_data, y_data, label=f"num particles = {exp_path.split('/')[-2].split('_')[-1]}")

    # plt.axvline(x=vert_line_x, ymax=max_y, color='r', linestyle='--', linewidth=1, label=f'x = {vert_line_x}')
    # plt.text(vert_line_x + 0.2, 0, f'x = {vert_line_x:.2f}', color='r', verticalalignment='bottom')
    # plt.axhline(y=max_y, xmin=0, xmax=vert_line_x, color='r', linestyle='--', linewidth=1, label=f'y = {max_y}')


    plt.title(f"Error Histogram at Dispersion Threshold = {dispersion_threshold}m")
    plt.xlabel("Meters")
    plt.ylabel(f"Percent Error < x meters")
    plt.legend()
    plt.show()


def min_dispersion_by_ori():
    """
    For different dispersion thresholds used as convergence criteria, measure the average error (distance) after that threshold
    """
    run_path = './results/ARKit vs. RoNIN/num_particle_500/ARKit/BE_1'
    ori_paths = [os.path.join(run_path, ori_name) for ori_name in os.listdir(run_path)]
    ori_paths = [ori_path for ori_path in ori_paths if os.path.isdir(ori_path)]
    for ori_path in ori_paths:
        json_path = os.path.join(ori_path, 'dispersion_vs_distance.json')

        dispersions = []
        distances = []

        with open(json_path, 'r') as file:
            data = json.load(file)
            dispersions += data['dispersion']
            distances += data['distance']

        assert len(dispersions) == len(distances), "Error, lengths of dispersion data and distance data don't match"
        dispersions = np.array(dispersions)
        distances = np.array(distances)

        print(np.min(dispersions))

        x_data = np.arange(0, len(dispersions), 1)
        y_data = dispersions
        plt.plot(x_data, y_data, label=f"{os.path.basename(ori_path)}")

    plt.title("E2, correct ori = Ori 1")
    plt.legend()
    plt.show()

    # y_data = []
    # z_data = []
    # for dispersion_threshold in x_data:
    #     avg_error = np.inf
    #     max_error = np.inf
    #     convergence_index = None
    #     for i in range(len(dispersions)):
    #         if dispersions[i] <= dispersion_threshold:
    #             convergence_index = i
    #             break
    #     if convergence_index is not None:
    #         n = len(distances) - convergence_index
    #         avg_error = np.sum(distances[convergence_index:]) / n
    #         max_error = np.max(distances[convergence_index:])

    #     y_data.append(avg_error)
    #     z_data.append(max_error)
    
    # y_data = np.array(y_data)

    # plt.plot(x_data, y_data, label="Avg error")
    # # plt.plot(x_data, z_data, label="Max error")

    # plt.title(f"Average Error after convergence for {ori_path.split('/')[-2:]}")
    # plt.xlabel("Dispersion threshold for convergence")
    # plt.ylabel(f"Averge Error")
    # plt.legend()
    # plt.show()


def dispersion_and_distance():
    run = "PS_1"
    ori_num = 0

    if "BE" in run:
        fp_path = "./maps/Baskin_New.Engineering.geojson.csv"
    elif "E2" in run:
        fp_path = "./maps/new_E2_3.geojson.csv"
    elif "PS" in run:
        fp_path = "./maps/Physical_New.Sciences.geojson.csv"

    floor_plan = load_map_csv(fp_path)

    tracking_data_path = f"./tracking_data/05.27/{run}/"
    tracking_data_path = os.path.join(tracking_data_path, [name for name in os.listdir(tracking_data_path) if 'json' in name][0])
    simulation_tracking_data_path = f"./results/05.29/exp0/{run}/Ori {ori_num}/simulation_tracking_data.json"

    tracking_data = load_tracking_data_json(tracking_data_path)
    sim_tracking_data = load_tracking_data_json(simulation_tracking_data_path)

    gt_trace = np.array(tracking_data["ARKit_PF"])
    trace_index = 0
    while True:
        sim_trace = np.array(sim_tracking_data['traces'][trace_index])
        dispersions = np.array(sim_tracking_data['dispersions'][trace_index])
        distances = np.array(sim_tracking_data['distances'][trace_index])

        visualize(segments_1=floor_plan, points_1=gt_trace, points_2=sim_trace)
        converge_idx = None
        for i, dispersion in enumerate(dispersions):
            if dispersion < 3:
                converge_idx = i
                break

        if converge_idx is None:
            trace_index += 1
            continue

        post_conv_distances = distances[converge_idx:]
        x_data = np.arange(0, len(distances), 1)
        plt.plot(x_data, distances, label="distance")
        plt.plot(x_data, dispersions, color='blue', label="dispersion")
        x_data = np.arange(converge_idx, len(distances), 1)
        plt.plot(x_data, post_conv_distances, color='red', label='post convergence distance')
        plt.ylabel("dispersion or distance")
        plt.xlabel("t")
        plt.legend()
        plt.show()
        trace_index += 1


def group_by_method(): 
    # Load the Excel file
    file_path = './results/05.23/num_particle_500/combined_results.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Replace 'inf' strings with NaN
    df.replace('inf', float('nan'), inplace=True)

    # Convert columns to numeric
    df['avg_converge_time'] = pd.to_numeric(df['avg_converge_time'], errors='coerce')
    df['avg_converge_distances'] = pd.to_numeric(df['avg_converge_distances'], errors='coerce')
    df['avg_RMSE'] = pd.to_numeric(df['avg_RMSE'], errors='coerce')
    df['success_rate'] = pd.to_numeric(df['success_rate'], errors='coerce')

    # Group by 'run' and 'method', get the maximum success rate
    best_success_rates = df.groupby(['run', 'method'])['success_rate'].max().reset_index()
    best_avg_converge_time = df.groupby(['run', 'method'])['avg_converge_time'].min().reset_index()
    best_avg_converge_distances = df.groupby(['run', 'method'])['avg_converge_distances'].min().reset_index()
    best_avg_RMSE = df.groupby(['run', 'method'])['avg_RMSE'].min().reset_index()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    best_success_rates = best_success_rates[best_success_rates['method'] != 'method']

    # Plot with specific labels
    sns.scatterplot(x='run', y='success_rate', hue='method', style='method', data=best_success_rates)
    plt.title('Best Success Rate for Each Run by Method')
    plt.xlabel('Run')
    plt.ylabel('Best Success Rate')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    best_avg_converge_time = best_avg_converge_time[best_avg_converge_time['method'] != 'method']

    # Plot with specific labels
    sns.scatterplot(x='run', y='avg_converge_time', hue='method', style='method', data=best_avg_converge_time)
    plt.title('Best Average Converge Time for Each Run by Method')
    plt.xlabel('Run')
    plt.ylabel('Best Average Converge Time')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    best_avg_converge_distances = best_avg_converge_distances[best_avg_converge_distances['method'] != 'method']

    # Plot with specific labels
    sns.scatterplot(x='run', y='avg_converge_distances', hue='method', style='method', data=best_avg_converge_distances)
    plt.title('Best Averge Converge Distance for Each Run by Method')
    plt.xlabel('Run')
    plt.ylabel('Best Averge Converge Distance')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    best_avg_RMSE = best_avg_RMSE[best_avg_RMSE['method'] != 'method']

    # Plot with specific labels
    sns.scatterplot(x='run', y='avg_RMSE', hue='method', style='method', data=best_avg_RMSE)
    plt.title('Best Average RMSE for Each Run by Method')
    plt.xlabel('Run')
    plt.ylabel('Best Average RMSE')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def group_by_data_source(): 
    # Load the Excel file
    file_path = './results/ARKit vs. RoNIN/num_particle_1000/combined_results.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Replace 'inf' strings with NaN
    df.replace('inf', float('nan'), inplace=True)

    # Convert columns to numeric
    df['avg_converge_time'] = pd.to_numeric(df['avg_converge_time'], errors='coerce')
    df['avg_converge_distances'] = pd.to_numeric(df['avg_converge_distances'], errors='coerce')
    df['avg_RMSE'] = pd.to_numeric(df['avg_RMSE'], errors='coerce')
    df['success_rate'] = pd.to_numeric(df['success_rate'], errors='coerce')

    # Group by 'run' and 'method', get the maximum success rate
    best_success_rates = df.groupby(['run', 'data_source'])['success_rate'].max().reset_index()
    best_avg_converge_time = df.groupby(['run', 'data_source'])['avg_converge_time'].min().reset_index()
    best_avg_converge_distances = df.groupby(['run', 'data_source'])['avg_converge_distances'].min().reset_index()
    best_avg_RMSE = df.groupby(['run', 'data_source'])['avg_RMSE'].min().reset_index()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    # best_success_rates = best_success_rates[best_success_rates['method'] != 'method']

    # Plot with specific labels
    sns.scatterplot(x='run', y='success_rate', hue='data_source', style='data_source', data=best_success_rates)
    plt.title('Best Success Rate for Each Run by data_source')
    plt.xlabel('Run')
    plt.ylabel('Best Success Rate')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    # best_avg_converge_time = best_avg_converge_time[best_avg_converge_time['data_source'] != 'data_source']

    # Plot with specific labels
    sns.scatterplot(x='run', y='avg_converge_time', hue='data_source', style='data_source', data=best_avg_converge_time)
    plt.title('Best Average Converge Time for Each Run by data_source')
    plt.xlabel('Run')
    plt.ylabel('Best Average Converge Time')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    # best_avg_converge_distances = best_avg_converge_distances[best_avg_converge_distances['method'] != 'method']

    # Plot with specific labels
    sns.scatterplot(x='run', y='avg_converge_distances', hue='data_source', style='data_source', data=best_avg_converge_distances)
    plt.title('Best Averge Converge Distance for Each Run by data_source')
    plt.xlabel('Run')
    plt.ylabel('Best Averge Converge Distance')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Filter out any potential "header" entries
    # best_avg_RMSE = best_avg_RMSE[best_avg_RMSE['method'] != 'method']

    # Plot with specific labels
    sns.scatterplot(x='run', y='avg_RMSE', hue='data_source', style='data_source', data=best_avg_RMSE)
    plt.title('Best Average RMSE for Each Run by data_source')
    plt.xlabel('Run')
    plt.ylabel('Best Average RMSE')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def convergence_time_vs_ori():
    exp_path = './results/05.28/exp1'

    dispersion_threshold = 1

    dispersions = []
    distances = []
    
    # run_paths = [os.path.join(exp_path, run_name) for run_name in os.listdir(exp_path)]
    # run_paths = [run_path for run_path in run_paths if os.path.isdir(run_path)]
    # for run_path in run_paths:
    run_path = os.path.join(exp_path, "E2")
    
    ori_paths = [os.path.join(run_path, ori_name) for ori_name in os.listdir(run_path)]
    ori_paths = [ori_path for ori_path in ori_paths if os.path.isdir(ori_path)]
    ori_names = [os.path.basename(ori_path) for ori_path in ori_paths]
    for ori_path in ori_paths:
        json_path = os.path.join(ori_path, 'dispersion_vs_distance.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as file:
            data = json.load(file)
            dispersions.append(data['dispersion'])
            distances.append(data['distance'])

    dispersions = np.array(dispersions)
    distances = np.array(distances)

    # plt.scatter(dispersions, distances)
    # plt.title("Particle Dispersions vs. Error")
    # plt.xlabel("particle dispersion")
    # plt.ylabel("distance from mean location to gt. (m)")
    # plt.show()

    x_data = np.arange(0, 4, 1)

    y_data = []

    for i in range(len(x_data)):
        for j in range(len(dispersions[i])):
            if dispersions[i][j] <= dispersion_threshold:
                y_data.append(j)
                break
        if len(y_data) == i:
            y_data.append(len(dispersions[i]))

    plt.plot(ori_names, y_data)

    plt.title(f"Convergence Speed, Threshold = {dispersion_threshold}m")
    plt.xlabel("Orientations")
    plt.ylabel(f"Convergence Index")
    plt.legend()
    plt.show()


def mean_velocity_vs_motion_velocity():
    run = "BE_1"
    ori_num = 0
    step_size = 10

    # if "BE" in run:
    #     fp_path = "./maps/Baskin_New.Engineering.geojson.csv"
    # elif "E2" in run:
    #     fp_path = "./maps/new_E2_3.geojson.csv"
    # elif "PS" in run:
    #     fp_path = "./maps/Physical_New.Sciences.geojson.csv"

    # floor_plan = load_map_csv(fp_path)
    y_data = []
    for ori_num in range(4):
        avg_max_diffs = []
        for step_size in range(1, 21):
            tracking_data_path = f"./tracking_data/05.27/{run}/"
            tracking_data_path = os.path.join(tracking_data_path, [name for name in os.listdir(tracking_data_path) if 'json' in name][0])
            simulation_tracking_data_path = f"./results/05.29/exp0/{run}/Ori {ori_num}/simulation_tracking_data.json"

            tracking_data = load_tracking_data_json(tracking_data_path)
            sim_tracking_data = load_tracking_data_json(simulation_tracking_data_path)

            gt_trace = np.array(tracking_data["RoNIN_raw_2D"])
            gt_velocity = np.linalg.norm(gt_trace[step_size:] - gt_trace[:-step_size], axis=1)
            
            max_diff = []

            for trace_index in range(len(sim_tracking_data['traces'])):
                sim_trace = np.array(sim_tracking_data['traces'][trace_index])
                dispersions = np.array(sim_tracking_data['dispersions'][trace_index])[step_size:]
                distances = np.array(sim_tracking_data['distances'][trace_index])


                # visualize(segments_1=floor_plan, points_1=gt_trace, points_2=sim_trace)
                dispersion_threshold = 3
                converge_idx = None
                for i, dispersion in enumerate(dispersions):
                    if dispersion < dispersion_threshold:
                        converge_idx = i
                        break
                    
                if converge_idx is None:
                    continue

                sim_velocity= np.linalg.norm(sim_trace[step_size:] - sim_trace[:-step_size], axis=1)
                x_data = np.arange(0, len(sim_velocity), 1)
                diff_velocity = np.abs(sim_velocity - gt_velocity[:len(sim_velocity)])

                max_diff.append(np.max(diff_velocity[converge_idx:converge_idx + 1]))

                # plt.axvline(x=converge_idx, color='r', linestyle='--', linewidth=1, label=f'dispersion < {dispersion_threshold}')
                # plt.axhline(y=1, color='r', linestyle='--', linewidth=1, label=f'y = 1')
                # plt.plot(x_data, sim_velocity, label="pred velocity")
                # plt.plot(x_data, gt_velocity[:len(sim_velocity)], label="gt velocity")
                # plt.plot(x_data, diff_velocity, label="diff")
                # plt.title("Predicted Velocity vs. Ground True Velocity")
                # plt.legend()
                # plt.show()
            
            avg_max_diff = np.mean(max_diff) / step_size
            avg_max_diffs.append(avg_max_diff)
        y_data.append(avg_max_diffs)

    x_data = np.arange(1, 21, 1)

    candidate_ori = None
    min_mmvd = np.inf
    for i in range(4):
        mmvd = np.mean(y_data[i][:10])
        if mmvd < min_mmvd:
            min_mmvd = mmvd
            candidate_ori = i
        plt.plot(x_data, y_data[i], label=f"Ori {i}")
    
    print(f"Candidate Orientation = Ori {candidate_ori}")

    plt.title(f"Mean Maximum Velocity Difference by Step Size - {run}")
    plt.ylabel("Mean Maximum Velocity Difference")
    plt.xlabel("Step Size")
    plt.legend()
    plt.show()


def mean_velocity_vs_motion_velocity_accuracy_test():
    run = "BE_1"
    correct_ori = 0

    tracking_data_path = f"./tracking_data/05.27/{run}/"
    tracking_data_path = os.path.join(tracking_data_path, [name for name in os.listdir(tracking_data_path) if 'json' in name][0])

    num_correct_predictions = 0
    num_failed_runs = 0

    for trace_index in range(10):
        candidate_ori = None
        min_mmvd = np.inf

        for ori_num in range(4):
            simulation_tracking_data_path = f"./results/05.29/exp0/{run}/Ori {ori_num}/simulation_tracking_data.json"
            tracking_data = load_tracking_data_json(tracking_data_path)
            sim_tracking_data = load_tracking_data_json(simulation_tracking_data_path)
            gt_trace = np.array(tracking_data["RoNIN_raw_2D"])
            sim_trace = np.array(sim_tracking_data['traces'][trace_index])
            assert len(sim_tracking_data['traces']) == 10, f"Error: Number of traces should be 100, but got {len(sim_tracking_data['traces'])} "


            dispersions = np.array(sim_tracking_data['dispersions'][trace_index])
            dispersion_threshold = 3
            converge_idx = None
            for i, dispersion in enumerate(dispersions):
                if dispersion < dispersion_threshold:
                    converge_idx = i
                    break
                
            if converge_idx is None:
                print(f"No convergence for Orientation {ori_num}")
                # plt.plot(np.arange(len(dispersions)), dispersions)
                # plt.show()
                continue

            max_diff = []
            for step_size in range(1, 21):
                gt_velocity = np.linalg.norm(gt_trace[step_size:] - gt_trace[:-step_size], axis=1)
                sim_velocity= np.linalg.norm(sim_trace[step_size:] - sim_trace[:-step_size], axis=1)
                diff_velocity = np.abs(sim_velocity - gt_velocity[:len(sim_velocity)])

                adjusted_converge_idx = converge_idx - step_size
                # print(f"Conv index = {adjusted_converge_idx}, Length diff velocity = {len(diff_velocity)}")
                max_diff.append(np.max(diff_velocity[adjusted_converge_idx:]))

            mmvd = np.mean(max_diff)  # Average max_diff of step size 0 ~ 10
            if mmvd < min_mmvd:
                min_mmvd = mmvd
                candidate_ori = ori_num

            x_data = np.arange(0, len(max_diff), 1)
            plt.plot(x_data, max_diff, label=f"Ori {ori_num}")

        if candidate_ori is None:
            # No success convergence, skipping this batch of 4 traces
            num_failed_runs += 1
            continue

        if candidate_ori == correct_ori:
            num_correct_predictions += 1
        
        
        print(f"Candidate Orientation = Ori {candidate_ori}")
        plt.title(f"Mean Maximum Velocity Difference by Step Size - {run}")
        plt.ylabel("Mean Maximum Velocity Difference")
        plt.xlabel("Step Size")
        plt.legend()
        plt.show()
    
    success_rate = num_correct_predictions / (10 - num_failed_runs)
    print(f"MMVD Prediction Success Rate for {run} = {success_rate}")


def velocity_diff():
    run = "BE_1"
    correct_ori = 0

    window_size = 40  # The windowed mean is the mean value of a window of size n

    tracking_data_path = f"./tracking_data/06.04/{run}/"
    tracking_data_path = os.path.join(tracking_data_path, [name for name in os.listdir(tracking_data_path) if 'json' in name][0])
    tracking_data = load_tracking_data_json(tracking_data_path)
    gt_trace = np.array(tracking_data["ARKit_raw_2D"])

    for trace_index in range(10):
        converge_indices = []
        vel_diff_by_ori = []
        for ori_num in range(4):
            simulation_tracking_data_path = f"./results/06.06/exp1/{run}/Ori {ori_num}/simulation_tracking_data.json"
            sim_tracking_data = load_tracking_data_json(simulation_tracking_data_path)

            sim_trace = np.array(sim_tracking_data['traces'][trace_index])
            avg_drifts = np.array(sim_tracking_data['avg_drifts'][trace_index])
            dispersions = np.array(sim_tracking_data['dispersions'][trace_index])
            theta = sim_tracking_data['theta']

            adjusted_gt_trace = gt_trace[:len(sim_trace)]
            assert len(adjusted_gt_trace) == len(sim_trace) == len(avg_drifts) == len(dispersions), "Error - tracking data length mismatch"
            

            dispersion_threshold = 3
            converge_idx = np.inf
            for i, dispersion in enumerate(dispersions):
                if dispersion < dispersion_threshold:
                    converge_idx = i
                    break
            converge_indices.append(converge_idx)
                
            if converge_idx == np.inf:
                print(f"No convergence for Orientation {ori_num}")
                # plt.plot(np.arange(len(dispersions)), dispersions)
                # plt.show()
                vel_diff_by_ori.append(None)
                continue
            
            vel_diff_by_step_size = []
            for step_size in range(1, 11):
                # Calculate Velocity
                gt_velocities = adjusted_gt_trace[step_size:] - adjusted_gt_trace[:-step_size]
                # Rotate the gt_velocities
                gt_velocities_rotated = []
                for i, gt_velocity in enumerate(gt_velocities):
                    avg_drift = avg_drifts[step_size + i]
                    R, _ = get_R_from_orientations(theta=theta + avg_drift)
                    T = np.eye(3)
                    T[:2, :2] = R
                    gt_velocity_rotated = apply_transformation_to_points(np.array([gt_velocity]), T)[0]
                    gt_velocities_rotated.append(gt_velocity_rotated)
                sim_velocities = sim_trace[step_size:] - sim_trace[:-step_size]
                # print(len(gt_velocities), len(gt_velocities_rotated), len(sim_velocities))

                sim_velocities = np.array(sim_velocities)
                gt_velocities_rotated = np.array(gt_velocities_rotated)

                # Calculate euclidean distance 
                diff_methods = ["euclidean", "mag_ratio", "cosine", "mag + cos"]
                diff_method = diff_methods[3]
                if diff_method == "euclidean":
                    vel_diff = np.linalg.norm(gt_velocities_rotated - sim_velocities, axis=1) / step_size
                elif diff_method == "mag_ratio":
                    sim_velocities_mag = np.linalg.norm(sim_velocities, axis=1)
                    gt_velocities_rotated_mag = np.linalg.norm(gt_velocities_rotated, axis=1)
                    vel_diff = np.abs(1 - (sim_velocities_mag / gt_velocities_rotated_mag))
                elif diff_method == "cosine":
                    cosine_distance = np.zeros(sim_velocities.shape[0])
                    for i in range(len(sim_velocities)):
                        v = sim_velocities[i]
                        u = gt_velocities_rotated[i]
                        dot_product = np.dot(v, u)
                        # Compute the norm (magnitude) of each vector
                        norm_v = np.linalg.norm(v)
                        norm_u = np.linalg.norm(u)
                        # Compute the cosine similarity
                        cosine_similarity = dot_product / (norm_v * norm_u)
                        # Compute the cosine distance
                        cosine_distance[i] = 1 - cosine_similarity

                    vel_diff = cosine_distance

                elif diff_method == "mag + cos":
                    cosine_distance = np.zeros(sim_velocities.shape[0])
                    for i in range(len(sim_velocities)):
                        v = sim_velocities[i]
                        u = gt_velocities_rotated[i]
                        dot_product = np.dot(v, u)
                        # Compute the norm (magnitude) of each vector
                        norm_v = np.linalg.norm(v)
                        norm_u = np.linalg.norm(u)
                        # Compute the cosine similarity
                        cosine_similarity = dot_product / (norm_v * norm_u)
                        # Compute the cosine distance
                        cosine_distance[i] = 1 - cosine_similarity

                    sim_velocities_mag = np.linalg.norm(sim_velocities, axis=1)
                    gt_velocities_rotated_mag = np.linalg.norm(gt_velocities_rotated, axis=1)
                    mag_diff = np.abs(1 - (sim_velocities_mag / gt_velocities_rotated_mag))

                    vel_diff = cosine_distance + mag_diff

                # Add paddings to the front to make vel_diff the same length as gt_trace
                padding_amount = step_size
                vel_diff = np.pad(vel_diff, (padding_amount, 0), mode='constant', constant_values=np.inf)

                assert len(vel_diff) == len(adjusted_gt_trace)

                vel_diff_by_step_size.append(vel_diff)
                # plt.plot(np.arange(step_size, len(gt_trace)), euclidean_distances, label=f"Step Size: {step_size}")
                # plt.axvline(x=converge_idx, linestyle='--', linewidth=1)
                # plt.legend()
                # plt.show()
                # sys.exit()
            vel_diff_by_ori.append(vel_diff_by_step_size)
        
        # Set a rolling windowed mean value
        windows = [[], [], [], []]
        candidates = []

        # Check which orientation converged first
        first_conv_idx = min(converge_indices)
        first_candidate = np.argmin(converge_indices)
        candidates.append(first_candidate)
        candidate_by_time_step = [None] * first_conv_idx
        
        step_size = 1  # Only use this step_size

        # Start rolling the time step
        for step in range(first_conv_idx, len(gt_trace)):
            # Check if any orientation have converged 
            if step in converge_indices:
                for candidate in range(len(converge_indices)):
                    if candidate not in candidates and converge_indices[candidate] == step:
                        candidates.append(candidate)
                        # print(f"New candidate {candidate} detected at time step {step}")

            # Update window for each converged orientations
            for candidate in candidates:
                # Check if the candidate ended early
                if len(vel_diff_by_ori[candidate][step_size - 1]) <= step:
                    candidates.remove(candidate)
                    windows[candidate] = [np.inf]
                    continue

                if len(windows[candidate]) == window_size:
                    windows[candidate].pop(0)
                
                windows[candidate].append(vel_diff_by_ori[candidate][step_size - 1][step]) # selecting step size = 1, TODO: Check if we need more step sizes
            
            # Compare window means and select a winning candidate
            window_mean = []
            for window in windows:
                if len(window) == 0 or window[0] == np.inf:
                    window_mean.append(np.inf)
                    continue
                window_mean.append(np.mean(window))
            # print(f"Step: {step}, Window Mean: {window_mean}")
            candidate_by_time_step.append(np.argmin(window_mean))
        

        # Visualize results
        fig, axs = plt.subplots(5, 1, sharex=True)
        for i in range(4):
            vel_diff_by_step_size = vel_diff_by_ori[i]
            if vel_diff_by_step_size is None:
                continue

            for step_size in range(1, 11):
            # step_size = 1
                x_data = np.arange(0, len(vel_diff_by_step_size[step_size-1]))
                y_data = vel_diff_by_step_size[step_size-1] if vel_diff_by_step_size is not None else [np.inf] * len(x_data)
                axs[i].plot(x_data, y_data, label=f"Step Size: {step_size}")
                axs[i].axvline(x=converge_indices[i], linestyle='--', linewidth=1)
                axs[i].set_ylim(0, 4)
            # axs[i].set_title(f"Ori {i}")

        axs[-1].plot(np.arange(0, len(gt_trace)), candidate_by_time_step, label="Selected Candidate")
        axs[-1].axhline(y=correct_ori, linestyle='--', linewidth=1, color='green', label="Correct Orientation")

        plt.title(f"{run}, Correct Ori = {correct_ori}", y=6)
        # plt.legend()
        plt.show()


def examine_fp_embed(fp_embed_path, floorplan_img):
    # Load data
    fp_embed = np.load(fp_embed_path)  # shape (512, 512, 128)

    if floorplan_img.ndim == 3:
        floorplan_img = floorplan_img[..., 0]  # Convert to grayscale if RGB

    assert fp_embed.shape[:2] == floorplan_img.shape[:2], "Shape mismatch"

    fig, ax = plt.subplots()
    ax.imshow(floorplan_img, cmap='gray', origin='lower')
    scatter = ax.scatter([], [], c='red', s=20, label='Top Matches')

    def onclick(event):
        if event.inaxes != ax:
            return
        x, y = int(event.xdata), int(event.ydata)

        # Get embedding at clicked point
        query_vec = fp_embed[y, x]  # note: y first because it's (H, W, C)

        # Reshape for distance calculation
        all_vecs = fp_embed.reshape(-1, fp_embed.shape[2])
        dists = np.linalg.norm(all_vecs - query_vec, axis=1)

        # Get top 5 positions
        topk_indices = np.argsort(dists)[:20]
        topk_coords = np.stack(np.unravel_index(topk_indices, fp_embed.shape[:2]), axis=1)

        # Update plot
        scatter.set_offsets(np.flip(topk_coords, axis=1))  # flip to (x, y)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    ax.set_title("Click on any point to find top-5 similar locations")
    plt.legend()
    plt.show()


def examine_fp_obs_embed(fp_embed_path, fp_img, obs_embed_path, obs_img):
    """
    Interactive viewer:
    - fp_embed_path: path to .npy file (H, W, C)
    - fp_img_path: path to floorplan image (H, W)
    - obs_embed: numpy array (H, W, C)
    - obs_img: numpy array (H, W) or (H, W, 3)
    """
    # Load and verify
    fp_embed = np.load(fp_embed_path)  # (H, W, C)
    obs_embed = np.load(obs_embed_path)
    
    if fp_img.ndim == 3:
        fp_img = fp_img[..., 0]
    if obs_img.ndim == 3:
        obs_img = obs_img[..., 0]
    
    assert fp_embed.shape[:2] == fp_img.shape[:2], "Mismatch in floorplan shape"
    assert obs_embed.shape[:2] == obs_img.shape[:2], "Mismatch in observation shape"

    H, W, C = fp_embed.shape

    # Flatten floorplan features
    fp_flat = fp_embed.reshape(-1, C)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax_fp, ax_obs = axs

    # Plot images
    ax_fp.imshow(fp_img, cmap='gray', origin='lower')
    scatter = ax_fp.scatter([], [], c='red', s=20, label='Top Matches')
    ax_fp.set_title("Floorplan")

    ax_obs.imshow(obs_img, cmap='gray', origin='lower')
    ax_obs.set_title("Observation — click to match")

    def onclick(event):
        if event.inaxes != ax_obs:
            return
        x, y = int(event.xdata), int(event.ydata)

        # Get observation embedding at clicked point
        obs_vec = obs_embed[y, x]  # (C,)
        dists = np.linalg.norm(fp_flat - obs_vec, axis=1)

        # Get top-5 matches in fp
        topk_indices = np.argsort(dists)[:5]
        topk_coords = np.stack(np.unravel_index(topk_indices, (H, W)), axis=1)

        # Update scatter on floorplan
        scatter.set_offsets(np.flip(topk_coords, axis=1))  # (x, y)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    ax_fp.legend()
    plt.tight_layout()
    plt.show()


def visualize_kernel_overlap(binary_map, kernel, gt_loc, kernel_center):
    """
    Overlays a kernel on a binary map centered at the ground truth location and visualizes the overlap.

    Args:
        binary_map (np.ndarray): 2D binary map (0: free, 1: occupied).
        kernel (np.ndarray): 2D kernel (e.g., gaussian or shape).
        gt_loc (tuple): (x, y) ground truth location in pixel coordinates.
        kernel_center (tuple): (cx, cy) center of the kernel (e.g., for a 15x15 kernel, it's (7, 7)).
    """
    map_h, map_w = binary_map.shape
    kh, kw = kernel.shape
    cx, cy = kernel_center
    x, y = int(gt_loc[0]), int(gt_loc[1])

    # Compute where the kernel will go on the map
    top_left_x = x - cx
    top_left_y = y - cy
    bottom_right_x = top_left_x + kw
    bottom_right_y = top_left_y + kh

    # Prepare canvas for overlay
    kernel_mask = np.zeros_like(binary_map, dtype=np.float32)

    # Compute valid region to paste kernel (in case it goes out of bounds)
    x1_k = max(0, -top_left_x)
    y1_k = max(0, -top_left_y)
    x2_k = min(kw, map_w - top_left_x)
    y2_k = min(kh, map_h - top_left_y)

    x1_m = max(0, top_left_x)
    y1_m = max(0, top_left_y)
    x2_m = x1_m + (x2_k - x1_k)
    y2_m = y1_m + (y2_k - y1_k)

    # Place kernel on the canvas
    kernel_mask[y1_m:y2_m, x1_m:x2_m] = kernel[y1_k:y2_k, x1_k:x2_k]

    # Highlight overlap: binary_map * kernel
    overlap = kernel_mask * binary_map

    # Crop for visualization
    crop_radius = max(kh, kw)
    crop_x1 = max(0, x - crop_radius)
    crop_x2 = min(map_w, x + crop_radius)
    crop_y1 = max(0, y - crop_radius)
    crop_y2 = min(map_h, y + crop_radius)

    binary_crop = binary_map[crop_y1:crop_y2, crop_x1:crop_x2]
    kernel_crop = kernel_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    overlap_crop = overlap[crop_y1:crop_y2, crop_x1:crop_x2]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_crop, cmap='gray', alpha=0.8, origin='lower')
    plt.imshow(kernel_crop, cmap='Blues', alpha=0.3, origin='lower')
    plt.imshow(overlap_crop, cmap='Reds', alpha=0.6, origin='lower')
    plt.scatter([x - crop_x1], [y - crop_y1], c='lime', marker='x', s=100, label='GT')
    plt.title("Kernel Overlap Visualization")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # distance_vs_dispersion()
    # group_by_data_source()
    # avg_error_vs_convergence_criteria()
    # avg_error_vs_convergence_criteria_single_ori()
    error_vs_at_dispersion_threshold()
    # convergence_time_vs_ori() 
    # min_dispersion_by_ori()
    # mean_velocity_vs_motion_velocity()
    # mean_velocity_vs_motion_velocity_accuracy_test()
    # velocity_diff()
        