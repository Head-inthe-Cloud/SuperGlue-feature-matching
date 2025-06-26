import os
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import math
import sys
from sklearn.cluster import KMeans
import shapely
from shapely.vectorized import contains
import alphashape
from shapely.geometry import Point, LineString, MultiLineString
from skimage import measure
from skimage.draw import line
from skimage.morphology import reconstruction
from scipy.ndimage import binary_fill_holes, binary_closing, binary_dilation
from scipy.spatial.transform import Rotation
import glob
import cv2


class Plane:
    def __init__(self, index, id, anchor_transform, plane_center, plane_extent, detected_time, updated_time):
        self.index = index
        self.id = id
        self.anchor_transform = np.matrix(anchor_transform)  # 4 x 4 numpy matrix
        self.plane_center = np.array(plane_center) # [cx, cy, cz]
        self.plane_extent = np.array(plane_extent) # [width, height, rotation_on_y_axis]
        self.detected_time = detected_time
        self.updated_time = updated_time
        self.T_sp = None   # Transforms coordinates from camera reference frame to session reference frame
    
    # Outputs the coordinates of the two end points in the form of [[x1, y1], [x2, y2]]
    # TODO: Change this method, it should find all 4 corners of the plane, project them onto the 2D plane, and pick 2 to represent the segment
    def to_2D(self):
        _ = self.get_plane_center()
        p1_p = np.array([[self.plane_extent[0] / 2, 0, 0, 1]]).T
        p2_p = np.array([[-self.plane_extent[0] / 2, 0, 0, 1]]).T
        p1_w = np.dot(self.T_sp, p1_p) # Transform the point from the plane reference frame to session frame
        p2_w = np.dot(self.T_sp, p2_p)

        # Project to 2D
        return np.array([[p1_w[0, 0], -p1_w[2, 0]], [p2_w[0, 0], -p2_w[2, 0]]])  # Extracting x and z coordinates, as y is the vertical direction

    # Return center of the plane in session frame, also calculates the transformation matrix form session frame to plane frame
    def get_plane_center(self):
        T_ap = np.eye(4) # transformation matrix for anchor frame to plane frame
        # Rotation
        rotation_on_y_axis = self.plane_extent[2]
        T_ap[0, 0] = np.cos(rotation_on_y_axis)
        T_ap[2, 2] = np.cos(rotation_on_y_axis)
        T_ap[0, 2] = np.sin(rotation_on_y_axis)
        T_ap[2, 0] = -np.sin(rotation_on_y_axis)

        # Translation
        T_ap[0, 3] = self.plane_center[0]
        T_ap[1, 3] = self.plane_center[1]
        T_ap[2, 3] = self.plane_center[2]
        T_ap = np.matrix(T_ap)

        self.T_sp = np.dot(self.anchor_transform, T_ap)

        Cc = np.array([[0, 0, 0, 1]]).T

        return np.dot(self.T_sp, Cc)

    def __str__(self):
        return 'Plane Object. Index: {}, ID: {}'.format(self.index, self.id)


class Segment:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.normal = None
        self.orientation = None

    def __str__(self):
        return f'Segment Object ({self.x1}, {self.y1}), ({self.x2}, {self.y2})'
    


# Inputs should be both in the form of [[x1, y1], [x2, y2]]
# The angle is measured from seg2 to seg1
def get_orientation_diff_from_segments(seg1, seg2, mode='radian'):
    v1 = np.array([seg1[1, 0] - seg1[0, 0], seg1[1, 1] - seg1[0, 1]])
    v2 = np.array([seg2[1, 0] - seg2[0, 0], seg2[1, 1] - seg2[0, 1]])
    theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    cross_product = np.cross(v1, v2)
    if cross_product < 0:
        theta = -theta

    if mode == 'degree':
        theta = np.degrees(theta)
    return theta



def angle_difference(angle1, angle2, mode='radian'):
    """
    Finds the absolute difference between two orientations in radians.

    Parameters:
    - angle1: The first orientation in radians.
    - angle2: The second orientation in radians.

    Returns:
    - The absolute difference between the two orientations in radians,
      constrained to the range [0, π].
    """
    # Normalize angles to the range [0, 2π)
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)
    
    # Calculate the absolute difference
    diff = np.abs(angle1 - angle2)
    
    # Ensure the difference is within the range [0, π]
    if diff > np.pi:
        diff = 2 * np.pi - diff
        
    if mode == 'degree':
        diff / np.pi * 180

    return diff


def get_segments_distance(seg1, seg2, mode):

    # Naive Solution
    # ------------------------------------
    # If segments intersect, return 0
    if mode == "point":
        if segments_intersect(seg1, seg2):
            return 0
        distances = []
        distances.append(point_segment_distance(seg1[0], seg2))
        distances.append(point_segment_distance(seg1[1], seg2))
        distances.append(point_segment_distance(seg2[0], seg1))
        distances.append(point_segment_distance(seg2[1], seg1))
        return min(distances)

    # Distance function from the ICL paper
    # Reference: https://www.sciencedirect.com/science/article/pii/S0898122100002303
    # TODO: Figure out how to deal with length difference in floor plan to observation matches
    # ------------------------------------
    elif mode == "ICL":
        # l1 = length(L1), l2 = length(L2)
        # D(L1, L2) = (l1 + l2) / 2 * (||O1 - O2||^2 + (l1*l2/12) * ||V1-V2||^2 + (l1-l2)^2 / 12)
        # O1 and O2 are the center of the segments, V1 and V2 are unit vectors that represent directions
        alpha = 1
        beta = 1
        gamma = 1

        l1 = np.linalg.norm(seg1)
        l2 = np.linalg.norm(seg2)
        O1 = (seg1[0] + seg1[1]) / 2
        O2 = (seg2[0] + seg2[1]) / 2
        V1 = seg1 / l1
        V2 = seg2 / l2

        dis_component = np.sum((O1 - O2)**2)
        ori_component = (l1 * l2 / 12) * np.sum((V1 - V2)**2)
        len_component = (l1 - l2) ** 2 / 12

        # print(f"Distance: {dis_component}, Orientation: {ori_component}, Length: {len_component}")
        D = (l1 + l2) / 2 * (alpha * dis_component + beta * ori_component + gamma * len_component)
        return D
    
    elif mode == "NL":
        # Similar to ICL, but ignore length diff
        l1 = np.linalg.norm(seg1)
        l2 = np.linalg.norm(seg2)
        O1 = (seg1[0] + seg1[1]) / 2
        O2 = (seg2[0] + seg2[1]) / 2
        V1 = seg1 / l1
        V2 = seg2 / l2

        D = (np.sum((O1 - O2)**2) + np.sum((V1 - V2)**2))
        return D
       

def segments_intersect(seg1, seg2):
    x11 = seg1[0, 0]
    y11 = seg1[0, 1]
    x12 = seg1[1, 0]
    y12 = seg1[1, 1]
    x21 = seg2[0, 0]
    y21 = seg2[0, 1]
    x22 = seg2[1, 0]
    y22 = seg2[1, 1]

    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21
    delta = dx2 * dy1 - dy2 * dx1
    if delta == 0: return False
    s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
    t = (dx2 * (y11- y21) + dy2 * (x21 - x11)) / (-delta)

    return (0 <= s <= 1) and (0 <= t <= 1)


def point_segment_distance(point, seg):
    px, py = point[0], point[1]
    x1, y1, x2, y2 = seg[0, 0], seg[0, 1], seg[1, 0], seg[1, 1]
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)

    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        near_x = x1 + t * dx
        near_y = y1 + t * dy
        dx = px - near_x
        dy = py - near_y

    return math.hypot(dx, dy)


def point_to_point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_intersection_point(seg1, seg2):
    xdiff = (seg1[0][0] - seg1[1][0], seg2[0][0] - seg2[1][0])
    ydiff = (seg1[0][1] - seg1[1][1], seg2[0][1] - seg2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*seg1), det(*seg2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


def get_T(l1=None, l2=None, L1=None, L2=None, segments=None, correspondences=None, mode="naive"):
    if mode == "naive":
        ori_diff_l1_L1 = get_orientation_diff(l1, L1)
        ori_diff_l2_L2 = get_orientation_diff(l2, L2)
        avg_ori_diff = (ori_diff_l1_L1 + ori_diff_l2_L2) / 2

        l1_l2_intersection = get_intersection_point(l1, l2)
        L1_L2_intersection = get_intersection_point(L1, L2)
        [dx1, dy1] = -l1_l2_intersection
        [dx2, dy2] = L1_L2_intersection

        Tr = [[np.cos(avg_ori_diff), -np.sin(avg_ori_diff), 0],
                [np.sin(avg_ori_diff), np.cos(avg_ori_diff), 0],
                [0, 0, 1]]
        Tr = np.matrix(Tr)
        Tt1 = [[1, 0, dx1],[0, 1, dy1],[0, 0, 1]]
        Tt1 = np.matrix(Tt1)
        Tt2 = [[1, 0, dx2],[0, 1, dy2],[0, 0, 1]]
        Tt2 = np.matrix(Tt2)
        T = np.dot(Tt2, np.dot(Tr, Tt1))
        return T
    
    elif mode == "SVD":
        if l1 is not None and l2 is not None and L1 is not None and L2 is not None:
            l_l1 = get_length(l1)
            l_l2 = get_length(l2)
            l_L1 = get_length(L1)
            l_L2 = get_length(L2)
            w = sum([l_l1, l_l2, l_L1, l_L2])
            w_1 = (l_l1 + l_L1) / w
            w_2 = (l_l2 + l_L2) / w
            p_hat = w_1 * (l1[0] + l1[1]) / 2 + w_2 * (l2[0] + l2[1]) / 2   # [2]
            q_hat = w_1 * (L1[0] + L1[1]) / 2 + w_2 * (L2[0] + L2[1]) / 2 

            l1_hat = l1 - p_hat   # [2, 2]
            l2_hat = l2 - p_hat 
            L1_hat = L1 - q_hat 
            L2_hat = L2 - q_hat

            A = (l_l1 + l_L1) / 6 * (2 * np.dot(L1_hat[0].reshape((2, -1)), l1_hat[0].reshape((-1, 2))) + 
                                        2 * np.dot(L1_hat[1].reshape((2, -1)), l1_hat[1].reshape((-1, 2))) + 
                                        np.dot(L1_hat[0].reshape((2, -1)), l1_hat[1].reshape((-1, 2))) + 
                                        np.dot(L1_hat[1].reshape((2, -1)), l1_hat[0].reshape((-1, 2))))
            
            A += (l_l2 + l_L2) / 6 * (2 * np.dot(L2_hat[0].reshape((2, -1)), l2_hat[0].reshape((-1, 2))) + 
                                        2 * np.dot(L2_hat[1].reshape((2, -1)), l2_hat[1].reshape((-1, 2))) + 
                                        np.dot(L2_hat[0].reshape((2, -1)), l2_hat[1].reshape((-1, 2))) + 
                                        np.dot(L2_hat[1].reshape((2, -1)), l2_hat[0].reshape((-1, 2))))
            
        elif segments is not None and correspondences is not None:
            A = np.diag(([0.0, 0.0]))

            w = 0
            l_ps = []
            l_qs = []
            for i in range(len(segments)):
                l_p = get_length(segments[i])
                l_q = get_length(correspondences[i])
                l_ps.append(l_p)
                l_qs.append(l_q)

                w += l_p + l_q
            
            p_hat = np.array([0, 0], dtype=np.float64)
            q_hat = np.array([0, 0], dtype=np.float64)
            for i in range(len(segments)):
                w_i = (l_ps[i] + l_qs[i]) / w
                p_hat += w_i * (segments[i][0] + segments[i][1]) / 2 
                q_hat += w_i * (correspondences[i][0] + correspondences[i][1]) / 2


            segments_hat = segments - p_hat
            correspondences_hat = correspondences - q_hat
            
            for i in range(len(segments)):
                A += (l_ps[i] + l_qs[i]) / 6 * (2 * np.dot(correspondences_hat[i][0].reshape((2, -1)), segments_hat[i][0].reshape((-1, 2))) + 
                                            2 * np.dot(correspondences_hat[i][1].reshape((2, -1)), segments_hat[i][1].reshape((-1, 2))) + 
                                            np.dot(correspondences_hat[i][0].reshape((2, -1)), segments_hat[i][1].reshape((-1, 2))) + 
                                            np.dot(correspondences_hat[i][1].reshape((2, -1)), segments_hat[i][0].reshape((-1, 2))))


        # Calculate R
        U, W, VT = np.linalg.svd(A)
        det_UVT = np.linalg.det(A)
        
        if det_UVT < 0:
            J = np.diag([1, -1])
            R = np.dot(np.dot(U, J), VT)
        else:
            R = np.dot(U, VT)
        
        # Calculate translation matrix t
        t = q_hat - np.dot(R, p_hat)

        return R, t

    
    else:
        print("You need to select a right mode for get_T()")


def apply_transformation_to_segments(segments, T):
    segments_transformed = np.copy(segments)
    for i in range(segments.shape[0]):
        for j in range(2):
            point_aug = np.array([[segments[i, j, 0], segments[i, j, 1], 1]]).T
            segments_transformed[i, j] = np.dot(T, point_aug).T[0, :-1]
    return segments_transformed


def apply_transformation_to_points(points, T):
    points_transformed = np.zeros_like(points)
    for i in range(points.shape[0]):
        point_aug = np.array([[points[i, 0], points[i, 1], 1]]).T
        points_transformed[i] = np.dot(T, point_aug).T[0, :-1]
    return points_transformed


def filter_segments_by_distance(segments, point, range_limit=20):
    filtered_segments = []
    for seg in segments:
        if point_segment_distance(point, seg) < range_limit:
            filtered_segments.append(seg)
    return np.array(filtered_segments)


def filter_segments_by_length(segments, min_length):
    filtered_segments = []
    for seg in segments:
        if get_length(seg) > min_length:
            filtered_segments.append(seg)
    return np.array(filtered_segments)


def get_length(seg):
    return np.sqrt((seg[0, 0] - seg[1, 0])**2 + (seg[0, 1] - seg[1, 1])**2)


def get_center(seg):
    x_center = (seg[0, 0] + seg[1, 0]) / 2
    y_center = (seg[0, 1] + seg[1, 1]) / 2
    return np.array([x_center, y_center])


def to_radian(angle):
    return angle / 180 * np.pi

def to_degree(angle):
    return angle / np.pi * 180


def load_map_geojson(input_path):
    pass

# Returns a 3D array [[[x1, y1], [x2, y2]]]
def load_map_csv(input_path):
    map_data = []
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            p1 = np.array([float(row[0]), float(row[1])])
            p2 = np.array([float(row[2]), float(row[3])])
            map_data.append(np.array([p1, p2]))
    
    return np.array(map_data)

def load_map_desdf(input_path):
    desdf = np.load(
        input_path, allow_pickle=True
    ).item()
    desdf = desdf['desdf']
    return desdf

# Return a list of Planes
def load_planes_json(input_path, format="new", alignment=1):
    assert format in ["old", "new", "2D"], "format should be either old, new, or 2D"

    if ".json" not in input_path:
        print("The input path is not a json file")
        return
    
    planes = []

    with open(input_path, 'r') as f:
        data = json.load(f)
        for entry in data:
            if format == "old":
                plane = Plane(entry['index'], entry['id'], entry['anchorTransform'], entry['planeCenter'], entry['planeExtent'], entry['detectedTime'], entry['updatedTime'])
                plane = plane.to_2D()
            elif format == "new":
                if entry['planeAlignment'] != alignment:
                    continue
                plane = Plane(entry['index'], entry['id'], entry['anchorTransform'], entry['centers'][-1], entry['extents'][-1], entry['updatedTimes'][0], entry['updatedTimes'][-1])
                plane = plane.to_2D()
            elif format == "2D":
                plane = entry # List [[x1, y1], [x2, y2]]
            planes.append(plane)
    
    return np.array(planes)


def load_depth_png(path, scale=0.001):
    """
    Read a 16-bit PNG depth map and convert to meters.

    Args:
        path (str): Path to the PNG depth file.
        scale (float): Scale factor to convert depth values to meters.
                       E.g., scale=0.001 if depth is in millimeters.

    Returns:
        np.ndarray: Depth map in meters (float32).
    """
    depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read depth image from {path}")
    depth = depth_raw.astype(np.float32) * scale
    return depth


def save_map_csv(map_data, output_path):
    """
    Save map data to a CSV file.

    Args:
        output_path (str): Path to save the CSV file.
        map_data (np.ndarray): Array of shape (N, 2, 2) where each element represents a line segment 
                               with two 2D points [p1, p2], and each point is [x, y].
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for segment in map_data:
            row = [segment[0][0], segment[0][1], segment[1][0], segment[1][1]]
            writer.writerow(row)


def load_tracking_data_json(input_path):
    if ".json" not in input_path:
        print("The input path is not a json file")
        return
    
    with open(input_path, 'r') as f:
        data = json.load(f)
        return data


def load_obs(obs_path):
    # Load images, arkit_depths, intrinsics, and poses (extrinsic) 
    # This only loads data where each session only captures one image, and one depth
    # Images should be in .jpg format, arkit_depths, intrinsics, and poses should be in .json format
    images = []
    arkit_depths = []
    intrinsics = []
    poses = []
    paths = {'image_paths': [], 
             'arkit_depth_paths': [], 
             'intrinsics_paths': [],
             'pose_paths': []}

    session_names = [name for name in os.listdir(obs_path) if '.' not in name]
    session_paths = [os.path.join(obs_path, name) for name in session_names] 
    for session_path in session_paths:
        image_path = glob.glob(os.path.join(session_path, 'images') + '/*.jpg')[0]
        arkit_depth_path = glob.glob(os.path.join(session_path, 'depths') + '/*.json')[0]
        intrinsics_path = os.path.join(session_path, 'cameraIntrinsics.json')
        pose_path = os.path.join(session_path, 'cameraPose.json')

        paths['image_paths'].append(image_path)
        paths['arkit_depth_paths'].append(arkit_depth_path)
        paths['intrinsics_paths'].append(intrinsics_path)
        paths['pose_paths'].append(pose_path)

        # Load image
        image = cv2.imread(image_path)
        height, width, _ = image.shape  # Get the dimensions of the original image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        images.append(image)

        # Load ARKit depth
        with open(arkit_depth_path, 'r') as f:
            depth = json.load(f)
        depth = np.array(depth, dtype=np.float32)
        depth = depth.reshape((192, 256))  # Adjust to LiDAR depth resolution
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
        arkit_depths.append(depth)
        
        # Load intrinsics
        with open(intrinsics_path, 'r') as file:
            data = json.load(file)
            camera_intrinsic = np.array(data["data"], dtype=np.float32)
            if camera_intrinsic.shape != (3, 3):
                raise ValueError("Loaded intrinsic matrix is not 3x3.")
            intrinsics.append(camera_intrinsic)

        # Load pose
        with open(pose_path, 'r') as file:
            data = json.load(file)
            pose = np.array(data["data"], dtype=np.float32)
            if pose.shape != (4, 4):
                raise ValueError("Loaded transformation matrix is not 4x4.")
            poses.append(pose)

    return images, arkit_depths, intrinsics, poses, paths, session_names


def load_full_obs(obs_path, return_numpy=True):
    # Load images, arkit_depths, confidence maps, intrinsics, poses, ARKit detected planes, and time stamps
    # Images should be in .png format, arkit_depths, intrinsics, and poses should be in .json format
    images = []
    arkit_depths = []
    confidences = []
    intrinsics = []
    poses = []
    planes = []
    time_stamps = []

    paths = {'image_paths': [], 
             'arkit_depth_paths': [],
             'confidence_paths': [], 
             'intrinsics_paths': [],
             'pose_paths': [],
             'planes_path': None,
             'time_stamps_path': None}

    planes_path = os.path.join(obs_path, 'detectedPlanes.json')
    paths['planes_path'] = planes_path
    time_stamps_path = os.path.join(obs_path, 'timeStamps.json')
    paths['time_stamps_path'] = time_stamps_path

    # load planes
    planes = load_planes_json(planes_path)
    # load time stamps
    with open(time_stamps_path, 'r') as file:
        time_stamps = json.load(file)

    frame_ids = [name[-11:-4] for name in os.listdir(os.path.join(obs_path, 'images')) if '.png' in name]
    frame_ids = sorted(frame_ids)
    for frame_id in frame_ids:
        image_path = os.path.join(obs_path, 'images', f'image_{frame_id}.png')
        arkit_depth_path = os.path.join(obs_path, 'depths', f'depth_{frame_id}.json')
        confidence_path = os.path.join(obs_path, 'confidences', f'confidence_{frame_id}.png')
        intrinsics_path = os.path.join(obs_path, 'intrinsics', f'cameraIntrinsics_{frame_id}.json')
        pose_path = os.path.join(obs_path, 'poses', f'cameraPose_{frame_id}.json')

        paths['image_paths'].append(image_path)
        paths['arkit_depth_paths'].append(arkit_depth_path)
        paths['confidence_paths'].append(confidence_path)
        paths['intrinsics_paths'].append(intrinsics_path)
        paths['pose_paths'].append(pose_path)

        # Load image
        image = cv2.imread(image_path)
        height, width, _ = image.shape  # Get the dimensions of the original image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        images.append(image)

        # Load ARKit depth
        with open(arkit_depth_path, 'r') as f:
            depth = json.load(f)
        depth = np.array(depth, dtype=np.float32)
        depth = depth.reshape((192, 256))  # Adjust to LiDAR depth resolution
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
        arkit_depths.append(depth)

        # Load confidence map
        confidence = cv2.imread(confidence_path)
        confidences.append(confidence)
        
        # Load intrinsics
        with open(intrinsics_path, 'r') as file:
            data = json.load(file)
            camera_intrinsic = np.array(data["data"], dtype=np.float32)
            if camera_intrinsic.shape != (3, 3):
                raise ValueError("Loaded intrinsic matrix is not 3x3.")
            intrinsics.append(camera_intrinsic)

        # Load pose
        with open(pose_path, 'r') as file:
            data = json.load(file)
            pose = np.array(data["data"], dtype=np.float32)
            if pose.shape != (4, 4):
                raise ValueError("Loaded transformation matrix is not 4x4.")
            poses.append(pose)
    
    if return_numpy:
        images = np.array(images)
        arkit_depths = np.array(arkit_depths)
        confidences = np.array(confidences)
        intrinsics = np.array(intrinsics)
        poses = np.array(poses)
        planes = np.array(planes)
        time_stamps = np.array(time_stamps)
        frame_ids = np.array(frame_ids)

        for key in paths.keys():
            if type(paths[key]) == list:
                paths[key] = np.array(paths[key])

    return images, arkit_depths, confidences, intrinsics, poses, planes, time_stamps, paths, frame_ids


def load_label(label_file):
    if not os.path.exists(label_file):
        print(f"Error: {label_file} doesn't exist")
        return
    with open(label_file, 'r') as file:
        for line in file:
            line_data = line.strip().split(',')
            x, y, theta = float(line_data[0]), float(line_data[1]), float(line_data[2])
            return x, y, theta


def decode_camera_pose(pose):
    """
    Decode a 4x4 camera-to-world pose matrix into translation and ZXY-based roll, pitch, yaw.

    Axis conventions:
        - Roll  = rotation around Z
        - Pitch = rotation around X
        - Yaw   = rotation around Y

    Args:
        pose (np.ndarray): 4x4 transformation matrix

    Returns:
        dict: translation and custom-order rotation values in degrees
    """
    assert pose.shape == (4, 4), "Pose must be a 4x4 matrix"

    # Extract translation
    translation = pose[:3, 3]

    # Extract rotation matrix
    rotation_matrix = pose[:3, :3]

    # Convert rotation matrix to roll (Z), pitch (X), yaw (Y) using ZXY convention
    r = Rotation.from_matrix(rotation_matrix)
    roll_z, pitch_x, yaw_y = r.as_euler('zxy', degrees=True)

    # Print nicely
    print(f"Translation: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")
    print(f"Rotation (degrees): roll={roll_z:.2f}° (Z), pitch={pitch_x:.2f}° (X), yaw={yaw_y:.2f}° (Y)")

    return {
        'translation': translation,
        'rotation_deg': {
            'roll_z': roll_z,
            'pitch_x': pitch_x,
            'yaw_y': yaw_y
        }
    }


def get_homography_from_rotation(pose_src, pose_tgt, K):
    """
    Compute the homography matrix that maps image coordinates from src to tgt,
    assuming only rotation between the two views.

    Args:
        pose_src (np.ndarray): 4x4 camera-to-world pose matrix of the source image.
        pose_tgt (np.ndarray): 4x4 camera-to-world pose matrix of the target image.
        K (np.ndarray): 3x3 intrinsic matrix (same for both images).

    Returns:
        np.ndarray: 3x3 homography matrix mapping src image to tgt image.
    """
    # Extract rotation matrices (R_c2w)
    R_src = pose_src[:3, :3]
    R_tgt = pose_tgt[:3, :3]

    # Relative rotation from tgt to src (i.e., src in tgt frame)
    R_rel = R_tgt @ R_src.T

    # Homography: H = K * R_rel * K^-1
    H = K @ R_rel @ np.linalg.inv(K)

    return H


def order_by_distance(segments, reference_seg, mode="point"):
    result = sorted(segments, key=lambda segment: get_segments_distance(reference_seg, segment, mode))
    return result


def order_by_length(segments, reversed=False):
    if reversed:
        result = sorted(segments, key=lambda segment: -get_length(segment))
    else:
        result = sorted(segments, key=lambda segment: get_length(segment))

    return np.array(result)


def get_matching_score(planes, floorplan, mode="naive", k=5, alpha=10):
        ms = []
        for i, l in enumerate(planes):
            # Pick the top k closest planes
            distances = []
            if mode == "naive":
                for L in floorplan:
                    distances.append(get_segments_distance(l, L, mode='point'))
                distances = np.array(distances)
                idx = np.argpartition(distances, k)[:k]

                # Filter, find the one with lowest orientation diff
                best_idx = None
                min_ori_diff = 10000
                for j in idx:
                    L = floorplan[j]
                    ori_diff = abs(get_orientation_diff(l, L))
                    if ori_diff < min_ori_diff:
                        min_ori_diff = ori_diff
                        best_idx = j
                dist = distances[best_idx]
                # ms.append(min_ori_diff * dist) # Original matching score, doesn't quite make sense in my opinion. O degree diff would give extremely small ms even if the distance is large
                ms.append(min_ori_diff * alpha + dist)

            else:
                for L in floorplan:
                    distances.append(get_segments_distance(l, L, mode=mode))
                distances = np.array(distances)
                ms.append(np.min(distances))

            
        return np.array(ms)


def get_closest_segment_idx_at_point(segments, point, k=1):
    """
    Returns the indices of the top K segments closest to the given point.
    The distance is calculated with the closest endpoint of the segment
    """
    assert len(segments) > 0 and point is not None

    if k == 1:
        min_d = math.inf
        min_idx = None
        for i in range(len(segments)):
            for j in range(2):
                d = point_to_point_distance(point, segments[i, j])
                if d < min_d:
                    min_d = d
                    min_idx = i

        return min_idx
    
    else:
        result = set()
        flattened = segments.reshape((-1, 2))
        distances = np.array([point_to_point_distance(point, p) for p in flattened])
        closest_point_idxs = np.argpartition(distances, k * 2)[:k * 2 + 1]

        meta_idxs = np.argsort(distances[closest_point_idxs])
        for meta_idx in meta_idxs:
            if len(result) >= k:
                return np.array(list(result))
            segment_idx = closest_point_idxs[meta_idx] // 2
            result.add(segment_idx)

# The input are the two pairs of indices corresponding to:
#   selected segments in the observed planes, with shape (2, 2, 2)
# Returns:
#   (8, 4, 2) - 8 permutations x (l1[0], l1[1], l2[0], l2[1]) x (segment_idx, point_idx)
def get_correspondance_permutations(segment_idxs):
    permutations = []  # (8, 4, 2) - 8 permutations x (l1[0], l1[1], l2[0], l2[1]) x (segment_idx, point_idx)
    for segment_idx in range(2):
        for point_idx_1 in range(2):
            for point_idx_2 in range(2):
                permutations.append([[segment_idxs[segment_idx], point_idx_1], 
                                    [segment_idxs[segment_idx], abs(point_idx_1-1)], 
                                    [segment_idxs[abs(segment_idx-1)], point_idx_2],
                                    [segment_idxs[abs(segment_idx-1)], abs(point_idx_2-1)]])
    
    return np.array(permutations)


def find_correspondences(segments1, segments2, mode="ICL"):
    correspondences = []
    for segment1 in segments1:
        segment2 = order_by_distance(segments2, segment1, mode=mode)[0] 
        min_distance = get_segments_distance(segment1, segment2, mode=mode)

        rev_segment1 = np.array([segment1[1], segment1[0]])
        rev_segment2 = order_by_distance(segments2, rev_segment1, mode=mode)[0]
        rev_segment2 = np.array([segment2[1], segment2[0]]) # Remove this line?

        
        if get_segments_distance(segment1, rev_segment2, mode=mode) < min_distance:
            correspondences.append(rev_segment2)
        else:
            correspondences.append(segment2)

    return np.array(correspondences)


def get_segment_orientations(segments):
    orientations = []
    for segment in segments:
        vector = segment[1] - segment[0]
        # Use arctan to find orientation g
        orientation = np.arctan2(vector[1], vector[0])  # This value is within the range [-pi/2, pi/2]
        if orientation < -np.pi/2:
            orientation += np.pi
        elif orientation > np.pi/2:
            orientation -= np.pi

        orientations.append(orientation)
    return np.array(orientations)


# Use k-means clustering to find n clusters, calculate the mean orientation for them, and return the top k mean orientation
# Input:
#    segments - numpy arrays with shape (n, 2, 2)
#    k - the number of main orientations to return
# Returns: 
#    orientations - numpy array with shape (k) angles within -pi/2 ~ pi/2. 
#                   Note that this orientation also represents the opposite direction, aka theta and theta + pi.
#    labels - numpy array containing k arrays, the ith array contains indices of segments correspond to the ith orientation

def find_principal_orientations(segments, k=2, method='search', n_clusters=3, threshold=to_radian(1)):
    assert method in ['kmeans', 'search']
    orientations = get_segment_orientations(segments)

    if method == 'search':
        threshold = threshold
        buckets = [0] * 180
        # Iterate over -pi/2 to pi/2
        for delta in range(180):
            theta = -np.pi/2 + to_radian(delta)
            theta_1 = theta - threshold
            theta_2 = theta + threshold

            if theta_1 < -np.pi/2:
                theta_1 = (theta_1 // (-np.pi/2)) * (np.pi/2) - theta_1 % (np.pi/2)
            if theta_2 > np.pi/2:
                theta_2 = (theta_2 // (np.pi/2)) * (-np.pi/2) + theta_2 % (np.pi/2)
            
            for idx, orientation in enumerate(orientations):
                if theta_1 <= orientation <= theta_2:
                    # buckets[delta] += 1                         # Only count number of segments, ignore length
                    buckets[delta] += get_length(segments[idx])   # Weighted by segment length

        best_ori = 0
        best_ori_score = 0
        for idx_1 in range(180):
            idx_2 = (idx_1 + 90) % 180
            score = buckets[idx_1] + buckets[idx_2]
            if score > best_ori_score:
                best_ori = to_radian(idx_1) - np.pi/2
                best_ori_score = score

        ortho_ori = best_ori + np.pi/2
        if ortho_ori > np.pi/2:
            ortho_ori = (ortho_ori // (np.pi/2)) * (-np.pi/2) + ortho_ori % (np.pi/2)

        return np.array([best_ori, ortho_ori])
            

    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans.fit(orientations.reshape((-1, 1)))
        labels = kmeans.labels_

        # Group orientations by label
        clustered_orientations = {}  # key = label, value = [orientations]
        for i, label in enumerate(labels):
            if label in clustered_orientations:
                clustered_orientations[label].append(orientations[i])
            else:
                clustered_orientations[label] = [orientations[i]]

        # # Pick top k clusters by size
        # top_k_clusters = sorted(list(clustered_orientations.values()), key=lambda x: len(x))[-k:]

        # # Calculate means for top k clusters
        # top_k_orientations = [np.mean(cluster) for cluster in top_k_clusters]
                
        # Pick top k clusters by size, shape (k, )
        top_k_clusters_labels = sorted(list(clustered_orientations.keys()), key=lambda x: len(clustered_orientations[x]))[-k:]

        # Calculate means for top k clusters, shape (k, )
        top_k_orientations = [np.mean(clustered_orientations[label]) for label in top_k_clusters_labels]

        # Reformat Labels, contains the indices of segments, shape (k, #segments_in_each_cluster)
        labels_reformatted = [[] for _ in range(k)]
        for label in top_k_clusters_labels:
            for i in range(len(labels)):
                if labels[i] == label:
                    j = top_k_clusters_labels.index(label)
                    labels_reformatted[j].append(i)  # The ith segment belongs to the jth output orientation

        # Return top k orientations
        return np.array(top_k_orientations), labels_reformatted
        

    
# Input:
#    ori_1 - an angle in radian form, -pi/2 ~ pi/2
#    ori_2 - an angle in radian form, -pi/2 ~ pi/2
#    theta - an angle in radian form, how much to rotate the segments by
# Return: 
#    R - 2x2 rotation matrix that transform ori_1 into ori_2
def get_R_from_orientations(ori_1=None, ori_2=None, theta=None):
    if theta is None:
        theta = ori_2 - ori_1
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    return R, theta


def get_theta_from_R(rotation_matrix):
    """
    Calculate the rotation angle from a 2x2 rotation matrix.

    Args:
        rotation_matrix (np.array): A 2x2 rotation matrix.

    Returns:
        float: The rotation angle in radians.
    """
    # Check if the matrix is 2x2
    if rotation_matrix.shape != (2, 2):
        raise ValueError("The input must be a 2x2 matrix.")

    # Ensure it's a rotation matrix by checking if its transpose is its inverse
    if not np.allclose(np.linalg.inv(rotation_matrix), np.transpose(rotation_matrix)):
        raise ValueError("The input must be a valid rotation matrix.")

    # Calculate the angle using the arctangent of the matrix elements
    angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return angle


def get_ori_matches(fp_oris, obs_oris, theta=0, threshold=to_radian(5)):
    # Given two sets of orientations A and B, match each of the orientations from one set to another, return the paired indices
    # Input:
    #    fp_oris - np array, main orientations of the floor plan
    #    obs_oris - np array, main orientations of the observed planes
    #    theta - the angle obs_oris are rotated by, default to 0
    #    threshold - the angle threshold to determine if two orientations should match, assign -1 otherwise
    # Return:
    #    ori_matches - np array of shape (k, 2) - [obs_ori_idx, matching fp_ori_idx] * k
    ori_matches = []
    diffs = np.zeros((len(obs_oris), len(fp_oris)))

    # Create a diff matrix storing the min difference between two orientations
    for i in range(len(obs_oris)):
        for j in range(len(fp_oris)):
            diff_1 = abs((obs_oris[i] + theta) % (2 * np.pi) - fp_oris[j])
            diff_1 = np.minimum(diff_1, 2*np.pi - diff_1)
            diff_2 = abs((obs_oris[i] + theta + np.pi) % (2 * np.pi) - fp_oris[j])
            diff_2 = np.minimum(diff_2, 2*np.pi - diff_2)
            diffs[i, j] = min(diff_1, diff_2)
    
    for i in range(len(obs_oris)):
        # Find the pair of orientations with the smallest difference
        min_indices = np.unravel_index(np.argmin(diffs), diffs.shape)
        # If the difference is large, stop the matching
        if diffs[min_indices] == np.inf or diffs[min_indices] > threshold:
            break
        # Add the match with the smallest difference to the list
        ori_matches.append(min_indices)
        # Make sure the each orientation is only matched once
        diffs[min_indices[0], :] = np.inf
        diffs[:, min_indices[1]] = np.inf

    # For any unmatched obs_ori_idx, assign -1 
    temp = [match[0] for match in ori_matches]
    for i in range(len(obs_oris)):
        if i not in temp:
            ori_matches.append((i, -1))

    ori_matches.sort(key=lambda x: x[0])

    return np.array(ori_matches)
    

def find_optimal_rotation(A, B):
    """
    Finds the rotation that minimizes the error between two sets of points with the same number of elements
    """
    # Center the points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the covariance matrix
    H = np.dot(A_centered.T, B_centered)

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R = np.dot(Vt.T, U.T)

    return R

# Input:
#       segments - numpy arrays with shape (n x 2 x 2)
#       sample_density - a number that determines the distance between each sampled points on a segment, default to 0.1, which stands for 10cm
#       segment_labels - a list of shape (k, #segments in each group), determines which group the points belong to, usually grouped by orientation
# Return:
#       points - if no segment labels are provided: numpy arrays with shape (n, 2) 
#                if segment labels are provided: a list of shape (k, #Points per group)
#       labels - which segment was the point sampled from, numpy array with shape (n, )
#                returns None if segment labels are provided
def sample_points_from_segments(segments, sample_density=0.1, segment_labels=None):
    if segment_labels is None:
        points = []
        labels = []
        for seg_idx, segment in enumerate(segments):
            vector = segment[1] - segment[0]
            length = get_length(segment)
            num_points = int(np.ceil(length / sample_density))
            dx = vector[0] / num_points
            dy = vector[1] / num_points
            for i in range(num_points + 1):
                point = [segment[0, 0] + dx * i, segment[0, 1] + dy * i]
                points.append(point)
                labels.append(seg_idx)
        
            
        return np.array(points), labels

    else:
        points = [[] for _ in range(len(segment_labels))]
        for seg_idx, segment in enumerate(segments):
            # Find which group the segment belongs to
            segment_label = None
            for i in range(len(segment_labels)):
                if seg_idx in segment_labels[i]:
                    segment_label = i
                    break

            # If the segment does not belong to any orientation group, we ignore it
            if segment_label is None:
                continue

            # Sample points from this segment
            vector = segment[1] - segment[0]
            length = get_length(segment)
            num_points = int(length // sample_density)
            dx = vector[0] / num_points
            dy = vector[1] / num_points
            for i in range(num_points + 1):
                point = [segment[0, 0] + dx * i, segment[0, 1] + dy * i]
                points[segment_label].append(point)
        points = [np.array(sub_points) for sub_points in points]
        
        return points, None


# def segments_to_binary_map(segments, cell_size=1.0):
#     converter = Bresenham(cell_size=cell_size)
#     points = None
#     for segment in segments:
#         x_list, y_list = converter.seg(segment[0, 0], segment[0, 1], segment[1, 0], segment[1, 1])
#         x_list = np.array(x_list)
#         y_list = np.array(y_list)
#         assert(len(x_list) == len(y_list)), "Sizes of x_list and y_list need to match, error from the results of the Bresenham algorithm"
#         if points is None:
#             points = np.column_stack((x_list, y_list))
#         else:
#             points = np.concatenate((points, np.column_stack((x_list, y_list))), axis=0)
    
#     return np.array(points)
    
def segments_to_binary_map(segments, cell_size, filler_value=0, width=None, height=None, to_bot_left=False):
    BH = Bresenham(cell_size)

    segments_copy = segments.copy()
    min_x = np.min(segments_copy[:, :, 0])
    min_y = np.min(segments_copy[:, :, 1])
    max_x = np.max(segments_copy[:, :, 0])
    max_y = np.max(segments_copy[:, :, 1])

    if to_bot_left:
        segments_copy[:, :, 0] -= min_x
        segments_copy[:, :, 1] -= min_y

    if width is None or height is None:
        width = int(np.ceil((max_x - min_x) / cell_size))
        height = int(np.ceil((max_y - min_y) / cell_size))


    if filler_value == 0:
        binary_map = np.zeros((height, width), dtype=np.uint8)
    else:
        binary_map = np.ones((height, width), dtype=np.uint8) * filler_value

    for segment in segments_copy:
        x0, y0 = segment[0]
        x1, y1 = segment[1]
        x_res_list, y_res_list = BH.seg(x0, y0, x1, y1)
        for x, y in zip(x_res_list, y_res_list):
            if 0 <= x < width and 0 <= y < height:
                binary_map[int(y), int(x)] = 1

    return binary_map


def crop_circle_from_binary_map(binary_map, center, radius):
    """
    Crop a square from a binary map centered at `center` and keep only pixels within a circular mask.

    Args:
        binary_map (np.ndarray): 2D binary image.
        center (tuple): (x, y) center in pixel coordinates.
        radius (int): Radius in pixels.

    Returns:
        np.ndarray: Cropped circular mask from binary map (shape: (2*radius, 2*radius)).
    """
    h, w = binary_map.shape
    cx, cy = center

    # Define the square bounds
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius)

    # Crop the square
    crop = binary_map[y1:y2, x1:x2]

    # Create a circular mask
    yy, xx = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')
    circle_mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2

    # Apply circular mask
    circular_crop = crop * circle_mask.astype(np.uint8)

    return circular_crop




def pad_image_to_square(image, filler_value=0, fixed_size=None):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else: 
        height, width = image.shape
    # Pad to square if needed
    if height != width:
        if fixed_size is None:
            size = max(height, width)
        else:
            size = fixed_size
        pad_top = (size - height) // 2
        pad_bottom = size - height - pad_top
        pad_left = (size - width) // 2
        pad_right = size - width - pad_left

        new_image = np.pad(
            image,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=filler_value
        )
        return new_image, pad_left, pad_top

    # Returns the image and the paddings in x and y directions in pixel units
    return image, 0, 0

def filter_visible_pixels(binary_map, location):
    """
    Filters visible pixels in a binary map from a given location using ray casting.

    Args:
        binary_map (np.ndarray): Binary image (H x W), values are 0 or 1.
        location (tuple): (x, y) in pixel coordinates, the source location.

    Returns:
        np.ndarray: Filtered binary map where only directly visible pixels remain.
    """
    H, W = binary_map.shape
    x0, y0 = location
    output_map = np.copy(binary_map)

    # Find all non-zero (candidate) pixels
    candidate_ys, candidate_xs = np.nonzero(binary_map)

    for x1, y1 in zip(candidate_xs, candidate_ys):
        if (x0, y0) == (x1, y1):
            continue  # skip self

        # Draw the line between the location and candidate pixel
        rr, cc = line(y0, x0, y1, x1)

        # Check for any other non-zero pixels along the line, excluding the last point
        if np.any(binary_map[rr[:-1], cc[:-1]]):  # skip the last pixel (target)
            output_map[y1, x1] = 0  # occluded, remove

    return output_map


def filter_visible_segments(vector_map, location, radius, sample_resolusion=0.1):
    """
    Given a vector map, observer location, and radius, return only the visible portions of segments.

    Args:
        vector_map (np.ndarray): (n, 2, 2) vector map of line segments.
        location (tuple): (x, y) of the observer.
        radius (float): Radius of visibility.
        sample_resolusion (float): How finely to sample each segment.

    Returns:
        np.ndarray: (m, 2, 2) visible parts of the segments.
    """
    origin = Point(location)
    circle = origin.buffer(radius)
    obstacles = [LineString(seg) for seg in vector_map]

    visible_segments = []

    for seg in obstacles:
        seg_clipped = seg.intersection(circle)
        if seg_clipped.is_empty:
            continue

        # Turn into iterable of LineStrings (in case result is MultiLineString)
        seg_parts = [seg_clipped] if isinstance(seg_clipped, LineString) else list(seg_clipped)

        for part in seg_parts:
            sampled_pts = [part.interpolate(dist, normalized=False) for dist in np.linspace(0, part.length, int(part.length / sample_resolusion))]
            coords = np.array([[pt.x, pt.y] for pt in sampled_pts])
            visible = [False] * (len(coords) - 1)

            for i in range(len(coords) - 1):
                mid = (coords[i] + coords[i+1]) / 2
                ray = LineString([location, mid])
                occluded = False
                for other in obstacles:
                    if other.equals(seg):
                        continue
                    inter = ray.intersection(other)
                    if not inter.is_empty and origin.distance(inter) < origin.distance(Point(mid)):
                        occluded = True
                        break
                if not occluded:
                    visible[i] = True

            # Reconstruct visible line segments from visible flags
            i = 0
            while i < len(visible):
                if visible[i]:
                    start = coords[i]
                    while i < len(visible) and visible[i]:
                        i += 1
                    end = coords[i] if i < len(coords) else coords[-1]
                    visible_segments.append(np.vstack([start, end]))
                else:
                    i += 1
    
    # Move segments to be zero centered
    visible_segments = np.array(visible_segments)
    visible_segments = translate_segments(visible_segments, -location[0], -location[1])
    return visible_segments


def crop_floorplan_around_point(segments, center, cell_size=0.1, side_length=10.0):
    """
    Converts segments to a binary map and crops a square region around a given center.

    Args:
        segments (np.ndarray): Array of shape (N, 2, 2), each entry [[x1, y1], [x2, y2]].
        center (tuple): (x, y) in meters - center of the crop.
        cell_size (float): Size of each cell in meters/pixel.
        side_length (float): Length of the square crop in meters.

    Returns:
        cropped_binary_map (np.ndarray): Cropped binary img (H, W) where 1 indicates lines.
    """
    BH = Bresenham(cell_size)
    half_len = side_length / 2.0
    x_min = center[0] - half_len
    y_min = center[1] - half_len
    x_max = center[0] + half_len
    y_max = center[1] + half_len

    width = math.ceil(side_length / cell_size)
    height = math.ceil(side_length / cell_size)
    binary_map = np.zeros((height, width), dtype=np.uint8)

    for segment in segments:
        x0, y0 = segment[0]
        x1, y1 = segment[1]

        # Skip if both points are outside the bounding box
        if (x0 < x_min and x1 < x_min) or (x0 > x_max and x1 > x_max) or \
           (y0 < y_min and y1 < y_min) or (y0 > y_max and y1 > y_max):
            continue

        # Translate points relative to top-left corner of the crop
        x0 -= x_min
        x1 -= x_min
        y0 -= y_min
        y1 -= y_min

        x_res_list, y_res_list = BH.seg(x0, y0, x1, y1)
        for x, y in zip(x_res_list, y_res_list):
            if 0 <= x < width and 0 <= y < height:
                binary_map[int(y), int(x)] = 1

    return binary_map



def merge_segments(segments, distance_threshold=0.2, angle_threshold=np.pi/180 * 10):
    segments = list(segments)
    
    merged_segments = []
    while len(segments) > 0:
        segment = segments[0]
        segments = segments[1:]
        segment_group = [segment]
        for i, other_segment in enumerate(segments):
            distance = get_segments_distance(segment, other_segment, mode="point")
            angle_diff = get_orientation_diff_from_segments(segment, other_segment, mode='radian')
            if distance < distance_threshold and abs(angle_diff) < angle_threshold:
                segment_group.append(other_segment)
                segments.pop(i)
        merged_segments.append(merge_segments_helper(segment_group))
    return np.array(merged_segments)

def merge_segments_helper(segments):
    orientations = get_segment_orientations(segments)
    # Calculate average angle
    average_ori = np.mean(orientations)
    
    # Calculate direction vector from average angle
    ori_vector = np.array([np.cos(average_ori), np.sin(average_ori)])
    
    # Project endpoints onto the average direction
    projections = []
    for segment in segments:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        projections.append((np.array([x1, y1]), np.dot(np.array([x1, y1]), ori_vector)))
        projections.append((np.array([x2, y2]), np.dot(np.array([x2, y2]), ori_vector)))
    
    # Find the furthest points along the projection
    projections.sort(key=lambda p: p[1])
    furthest_points = [projections[0][0], projections[-1][0]]
    
    return np.array(furthest_points)


def rotate_segments(segments, ori_1=None, ori_2=None, theta=None):
    if theta is None:
        theta = ori_2 - ori_1
    R, theta = get_R_from_orientations(theta=theta)
    T = np.eye(3)
    T[:2, :2] = R
    segments_T = apply_transformation_to_segments(segments, T=T)
    return segments_T


def translate_segments(segments, x, y):
    """
    Translates a list of 2D line segments by (x, y) using a transformation matrix.

    Args:
    - segments (list): List of line segments in the form [[[x1, y1], [x2, y2]], ...].
    - x (float): Translation offset in the x-direction.
    - y (float): Translation offset in the y-direction.

    Returns:
    - translated_segments (list): List of translated segments.
    """
    # Define the 3x3 translation matrix for homogeneous coordinates
    T = np.array([
        [1, 0, x],  # Translate in X
        [0, 1, y],  # Translate in Y
        [0, 0, 1]   # Homogeneous coordinate
    ])

    # Apply transformation using the helper function
    translated_segments = apply_transformation_to_segments(segments, T=T)

    return translated_segments


def rotate_segments_to_landscape(segments):
    """
    Takes a set of segments, usually a vector floor plan, and rotate it according to its main orientation
    Such that the width is greater than the height, aka landscape position

    Parameters:
        segments - the segments to be rotated, shape (n, 2, 2)
    Returns:
        segments_T - the rotated segments
        T - the transformation matrix
        theta - the angle the segments were rotated by, in radians
    """
    # Find the main orientation
    main_orientations = find_principal_orientations(segments, threshold=to_radian(0.5))
    original_orientation = main_orientations[1]
    R, theta = get_R_from_orientations(original_orientation, 0)
    T = np.eye(3)
    T[:2, :2] = R

    # Aligns with the 0 degree
    segments_T = apply_transformation_to_segments(segments, T=T)

    # Check if width is greater than height, otherwise, rotate it +- 90 degrees, which ever is closer to the original orientation
   
    # Find the max and min values among the x-coordinates of the rotated points

    # Compute the span
    x_coords = np.concatenate((segments_T[:, 0, 0], segments_T[:, 1, 0]))
    y_coords = np.concatenate((segments_T[:, 0, 1], segments_T[:, 1, 1]))
    max_x = np.max(x_coords)
    min_x = np.min(x_coords)
    max_y = np.max(y_coords)
    min_y = np.min(y_coords)
    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_y > span_x:
        final_orientation = np.pi / 2

        if angle_difference(original_orientation, np.pi / 2) > np.pi / 2:
            final_orientation *= -1
        
        R, theta = get_R_from_orientations(original_orientation, final_orientation)
        T = np.eye(3)
        T[:2, :2] = R
        segments_T = apply_transformation_to_segments(segments, T=T)
        x_coords = np.concatenate((segments_T[:, 0, 0], segments_T[:, 1, 0]))
        y_coords = np.concatenate((segments_T[:, 0, 1], segments_T[:, 1, 1]))
        min_x = np.min(x_coords)
        min_y = np.min(y_coords)

    # Translate the map such that it starts from 0
    T_t = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0,  1]
    ])
    segments_T = apply_transformation_to_segments(segments_T, T_t)

    T = T_t @ T

    return segments_T, T, theta


def rotate_binary_image(binary_image, angle, x_min, z_min, resolution=0.1):
    """
    Rotates a binary image by a given angle with respect to the original (x_min, z_min) origin,
    and translates the image so that it fully fits in the new frame.

    Args:
    - binary_image (np.ndarray): Input binary image (H x W).
    - angle (float): Angle in degrees to rotate the image.
    - x_min (float): Minimum x-coordinate from original point cloud.
    - z_min (float): Minimum z-coordinate from original point cloud.
    - resolution (float): Grid resolution in meters per pixel.

    Returns:
    - rotated_binary (np.ndarray): Rotated and translated binary image.
    """
    # Get image dimensions
    h, w = binary_image.shape

    # Compute the original real-world center
    x_center = x_min + (w // 2) * resolution
    z_center = z_min + (h // 2) * resolution

    # Convert real-world center to pixel coordinates
    pixel_center_x = (x_center - x_min) / resolution
    pixel_center_z = (z_center - z_min) / resolution
    center = (int(pixel_center_x), int(pixel_center_z))

    angle = to_degree(angle)
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding box size after rotation
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the translation part of the rotation matrix
    rotation_matrix[0, 2] += (new_w - w) // 2  # Shift right
    rotation_matrix[1, 2] += (new_h - h) // 2  # Shift down

    # Perform the rotation with translation
    rotated_binary = cv2.warpAffine(binary_image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_NEAREST)

    # Find bounding box of non-zero pixels
    coords = np.column_stack(np.where(rotated_binary > 0))  # Find all non-zero pixel locations
    if coords.shape[0] == 0:
        return rotated_binary  # If no foreground pixels, return as is

    x_min_crop, y_min_crop = coords.min(axis=0)
    x_max_crop, y_max_crop = coords.max(axis=0)

    # Crop to bounding box
    cropped_binary = rotated_binary[x_min_crop:x_max_crop+1, y_min_crop:y_max_crop+1]

    return cropped_binary

def split_into_patches(image, n=5):
    """
    Splits the image into n x n non-overlapping patches.

    Args:
        image (np.ndarray): Input image as a 2D or 3D NumPy array (H x W or H x W x C).
        n (int): Number of patches along each dimension (default 5).

    Returns:
        patches (list of np.ndarray): List of image patches (each of shape h x w or h x w x C).
    """
    H, W = image.shape[:2]
    pad_H = (n - H % n) % n
    pad_W = (n - W % n) % n

    if pad_H > 0 or pad_W > 0:
        pad_width = ((0, pad_H), (0, pad_W)) if image.ndim == 2 else ((0, pad_H), (0, pad_W), (0, 0))
        image = np.pad(image, pad_width, mode='constant', constant_values=0)
        H, W = image.shape[:2]

    patch_h = H // n
    patch_w = W // n

    patches = []
    for i in range(n):
        for j in range(n):
            patch = image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            patches.append(patch)

    return patches

def position_to_velocity(points, time_intervals):
    assert len(points) == len(time_intervals), f"The length of points({len(points)}) must equal to the length of time_intervals({len(time_intervals)})"
    pre_point = np.array([0, 0])   # Assuming all tracking data start at (0, 0), and the first step is not far from the origin
    velocities = []
    for i in range(len(points)):
        displacement = points[i] - pre_point
        velocities.append(displacement / time_intervals[i])
        pre_point = points[i]

    assert len(velocities) == len(points), "The number of velocities should be equal the number of points"
    return np.array(velocities)


def velocity_to_position(velocities, time_intervals):
    assert len(velocities) == len(time_intervals), f"The length of points({len(velocities)}) must equal to the length of time_intervals({len(time_intervals)})"
    current_point = np.array([0.0, 0.0])   # Assuming all tracking data start at (0, 0), and the first step is not far from the origin
    points = []
    for i in range(len(velocities)):
        displacement = velocities[i] * time_intervals[i]
        current_point = current_point + displacement 
        points.append(current_point)

    assert len(points) == len(points), "The number of velocities should be equal the number of points"
    return np.array(points)



def concave_hull(segments: np.ndarray, ratio=0.05, visualize=False) -> shapely.geometry.polygon.Polygon:
    sampled_points = sample_points_from_segments(segments=segments, sample_density=1)[0]
    sampled_points = [(p[0], p[1]) for p in sampled_points]
    point_collection = shapely.geometry.MultiPoint(sampled_points)

    result = shapely.concave_hull(point_collection, ratio=ratio)

    if visualize:
        x, y = result.exterior.xy
        for seg in segments:
            plt.plot(*zip(*seg), color='black')

        plt.plot(*zip(*sampled_points), 'ro', markersize=1)

        plt.plot(x, y)
        plt.fill(x, y, alpha=0.5)
        plt.axis('equal')
        plt.show()

    return result



def convex_hull(segments: np.ndarray, visualize=False) -> shapely.geometry.polygon.Polygon:
    sampled_points = sample_points_from_segments(segments=segments, sample_density=1)[0]
    sampled_points = [(p[0], p[1]) for p in sampled_points]
    point_collection = shapely.geometry.MultiPoint(sampled_points)

    result = shapely.convex_hull(point_collection)

    if visualize:
        x, y = result.exterior.xy
        for seg in segments:
            plt.plot(*zip(*seg), color='black')

        plt.plot(*zip(*sampled_points), 'ro', markersize=1)

        plt.plot(x, y)
        plt.fill(x, y, alpha=0.5)
        plt.axis('equal')
        plt.show()

    return result


def alpha_shape(segments: np.ndarray, alpha=0.15, cell_size=0.1, visualize=False) -> np.ndarray:
    """
    Compute the alpha shape (concave hull) of a set of points.
    Parameters:
    points (np.array): Array of points.
    alpha (float): Alpha parameter.
    Returns:
    shapely.geometry.Polygon: The alpha shape.
    """
    sampled_points = sample_points_from_segments(segments=segments, sample_density=1)[0]
    
    mask = alphashape.alphashape(sampled_points, alpha)

    if visualize:
        try:
            x, y = mask.exterior.xy
            for seg in segments:
                plt.plot(*zip(*seg), color='black')

            plt.plot(*zip(*sampled_points), 'ro', markersize=1)

            plt.plot(x, y)
            plt.fill(x, y, alpha=0.5)
            plt.axis('equal')
            plt.show()
        except:
            print("No mask found")
            return
    
    points = segments.reshape((segments.shape[0] * 2, 2))
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    x_coords = np.arange(min_x, max_x, cell_size)
    y_coords = np.arange(min_y, max_y, cell_size)
    xv, yv = np.meshgrid(x_coords, y_coords)
    points = np.vstack((xv.flatten(), yv.flatten())).T

    binary_mask = contains(mask, points[:, 0], points[:, 1])
    binary_mask = binary_mask.reshape(xv.shape)

    return binary_mask

from skimage.morphology import reconstruction

def point_distances(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Find the distance between two arrays of points

    param:
        points1: an array of points, numpy array, shape (n, 2)
        points2: an array of points, numpy array, shape (n, 2)

    return:
        distances: distance between points, numpy array, shape (n)
    """
    assert len(points1) == len(points2), f"Error: the lengths of the arrays of points must be the same, the first array has length {len(points1)}, while the second one is {len(points2)}"

    distances = []
    for i in range(len(points1)):
        distances.append(np.linalg.norm(points2[i] - points1[i]))
    
    return np.array(distances)


def modify_tracking_data(tracking_data, time_intervals, scale_factor):
    """
    Takes tracking data, extracts the velocity at each step and apply a scaling factor
    Then use this modified velocity to recreate and return the tracking data
    """
    velocities = position_to_velocity(points=tracking_data, time_intervals=time_intervals)
    velocities *= scale_factor
    new_tracking_data = velocity_to_position(velocities=velocities, time_intervals=time_intervals)
    return new_tracking_data


def get_segment_orientation_at_xy(segments, point, resolution=0.1):
    """
    Given a 2D point (x, y), finds the segment it lands on and return the orientation of that segment

    param:
        segments: ndarray, (n, 2, 2)
        point: ndarray, (2,)
    return:
        orientation: -pi/2 ~ pi/2
    """
    filtered_segments = filter_segments_by_distance(segments, point, range_limit=resolution)
    n = len(filtered_segments)
    if n == 1:
        return(get_segment_orientations(filtered_segments)[0])
    return None
        
def normalize_heatmap(heatmap, linear_normalization=True, as_prob_distribution=False):
    normalized_heatmap = heatmap
    if linear_normalization:
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        if min_val == max_val:
            return heatmap
        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)

    if as_prob_distribution:
        eps = 1e-8
        total = np.sum(normalized_heatmap)
        normalized_heatmap /= total  # All sum to 1

        # Makes the lowest probability equal to eps, while still all sum to 1
        mask = normalized_heatmap < eps
        inverse_mask = ~mask
        normalized_heatmap[inverse_mask] *= 1 - (np.sum(mask) * eps)
        normalized_heatmap[mask] += eps
        
    return normalized_heatmap


def get_yaw_from_pose(pose):
    """
    Extract the yaw angle (rotation around Y-up axis) from a 4x4 camera pose matrix.
    """
    forward = pose[:3, 2]  # Camera forward is Z-axis in camera frame
    yaw = np.arctan2(forward[0], forward[2])  # atan2(x, z)
    return (yaw + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2pi)





def get_rotation_matrix_from_two_vectors(a, b):
    """
    Returns the rotation matrix that rotates vector a to vector b.
    Both a and b must be 3D vectors.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    if np.isclose(c, 1.0):
        return np.eye(3)  # No rotation needed
    elif np.isclose(c, -1.0):
        # 180-degree rotation around any axis orthogonal to a
        axis = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return rotation_matrix_from_axis_angle(v, np.pi)

    skew = np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ])
    R = np.eye(3) + skew + skew @ skew * ((1 - c) / (np.linalg.norm(v) ** 2))
    return R

def rotation_matrix_from_axis_angle(axis, angle):
    """
    Rodrigues' rotation formula
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R

class Bresenham:
    def __init__(self, cell_size):
        self.cell_size = cell_size

    def point_rounder(self, value):
        round_res = np.floor(value / self.cell_size)
        if round_res < 0:
            return 0.0
        return round_res

    def seg(self, x0, y0, x1, y1):
        x0 = self.point_rounder(x0)
        y0 = self.point_rounder(y0)
        x1 = self.point_rounder(x1)
        y1 = self.point_rounder(y1)

        dx = np.abs(x1 - x0)
        dy = np.abs(y1 - y0)
        x = x0
        y = y0
        sx = -1.0 if x0 > x1 else 1.0
        sy = -1.0 if y0 > y1 else 1.0

        x_res_list = []
        y_res_list = []

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                x_res_list.append(x)
                y_res_list.append(y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx

                    # enhance the wall
                    x_res_list.append(x)
                    y_res_list.append(y)
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                x_res_list.append(x)
                y_res_list.append(y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy

                    # enhance the wall
                    x_res_list.append(x)
                    y_res_list.append(y)
                y += sy

        x_res_list.append(x)
        y_res_list.append(y)
        return x_res_list, y_res_list
    

from utils.visualization import visualize
# Structured3D Functionalities

def project_segment_onto_another(base_seg: LineString, proj_seg: LineString):
    """
    Projects `proj_seg` onto `base_seg` and computes the length of the projection.
    """
    base_start, base_end = np.array(base_seg.coords)
    proj_start, proj_end = np.array(proj_seg.coords)

    base_dir = base_end - base_start
    base_dir_norm = base_dir / np.linalg.norm(base_dir)

    # Vectors from base_start to proj points
    v1 = proj_start - base_start
    v2 = proj_end - base_start

    # Project onto base_dir
    t1 = np.dot(v1, base_dir_norm)
    t2 = np.dot(v2, base_dir_norm)

    # Get projection segment length (clipped to the base segment)
    projected_length = np.clip(np.abs(t2 - t1), 0, base_seg.length)
    return projected_length


def are_colinear(line1, line2, tol=1e-4):
    """Check if two lines are approximately colinear"""
    # Convert to vectors
    p1, p2 = np.array(line1.coords)
    q1, q2 = np.array(line2.coords)

    v1 = p2 - p1
    v2 = q2 - q1

    # Normalize
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot_product = np.dot(v1, v2)
    return abs(abs(dot_product) - 1) < tol  # direction is aligned or opposite

def subtract_colinear_doors(roomLines, doorLines, tol=1e-3, overlap_thresh=0.8):
    updated_lines = []
    used_door_indices = set()

    # For each set of room lines in a set of 4, find the two long room lines and two short lines
    assert len(doorLines) % 4 == 0, 'The number of door lines should be multiples of 4'
    long_segment_indices = []
    for i in range(0, len(doorLines), 4):
        group = doorLines[i:i+4]
        
        # Compute lengths of each segment in the group
        lengths = [np.linalg.norm(seg[0] - seg[1]) for seg in group]
        
        # Get indices of the two longest segments in the group
        group_indices = list(range(i, i+4))
        sorted_indices = sorted(zip(group_indices, lengths), key=lambda x: x[1], reverse=True)
        long_indices = [sorted_indices[0][0], sorted_indices[1][0]]
        
        long_segment_indices.extend(long_indices)

    # Remove overlapping door segments from room segments
    for room_seg_coords in roomLines:
        room_seg = LineString(room_seg_coords)
        segments_to_subtract = []

        for i, door_seg_coords in enumerate(doorLines):
            # Skip short segments
            if i not in long_segment_indices:
                continue

            door_seg = LineString(door_seg_coords)
            if room_seg.distance(door_seg) < tol and are_colinear(room_seg, door_seg):
                segments_to_subtract.append(door_seg)
                used_door_indices.add(i)
                # Use projection-based overlap
                # proj_len = project_segment_onto_another(room_seg, door_seg)
                # door_len = door_seg.length

                # if proj_len / door_len >= overlap_thresh:
                #     segments_to_subtract.append(door_seg)
                #     used_door_indices.add(i)

        if segments_to_subtract:
            diff = room_seg
            for door in segments_to_subtract:
                diff = diff.difference(door, 0.05)

            if diff.is_empty:
                continue
            elif diff.geom_type == "LineString":
                updated_lines.append(np.array(diff.coords))
            elif diff.geom_type == "MultiLineString":
                for part in diff.geoms:
                    updated_lines.append(np.array(part.coords))
        else:
            updated_lines.append(room_seg_coords)

    remaining_door_lines = [doorLines[i] for i in range(len(doorLines)) if i not in used_door_indices]

    return np.array(updated_lines), np.array(remaining_door_lines)

def door_connects_two_rooms(door_line, room_lines, tol=1e-3):
    """
    Checks if both endpoints of a door_line lie near different room_lines.
    """
    p1, p2 = map(Point, door_line.coords)

    touching_segments = []
    for i, room_coords in enumerate(room_lines):
        room_line = LineString(room_coords)
        if p1.distance(room_line) < tol or p2.distance(room_line) < tol:
            touching_segments.append(i)

    return len(set(touching_segments)) >= 2

def find_doors_connecting_rooms(roomLines, doorLines, tol=0.05):
    """
    Returns a list of door lines that connect two different room lines.
    """
    connecting_doors = []

    for door_coords in doorLines:
        door_line = LineString(door_coords)
        if door_connects_two_rooms(door_line, roomLines, tol):
            connecting_doors.append(door_coords)

    return np.array(connecting_doors)