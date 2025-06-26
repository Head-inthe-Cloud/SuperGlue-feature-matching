import cv2
import numpy as np

# Most of the code here are from the LASER Github

def rot_verts(verts, rot):
    theta = np.deg2rad(rot)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], float
    )
    org_shape = verts.shape
    return ((R @ verts.reshape(-1, 2).T).T).reshape(org_shape)

def is_polygon_clockwise(lines):
    return np.sum(np.cross(lines[:, 0], lines[:, 1], axis=-1)) > 0

def poly_verts_to_lines_append_head(verts):
    n_verts = verts.shape[0]
    if n_verts == 0:
        return
    assert n_verts > 1
    verts = np.concatenate([verts, verts[0:1]], axis=0)  # append head to tail
    lines = np.stack([verts[:-1], verts[1:]], axis=1)  # N,2,2
    return lines

def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices"""
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons

def sample_points_from_lines(lines, interval):
    n_lines = lines.shape[0]
    lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
    n_samples_per_line = np.ceil(lengths / interval).astype(np.int)

    lines_normal = (lines[:, 0] - lines[:, 1]) / np.maximum(
        np.linalg.norm(lines[:, 0] - lines[:, 1], axis=-1, keepdims=True), 1e-8
    )  # N,2
    lines_normal = np.stack([lines_normal[:, 1], -lines_normal[:, 0]], axis=1)

    samples = []
    samples_normal = []
    for l in range(n_lines):
        if n_samples_per_line[l] == 0:
            continue
        p = np.arange(n_samples_per_line[l]).reshape(-1, 1) / n_samples_per_line[l] + (
            0.5 / n_samples_per_line[l]
        )  # uniform sampling
        # p = np.random.rand(n_samples_per_line[l]).reshape(-1,1) # random sampling
        samples.append(p * lines[l : l + 1, 0] + (1 - p) * lines[l : l + 1, 1])
        samples_normal.append(np.repeat(lines_normal[l].reshape(1, 2), p.size, axis=0))
    samples = np.concatenate(samples, axis=0)
    samples_normal = np.concatenate(samples_normal, axis=0)
    return samples, samples_normal

def read_s3d_floorplan(annos):
    """visualize floorplan"""
    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos["semantics"]:
        for planeID in semantic["planeID"]:
            if annos["planes"][planeID]["type"] == "floor":
                planes.append({"planeID": planeID, "type": semantic["type"]})

        if semantic["type"] == "outwall":
            outerwall_planes = semantic["planeID"]

    # extract hole vertices
    lines_holes = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["window", "door"]:
            for planeID in semantic["planeID"]:
                lines_holes.extend(
                    np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
                )
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc["coordinate"] for junc in annos["junctions"]])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos["planeLineMatrix"][plane["planeID"]]))[
            0
        ].tolist()
        junction_pairs = [
            np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane["type"]])
    """
    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    polygons.append([outerwall_polygon[0], 'outwall'])
    """

    junctions = np.array([junc["coordinate"][:2] for junc in annos["junctions"]])
    door_lines = []
    window_lines = []
    room_lines = []
    n_rooms = 0
    for (polygon, poly_type) in polygons:
        polygon = junctions[np.array(polygon)] / 1000.0  # mm to meter
        lines = poly_verts_to_lines_append_head(polygon)
        if not is_polygon_clockwise(lines):
            lines = poly_verts_to_lines_append_head(np.flip(polygon, axis=0))
        if poly_type == "door":
            door_lines.append(lines)
        elif poly_type == "window":
            window_lines.append(lines)
        else:
            n_rooms += 1
            room_lines.append(lines)

    room_lines = np.concatenate(room_lines, axis=0)
    door_lines = (
        np.zeros((0, 2, 2), float)
        if len(door_lines) == 0
        else np.concatenate(door_lines, axis=0)
    )
    window_lines = (
        np.zeros((0, 2, 2), float)
        if len(window_lines) == 0
        else np.concatenate(window_lines, axis=0)
    )

    return n_rooms, room_lines, door_lines, window_lines

def pano2persp(img, fov, yaw, pitch, roll, size, RADIUS=128):

    equ_h, equ_w = img.shape[:2]
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    height, width = size
    wFOV = fov
    hFOV = float(height) / width * wFOV

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (180 - wFOV) / 2.0
    w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
    h_interval = h_len / (height - 1)
    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.zeros([height, width, 3], float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(yaw - 180))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-pitch))
    [R3, _] = cv2.Rodrigues(np.dot(R2 @ R1, x_axis) * np.radians(-roll))

    xyz = xyz.reshape(height * width, 3).T
    xyz = ((R3 @ R2 @ R1) @ xyz).T
    lat = np.arcsin(xyz[:, 2] / RADIUS)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lon = ((lon / np.pi + 1) * equ_cx).reshape(height, width)
    lat = ((-lat / np.pi * 2 + 1) * equ_cy).reshape(height, width)

    persp = cv2.remap(
        img,
        lon.astype(np.float32),
        lat.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return persp


def get_intrinsics_from_pano2persp(fov_deg, image_size):
    """
    Recover camera intrinsic matrix K from pano2persp parameters.
    
    Args:
        fov_deg (float): horizontal field of view in degrees
        image_size (tuple): (height, width) of the perspective image

    Returns:
        K (np.ndarray): 3x3 camera intrinsic matrix
    """
    height, width = image_size
    fov_rad = np.radians(fov_deg)
    f_x = (width / 2) / np.tan(fov_rad / 2)

    # pano2persp calculates vertical FOV as:
    # hFOV = (height / width) * wFOV
    fov_y_rad = (height / width) * fov_rad
    f_y = (height / 2) / np.tan(fov_y_rad / 2)

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    K = np.array([
        [f_x,   0,  c_x],
        [0,   f_y,  c_y],
        [0,     0,   1 ]
    ], dtype=np.float32)

    return K