import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

def convert_csv_to_structure3d(csv_path, output_path):
    # Step 1: Load 2D line segments from CSV
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        segments = [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in reader]

    # Step 2: Identify unique junctions
    junctions = {}
    junction_id = 0
    lines = []
    for x1, y1, x2, y2 in segments:
        p1 = (x1, y1, 0)
        p2 = (x2, y2, 0)
        if p1 not in junctions:
            junctions[p1] = junction_id
            junction_id += 1
        if p2 not in junctions:
            junctions[p2] = junction_id
            junction_id += 1
        lines.append((junctions[p1], junctions[p2], p1, p2))

    # Step 3: Create junction list
    junction_list = [
        {"ID": i, "coordinate": list(coord)} for coord, i in junctions.items()
    ]

    # Step 4: Create line list with direction and point
    line_list = []
    line_junction_matrix = np.zeros((len(lines), len(junctions)), dtype=int)
    for idx, (j1, j2, p1, p2) in enumerate(lines):
        direction = np.array(p2) - np.array(p1)
        direction /= np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else 1
        line_list.append({
            "ID": idx,
            "point": list(p1),
            "direction": direction.tolist(),
        })
        line_junction_matrix[idx][j1] = 1
        line_junction_matrix[idx][j2] = 1

    # Step 5: Create a single floor plane
    coords = np.array([j["coordinate"] for j in junction_list])
    centroid = coords.mean(axis=0).tolist()
    plane_list = [{
        "ID": 0,
        "type": "floor",
        "normal": [0, 0, 1],
        "centroid": centroid,
    }]

    # Step 6: Associate all lines with the one plane
    plane_line_matrix = np.ones((1, len(lines)), dtype=int).tolist()

    # Step 7: Semantics definition
    semantics = [{
        "ID": 0,
        "type": "floor",
        "planeID": [0]
    }]

    # Step 8: Build the structure3D dictionary
    structure3d_dict = {
        "junctions": junction_list,
        "lines": line_list,
        "planes": plane_list,
        "semantics": semantics,
        "planeLineMatrix": plane_line_matrix,
        "lineJunctionMatrix": line_junction_matrix.T.tolist(),
        "cuboids": [],
        "manhattan": []
    }

    # Step 9: Write to output JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structure3d_dict, f, indent=2)

    print(f"Saved Structure3D JSON to: {output_path}")

 # Example usage:
 # convert_csv_to_structure3d("path/to/floor.csv", "output/scene_0000/annotation_3d.json")