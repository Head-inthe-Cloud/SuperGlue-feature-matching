import os
import json
import tqdm
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *
from src.monocular_plane_estimation_module.monocular_plane_estimation import get_pcd_from_full_obs, get_projection_from_pcd
from src.monocular_plane_estimation_module.pointcloud import subsample_point_cloud, extract_points_at_height
from src.f3loc.generate_desdf import make_desdf_map
import glob

class PALMSPlusLoader(Dataset):
    def __init__(self, data_dict, depth_pcd_dir, mde, scale_alignment_mode):
        """
        Args:
            data_dict (dict): {building: [list of obs_paths]}
        """
        self.data = []
        for building, obs_paths in data_dict.items():
            for obs_path in tqdm.tqdm(obs_paths, desc=f'Prepping data for {building}'):
                obs_id = os.path.basename(obs_path)
                pcd_path = os.path.join(depth_pcd_dir, building, obs_id, mde)
                pcd = get_pcd_from_full_obs(obs_path, pcd_path, mde=mde, use_ICP=False, scale_alignment_mode=scale_alignment_mode)
                pcd = subsample_point_cloud(pcd)
                filtered_pcd = extract_points_at_height(pcd, target_height=0) # Extract points at camera height
                projection, planes, obs_oris = get_projection_from_pcd(filtered_pcd, show_result=False)

                label_path = os.path.join(obs_path, 'label.txt')
                assert os.path.exists(label_path), f'Path not found: {label_path}'

                x, y, theta = load_label(label_path)
                self.data.append({
                    'building': building,
                    'obs_path': obs_path,
                    'planes': planes,
                    'obs_oris': obs_oris,
                    'label': np.array([x, y, theta])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PALMSLoader(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): {building: [list of obs_paths]}
        """
        self.data = []
        for building, obs_paths in data_dict.items():
            for obs_path in obs_paths:
                planes_path = os.path.join(obs_path, 'detectedPlanes.json')
                label_path = os.path.join(obs_path, 'label.txt')
                assert os.path.exists(planes_path), f'Path not found: {planes_path}'
                assert os.path.exists(label_path), f'Path not found: {label_path}'

                planes = load_planes_json(planes_path)
                x, y, theta = load_label(label_path)

                self.data.append({
                    'building': building,
                    'obs_path': obs_path,
                    'planes': planes,
                    'label': np.array([x, y, theta])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


from src.f3loc.generate_desdf import raycast_observation_torch
class EigenRayLoader(Dataset):
    def __init__(self, data_dict, fp_paths, resolution=0.1, orn_slice=36,  max_dist=50):
        """
        Args:
            data_dict (dict): {building: [list of obs_paths]}
        """
        self.data = []
        for building, obs_paths in data_dict.items():
            # Load building map
            fp_path = fp_paths[building]
            vector_map = load_map_csv(fp_path)
            vector_map, map_transform, map_theta = rotate_segments_to_landscape(vector_map)
            binary_map = segments_to_binary_map(vector_map, cell_size=resolution)

            for obs_path in obs_paths:
                label_path = os.path.join(obs_path, 'label.txt')
                assert os.path.exists(label_path), f'Path not found: {label_path}'

                x, y, theta = load_label(label_path)
                gt_loc = apply_transformation_to_points(np.array([[x, y]]), map_transform)[0]

                obs_1d = raycast_observation_torch(
                    binary_map,
                    origin=gt_loc,
                    orientation=0,
                    fov=to_radian(360),
                    max_dist=max_dist,
                    resolution=resolution,
                    degree_interval= 360 / orn_slice
                )

                self.data.append({
                    'building': building,
                    'obs_path': obs_path,
                    'obs_1d': obs_1d,
                    'gt_loc': np.array(gt_loc)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EigenPlaneLoader(Dataset):
    def __init__(self, data_dict, fp_paths,  max_dist=10):
        """
        Args:
            data_dict (dict): {building: [list of obs_paths]}
        """
        self.data = []
        for building, obs_paths in data_dict.items():
            # Load building map
            fp_path = fp_paths[building]
            vector_map = load_map_csv(fp_path)
            vector_map, map_transform, map_theta = rotate_segments_to_landscape(vector_map)

            for obs_path in tqdm.tqdm(obs_paths, desc=f'Preparing dataset for building {building}'):
                label_path = os.path.join(obs_path, 'label.txt')
                assert os.path.exists(label_path), f'Path not found: {label_path}'
                x, y, theta = load_label(label_path)
                gt_loc = apply_transformation_to_points(np.array([[x, y]]), map_transform)[0]

                gt_obs_path = os.path.join(obs_path, 'gt_obs.csv')
                if os.path.exists(gt_obs_path):
                    planes = load_map_csv(gt_obs_path)
                else:
                    planes = filter_visible_segments(vector_map=vector_map, location=gt_loc, radius=max_dist)
                    save_map_csv(planes, gt_obs_path)

                self.data.append({
                    'building': building,
                    'obs_path': obs_path,
                    'planes': planes,
                    'gt_loc': np.array(gt_loc)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

from utils.dataloader_utils import pano2persp, read_s3d_floorplan, get_intrinsics_from_pano2persp
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class Structured3DDataset(Dataset):
    def __init__(self, root_dir, step_size, FOV, num_images, pitch=0, size=(512, 512), skip_scenes=None):
        """

        Args:
            root_dir (string): Directory with all the scenes.
            step_size (float): Number of degrees of rotation between images.
            FOV (float): Horizontal FOV of each cropped image.
            num_images (int): Number of images to crop.
            size (tuple): Contains size of perspective images, (width, height)
            skip_scenes: A list of names of the scenes to skip
        """
        self._size = size  # Height, Width
        self._FOV = FOV
        self._num_images = num_images
        self._pitch = pitch
        self._step_size = step_size
        self._root_dir = root_dir
        self._scenes = sorted(glob.glob(os.path.join(self._root_dir, 'scene*')))
        if skip_scenes is not None:
            self._scenes = [scene for scene in self._scenes if os.path.basename(scene) not in skip_scenes]
            print(f'Skipped {len(skip_scenes)} Scenes')
        self._intrinsics = get_intrinsics_from_pano2persp(FOV, size)

        # path is 2D_rendering/<listdir>/panorama/full
        self._render_paths = []
        for scene_path in self._scenes:
            render_paths = sorted(glob.glob(os.path.join(scene_path, '**/full/rgb_rawlight.png'), recursive=True))
            for render_path in render_paths:
                self._render_paths.append(render_path)

        # print(self._render_paths)
        # for each scene there are multiple 2D renders?


    def get_camera_pose_from_direction(self, direction_deg, pitch=0):
        """
        Compute a 4x4 camera-to-world transformation matrix from yaw and pitch.
        Matches the orientation used in pano2persp (camera looks along -Z, yaw includes -180 flip).

        Args:
            direction_deg (float): Yaw angle in degrees in floor plan (world) reference frame.
            pitch (float): Pitch angle in degrees (positive looks upward).

        Returns:
            np.ndarray: 4x4 camera-to-world pose matrix.
        """
        # Match pano2persp's yaw: subtract 180 degrees (camera looks -Z)
        yaw_rad = np.radians(direction_deg - 180)
        yaw_rad = (yaw_rad + np.pi) % (2 * np.pi) - np.pi # Normalize it to (-pi, pi)

        # Rotation around Y axis (yaw)
        R_yaw = np.array([
            [ np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [ 0,               1, 0              ],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])

        # Rotation around camera's local X axis (pitch)
        pitch_rad = np.radians(pitch)  # negative pitch to match pano2persp (down is positive)
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad),  np.cos(pitch_rad)],
        ])

        # Total rotation: apply yaw first, then pitch in the local (rotated) frame
        R = R_yaw @ R_pitch

        # Assemble transformation matrix (no translation)
        T = np.eye(4)
        T[:3, :3] = R
        return T


    def get_vector_map(self, anno_path):
        with open(anno_path, "r") as f:
            annos = json.load(f)
        _, room_lines, door_lines, _ = read_s3d_floorplan(annos)

        # Create vector map
        room_lines, door_lines = subtract_colinear_doors(room_lines, door_lines)
        connecting_doors = find_doors_connecting_rooms(room_lines, door_lines)
        if len(connecting_doors) > 0 :
            vector_map = np.concatenate([room_lines, connecting_doors], axis=0)
        else:
            vector_map = room_lines

        return vector_map


    def __len__(self):
        return len(self._render_paths)

    def __getitem__(self, idx):
        render = self._render_paths[idx]
        obs_path = render.split('panorama')[0][:-1]
        scene_path = render.split('2D_rendering')[0][:-1]
        scene_id = os.path.basename(scene_path)

        label_path = os.path.join(obs_path, 'panorama', 'camera_xyz.txt')
        anno_path = os.path.join(scene_path, 'annotation_3d.json')
        if not os.path.exists(anno_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing annotation or label for {render}")
        
        img = cv2.imread(render)
        
        vector_map = self.get_vector_map(anno_path)

        # Load labels
        with open(label_path, 'r') as f:
            x, y, z = f.read().split(' ')
        label = np.array([x, y, z], dtype=float)

        # pano image starts and ends at -90
        angles = [-1 * i * self._step_size - 90 for i in range(self._num_images)] # in pano reference frame
        actualAngles = [i * self._step_size for i in range(self._num_images)]  # in floorplan reference frame

        imgs = [
            pano2persp(img, self._FOV, angle, self._pitch, 0, self._size) for angle in angles
        ]
        
        poses = [self.get_camera_pose_from_direction(direction, self._pitch) for direction in actualAngles]

        return imgs, poses, vector_map, label, obs_path, scene_id


class S3DEigenRayLoader(Dataset):
    def __init__(self, data_dict, resolution=0.1, orn_slice=36,  max_dist=50):
        """
        Args:
            data_dict (dict): {scene_id: [list of obs_paths]}
        """
        self.data = []
        self.resolution = resolution
        self.orn_slice = orn_slice
        self.max_dist = max_dist
        self.current_scene_id = ''
        self.map_transform = None
        self.binary_map = None

        for scene_id, obs_paths in tqdm.tqdm(data_dict.items(), desc='Preparing dataset'):
            scene_path = obs_paths[0].split('2D_rendering')[0]
            fp_desdf_path = os.path.join(scene_path, f'{scene_id}_{max_dist}m_{orn_slice}.npy')

            for obs_path in obs_paths:
                self.data.append({
                    'scene_id': scene_id,
                    'obs_path': obs_path,
                    'fp_desdf_path': fp_desdf_path,
                })

    def get_vector_map(self, anno_path):
        with open(anno_path, "r") as f:
            annos = json.load(f)
        _, room_lines, door_lines, _ = read_s3d_floorplan(annos)

        # Create vector map
        room_lines, door_lines = subtract_colinear_doors(room_lines, door_lines)
        connecting_doors = find_doors_connecting_rooms(room_lines, door_lines)
        if len(connecting_doors) > 0 :
            vector_map = np.concatenate([room_lines, connecting_doors], axis=0)
        else:
            vector_map = room_lines

        return vector_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_data = self.data[idx]
        obs_path = current_data['obs_path']
        scene_path = obs_path.split('2D_rendering')[0]

        if self.current_scene_id != current_data['scene_id']:
            self.current_scene_id = current_data['scene_id']

            # Load building map
            anno_path = os.path.join(scene_path, 'annotation_3d.json')
            vector_map = self.get_vector_map(anno_path)
            vector_map, map_transform, map_theta = rotate_segments_to_landscape(vector_map)
            self.binary_map = segments_to_binary_map(vector_map, cell_size=self.resolution)
            self.map_transform = map_transform

            fp_desdf_path = current_data['fp_desdf_path']
            if not os.path.exists(fp_desdf_path):
                print(f'Creating desdf for floor plan stored at {fp_desdf_path} \n This could take a while ...')
                make_desdf_map(self.binary_map, save_path=fp_desdf_path, orn_slice=self.orn_slice, resolution=self.resolution, max_dist=self.max_dist)


        # get label
        label_path = os.path.join(obs_path, 'panorama', 'camera_xyz.txt')
        
        # Load labels
        with open(label_path, 'r') as f:
            x, y, z = f.read().split(' ')
        label = np.array([x, y, z], dtype=float) # xyz are in milimeters
        label /= 1000 # Change it back to meters

        location = apply_transformation_to_points(np.reshape(label[:2], (1, -1)), self.map_transform)[0]
        theta = 0
        current_data['label'] = np.concatenate((location, np.array([theta])))


        # get obs_1d
        obs_1d = raycast_observation_torch(
            self.binary_map,
            origin=location,
            orientation=0,
            fov=to_radian(360),
            max_dist=self.max_dist,
            resolution=self.resolution,
            degree_interval= 360 / self.orn_slice
        )
        current_data['obs_1d'] = obs_1d

 
        return current_data



class S3DEigenPlaneLoader(Dataset):
    def __init__(self, data_dict, resolution=0.1,  max_dist=50):
        """
        Args:
            data_dict (dict): {scene_id: [list of obs_paths]}
        """
        self.data = []
        self.resolution = resolution
        self.max_dist = max_dist
        self.current_scene_id = ''
        self.map_transform = None
        self.map_theta = 0
        self.vector_map = None

        for scene_id, obs_paths in tqdm.tqdm(data_dict.items(), desc='Preparing dataset'):
            for obs_path in obs_paths:
                self.data.append({
                    'scene_id': scene_id,
                    'obs_path': obs_path
                })

    def get_vector_map(self, anno_path):
        with open(anno_path, "r") as f:
            annos = json.load(f)
        _, room_lines, door_lines, _ = read_s3d_floorplan(annos)

        # Create vector map
        room_lines, door_lines = subtract_colinear_doors(room_lines, door_lines)
        connecting_doors = find_doors_connecting_rooms(room_lines, door_lines)
        if len(connecting_doors) > 0 :
            vector_map = np.concatenate([room_lines, connecting_doors], axis=0)
        else:
            vector_map = room_lines

        return vector_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_data = self.data[idx]
        obs_path = current_data['obs_path']
        scene_path = obs_path.split('2D_rendering')[0]

        if self.current_scene_id != current_data['scene_id']:
            self.current_scene_id = current_data['scene_id']

            # Load building map
            anno_path = os.path.join(scene_path, 'annotation_3d.json')
            vector_map = self.get_vector_map(anno_path)
            self.vector_map, map_transform, self.map_theta = rotate_segments_to_landscape(vector_map)
            self.map_transform = map_transform

        # get label
        label_path = os.path.join(obs_path, 'panorama', 'camera_xyz.txt')
        
        # Load labels
        with open(label_path, 'r') as f:
            x, y, z = f.read().split(' ')
        label = np.array([x, y, z], dtype=float) # xyz are in milimeters
        label /= 1000 # Change it back to meters

        location = apply_transformation_to_points(np.reshape(label[:2], (1, -1)), self.map_transform)[0]

        theta = 0
        current_data['label'] = np.concatenate((location, np.array([theta])))

        # get obs_planes
        obs_planes = filter_visible_segments(vector_map=self.vector_map, location=location, radius=self.max_dist)
        current_data['obs_planes'] = obs_planes

        return current_data