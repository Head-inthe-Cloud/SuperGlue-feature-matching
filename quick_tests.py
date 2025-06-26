# This file used to help testing the PALMS algorithm implemented on the iOS

import sys
import os
import shutil
import numpy as np
import torch
from utils.utils import *
from utils.visualization import *
import matplotlib.pyplot as plt
from src.CES import *
from src.simulator import *
from src.metrics import *
# from src.DL_matcher import ObsToFloorplanMatcher
from tqdm import tqdm
import json
import open3d as o3d
import scipy.spatial
from src.monocular_plane_estimation_module.monocular_plane_estimation import get_pcd_from_full_obs, get_projection_from_pcd, get_pcd_from_s3d_obs
from src.monocular_plane_estimation_module.pointcloud import extract_points_at_height, subsample_point_cloud
from src.monocular_plane_estimation_module.alignment import filter_vertical_plane_points
from src.f3loc.localization_utils import localize
from src.f3loc.generate_desdf import *
from src.f3loc.utils import visualize_depth_with_map

from utils.labeling_tool import LabelingTool
from utils.image_sampling import compute_geometric_horizontal_fov, visualize_plane_intersection

from utils.dataloader import Structured3DDataset


def show_top_locs(heatmap, binary_map, label, num_indices=5, resolution=0.1):
    flat_indices = np.argpartition(heatmap.ravel(), -num_indices)[-num_indices:]
    top_coords = np.array(np.unravel_index(flat_indices, heatmap.shape)).T  # shape: (n, 2)
    # Plot the result
    plt.imshow(binary_map, cmap='gray')
    img = plt.imshow(heatmap, cmap='viridis', origin='lower', alpha=0.5)
    plt.plot(label[0] / resolution, label[1] / resolution, 'o', color='green', alpha=0.5)

    # Plot top 10 as red dots
    for y, x in top_coords:
        plt.plot(x, y, 'ro')  # Note: matplotlib expects (x, y)

    plt.colorbar(img, label='Intensity')
    plt.title(f"Top {num_indices} Maxima on Heatmap")
    plt.show()


def visualize_heatmaps_with_metrics(
    final_heatmap, gt_heatmap, label, resolution,
    binary_map=None, num_top=5
):
    # Convert label to pixel coordinates
    label_pix = (label[0] / resolution, label[1] / resolution)
    tolerance = 1 / resolution

    # Compute metrics for predicted heatmap
    rank_pred = ranking_score(final_heatmap, gt_loc=label_pix, tolerance=tolerance)
    conf_pred = confidence_at_truth(final_heatmap, gt_loc=label_pix, tolerance=tolerance)
    xent_pred = gaussian_cross_entropy_score(final_heatmap, gt_loc=label_pix, show_gaussian=False)

    # Compute metrics for GT heatmap
    rank_gt = ranking_score(gt_heatmap, gt_loc=label_pix, tolerance=tolerance)
    conf_gt = confidence_at_truth(gt_heatmap, gt_loc=label_pix, tolerance=tolerance)
    xent_gt = gaussian_cross_entropy_score(gt_heatmap, gt_loc=label_pix, show_gaussian=False)

    # Cross-entropy between maps
    cross_entropy = cross_entropy_score(final_heatmap, gt_heatmap)

    # Setup figure
    fig, axs = plt.subplots(3, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [6, 1, 1]})
    
    def plot_heatmap(ax, heatmap, title, label_pix, color='green'):
        if binary_map is not None:
            ax.imshow(binary_map, cmap='gray', alpha=0.6)
        im = ax.imshow(heatmap, cmap='viridis', origin='lower', alpha=0.6)
        ax.plot(label_pix[0], label_pix[1], 'o', color=color, markersize=6, label="GT Location", alpha=0.4)

        # Top-k points
        flat_indices = np.argpartition(heatmap.ravel(), -num_top)[-num_top:]
        top_coords = np.array(np.unravel_index(flat_indices, heatmap.shape)).T
        for y, x in top_coords:
            ax.plot(x, y, 'ro', markersize=4, alpha=0.2)

        ax.set_title(title)
        ax.legend()
        fig.colorbar(im, ax=ax)

    # Plot heatmaps
    plot_heatmap(axs[0, 0], gt_heatmap, "GT Heatmap", label_pix)
    plot_heatmap(axs[0, 1], final_heatmap, "Estimated Heatmap", label_pix)

    # Show GT metrics
    axs[1, 0].axis('off')
    axs[1, 0].text(0.1, 0.5,
        f"GT Heatmap\nRanking: {rank_gt:.3f}\nConfidence: {conf_gt:.3f}\nGaussian Xent: {xent_gt:.3f}",
        fontsize=10, verticalalignment='center'
    )

    # Show predicted metrics
    axs[1, 1].axis('off')
    axs[1, 1].text(0.1, 0.5,
        f"Pred Heatmap\nRanking: {rank_pred:.3f}\nConfidence: {conf_pred:.3f}\nGaussian Xent: {xent_pred:.3f}",
        fontsize=10, verticalalignment='center'
    )

    # Bottom row: cross-entropy
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')
    axs[2, 0].text(0.1, 0.5,
        f"Cross Entropy Score (Max: 1) Between GT and Est Heatmap: {cross_entropy:.4f}",
        fontsize=12, fontweight='bold', verticalalignment='center'
    )

    plt.tight_layout()
    plt.show()


def main_custom():
    # Custom dataset
    building = 'PS' # E2, BE, PS, SVC
    fp_path = {'E2': './maps/E2.csv',
            'BE': './maps/BE.csv',
            'PS': './maps/PS.csv', 
            'SVC': './maps/SVC.csv'} [building]
    
    alpha = 50
    mde = 'dp'
    scale_alignment_mode = 'ground'
    resolution = 0.1

    # F3Loc
    max_dist = 10
    orn_slice = 36
    fp_desdf_path = f'./maps/desdf/custom/{building}_{max_dist}m_{orn_slice}.npy'

    # Load map data
    vector_map = load_map_csv(fp_path)
    vector_map, map_transform, map_theta = rotate_segments_to_landscape(vector_map)

    binary_map = segments_to_binary_map(vector_map, cell_size=resolution)

    
    # Load desdf for GT heatmap creation
    if os.path.exists(fp_desdf_path):
        desdf_map = load_map_desdf(fp_desdf_path)
    else:
        print(f'Creating desdf for floor plan stored at {fp_path} \n This could take a while ...')
        # desdf_map = make_desdf_map_from_vector_map(vector_map, save_path=fp_desdf_path, orn_slice=orn_slice, resolution=resolution, max_dist=max_dist) # Default resolution to 1m
        desdf_map = make_desdf_map(binary_map, save_path=fp_desdf_path, orn_slice=orn_slice, resolution=resolution, max_dist=max_dist)

    map_mask = alpha_shape(vector_map, visualize=False)

    labeling_tool = LabelingTool()
    labeling_tool.floor_plan = vector_map

    # Get pcd
    obs_paths = glob.glob(f'/Users/cyqpp/Work/Research/Plane_Based_Localization/Dataset/PALMS+/{building}/Session*')
    obs_paths.sort()
    for obs_path in obs_paths: 
        obs_id = os.path.basename(obs_path)
        result_path = f'./results/PALMS+/depth_and_pcd/custom/{building}/{obs_id}/{mde}'
        label_path = os.path.join(obs_path, 'label.txt')
        os.makedirs(result_path, exist_ok=True)

        # PALMS+
        pcd = get_pcd_from_full_obs(obs_path, result_path, mde=mde, use_ICP=False, scale_alignment_mode=scale_alignment_mode, verbose=True)
        o3d.visualization.draw_geometries([pcd], window_name="Aligned Point Cloud")
        continue
        pcd = subsample_point_cloud(pcd)
        filtered_pcd = extract_points_at_height(pcd, target_height=0) # Extract points at camera height
        projection, obs_planes, obs_oris = get_projection_from_pcd(filtered_pcd, show_result=False)

        # PALMS
        # obs_planes = load_planes_json(os.path.join(obs_path, 'detectedPlanes.json'))
        # obs_oris = find_principal_orientations(obs_planes)

        # Visualize label
        labeling_tool.planes = obs_planes
        labeling_tool.load_label(label_path)
        labeling_tool.apply_transformation(map_transform)
        labeling_tool.view_label()

        # labeling_tool.rotate_planes(None, 180, update_plot=False)
        # labeling_tool.start_labeling()

        # obs_planes = load_planes_json(os.path.join(obs_path, 'detectedPlanes.json'))
        # labeling_tool.planes = obs_planes
        # labeling_tool.load_label(label_path)
        # labeling_tool.start_labeling()
        # labeling_tool.view_label()
        # labeling_tool.load_planes(pcd_obs_planes)
        # labeling_tool.view_label()
        

        label = np.array([labeling_tool.x, labeling_tool.y])
        label_pix = (label / resolution).astype(int)


        ###############
        #### PALMS+ ###
        ###############
        # Eigen_conv
        obs_planes = filter_visible_segments(vector_map=vector_map, location=label, radius=10)

        # scale_range = np.arange(2/3, 1.2, 0.1)
        scale_range = np.arange(1, 1.1, 0.1)
        # Visualize scaled planes
        # labeling_tool.scale_planes(None, scale_range[0], update_plot=False)
        # labeling_tool.view_label()

        # labeling_tool.scale_planes(None, scale_range[1] / scale_range[0], update_plot=False)
        # labeling_tool.view_label()

        scaled_heatmaps = []
        for obs_scale in scale_range:
            obs_planes_scaled = obs_planes * obs_scale
            # PALMS heatmap creation Prep
            gaussian_kernel_config = (7, 3)   # kernel size, sigma

            obs_thetas = [0 - obs_oris[0], 0 - obs_oris[0] + np.pi/2,  0 - obs_oris[0] + np.pi,  0 - obs_oris[0] + 3 * np.pi/2]
            # obs_thetas = [2 * np.pi / orn_slice * i for i in range(orn_slice)] # 0 ~ 355 degrees, split into n slices

            CES = Conv_CES(obs_planes_scaled, resolution=resolution, mode='weighted', gaussian_kernel_config=gaussian_kernel_config)
            if obs_scale == scale_range[0]:
                CES.visualize_kernels()

            # TEMP: Debug Eigenmap
            print("Label", label)
            print("Label Pix", label_pix)
            print("CES Center", -CES.min_x, -CES.min_y, [int(-CES.min_x / resolution), int(-CES.min_y / resolution)])
            visualize_kernel_overlap(binary_map, kernel=CES.CES_kernel, gt_loc=label_pix, kernel_center=np.array([int(-CES.min_x / resolution), int(-CES.min_y / resolution)]))
            
            sys.exit()

            heatmaps = []
            pre_obs_theta = 0
            for ori_idx, obs_theta in enumerate(obs_thetas):
                # Rotate the kernels and create new heatmaps
                CES.rotate(obs_theta - pre_obs_theta)
                # CES.visualize_kernels()

                heatmap = CES.create_heatmap(binary_fp=binary_map, kernel='com', visualize_heatmap=False, alpha=alpha)
                masked_heatmap = np.where(map_mask, heatmap, 0)
                heatmaps.append(masked_heatmap)
                pre_obs_theta = obs_theta

            heatmaps = np.array(heatmaps)
            intermediate_heatmap = np.max(heatmaps, axis=0)

            # Debug only
            if False:
                labeling_tool.planes = obs_planes_scaled
                labeling_tool.output_file = label_path
                labeling_tool.view_label()
                plt.imshow(binary_map, cmap='gray')
                img = plt.imshow(intermediate_heatmap, cmap='viridis', origin='lower', alpha=0.5)
                plt.plot(label[0] / resolution, label[1] / resolution, 'o', color='green', alpha=0.5)
                plt.colorbar(img, label='Intensity')
                plt.show()

            scaled_heatmaps.append(intermediate_heatmap)
        
        final_heatmap = np.max(scaled_heatmaps, axis=0)
        # final_heatmap = np.sum(scaled_heatmaps, axis=0)
        final_heatmap = normalize_heatmap(final_heatmap, as_prob_distribution=True)

        # Get GT heatmap
        obs_1d = raycast_observation_torch(
            binary_map,
            origin=label,
            orientation=0,
            fov=to_radian(360),
            max_dist=max_dist,
            resolution=resolution,
            degree_interval= 360 / orn_slice
        )

        prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = localize(
            torch.tensor(desdf_map), torch.tensor(obs_1d), orn_slice=orn_slice, lambd=40
        )
        gt_heatmap = prob_dist_pred
        gt_heatmap = normalize_heatmap(gt_heatmap, as_prob_distribution=True)
       
        # Debug only
        if False:
            uniform_heatmap = np.ones_like(gt_heatmap) / gt_heatmap.size
            # in the uniform case, xent should equal to logN. Note: it could get worse than uniform
            uniform_xent = cross_entropy_score(uniform_heatmap, gt_heatmap)
            # in the best case, xent sould equal to ent. 
            best_xent = cross_entropy_score(gt_heatmap, gt_heatmap)
            uniform_kl = kl_div_score(uniform_heatmap, gt_heatmap)
            best_kl = kl_div_score(gt_heatmap, gt_heatmap)
            print(f'uniform = {uniform_xent}, best = {best_xent}')
            print(f'uniform = {uniform_kl}, best = {best_kl}')
            sys.exit()

        assert np.isclose(np.sum(gt_heatmap), 1.0) and np.isclose(np.sum(final_heatmap), 1.0), 'Heatmaps should be probability distributions'
        visualize_heatmaps_with_metrics(
            final_heatmap, gt_heatmap, label, resolution,
            binary_map, num_top=5
        )
        

def main_s3d():
    data_dir = '/Users/cyqpp/Work/Research/Plane_Based_Localization/Dataset/Structured3D'
    
    # PALMS+ Settings
    alpha = 1
    mde = 'dp'
    scale_alignment_mode = 'overlap'
    resolution = 0.1

    # F3Loc
    max_dist = 10
    orn_slice = 36
    
    # Structured3D Setting
    num_images = 8
    step_size = 360 / num_images
    fov=60
    pitch = -15
    img_size = (1920, 1440)  # Height, Width


    dset = Structured3DDataset(data_dir, step_size=step_size, FOV=fov, num_images=num_images, pitch=pitch, size=img_size)
    intrinsics = dset._intrinsics

    # Loop over each observation
    for imgs, poses, vector_map, label, obs_path, scene_id in tqdm.tqdm(dset, desc='Estimating depths'):
        obs_id = os.path.basename(obs_path)
        frame_ids = list(range(len(imgs))) # Use 0 ~ n to name the frames
        fp_desdf_path = f'./maps/desdf/s3d/{scene_id}_{max_dist}m_{orn_slice}.npy'

        result_path = f'./results/PALMS+/depth_and_pcd/s3d/{scene_id}/{obs_id}/{mde}'
        os.makedirs(result_path, exist_ok=True)

        # Load map data
        vector_map, map_transform, map_theta = rotate_segments_to_landscape(vector_map)
        binary_map = segments_to_binary_map(vector_map, cell_size=resolution)
        
        label = apply_transformation_to_points(np.reshape(label[:2], (1, -1)), map_transform)
        map_mask = alpha_shape(vector_map, visualize=False)

        # For visualizations 
        labeling_tool = LabelingTool()
        labeling_tool.floor_plan = vector_map

        # Load desdf for GT heatmap creation
        if os.path.exists(fp_desdf_path):
            desdf_map = load_map_desdf(fp_desdf_path)
        else:
            print(f'Creating desdf for floor plan of {obs_id} \n This could take a while ...')
            # desdf_map = make_desdf_map_from_vector_map(vector_map, save_path=fp_desdf_path, orn_slice=orn_slice, resolution=resolution, max_dist=max_dist) # Default resolution to 1m
            desdf_map = make_desdf_map(binary_map, save_path=fp_desdf_path, orn_slice=orn_slice, resolution=resolution, max_dist=max_dist)
            
        # PALMS+
        pcd = get_pcd_from_s3d_obs(imgs, poses, intrinsics, frame_ids, result_path, mde=mde, use_ICP=False, scale_alignment_mode=scale_alignment_mode, verbose=True)
        o3d.visualization.draw_geometries([pcd], window_name="Aligned Point Cloud")
        continue
        filtered_pcd = extract_points_at_height(pcd, target_height=0) # Extract points at camera height
        projection, obs_planes, obs_oris = get_projection_from_pcd(filtered_pcd, show_result=False)

        # PALMS
        # obs_planes = load_planes_json(os.path.join(obs_path, 'detectedPlanes.json'))
        # obs_oris = find_principal_orientations(obs_planes)

        # Visualize label
        labeling_tool.planes = obs_planes
        labeling_tool.load_label(label=label)
        labeling_tool.apply_transformation(map_transform)
        labeling_tool.view_label()

        sys.exit()

        # labeling_tool.rotate_planes(None, 180, update_plot=False)
        # labeling_tool.start_labeling()

        # obs_planes = load_planes_json(os.path.join(obs_path, 'detectedPlanes.json'))
        # labeling_tool.planes = obs_planes
        # labeling_tool.load_label(label_path)
        # labeling_tool.start_labeling()
        # labeling_tool.view_label()
        # labeling_tool.load_planes(pcd_obs_planes)
        # labeling_tool.view_label()
        

        label = np.array([labeling_tool.x, labeling_tool.y])
        label_pix = (label / resolution).astype(int)


        ###############
        #### PALMS+ ###
        ###############
        # Eigen_conv
        obs_planes = filter_visible_segments(vector_map=vector_map, location=label, radius=10)

        # scale_range = np.arange(2/3, 1.2, 0.1)
        scale_range = np.arange(1, 1.1, 0.1)
        # Visualize scaled planes
        # labeling_tool.scale_planes(None, scale_range[0], update_plot=False)
        # labeling_tool.view_label()

        # labeling_tool.scale_planes(None, scale_range[1] / scale_range[0], update_plot=False)
        # labeling_tool.view_label()

        scaled_heatmaps = []
        for obs_scale in scale_range:
            obs_planes_scaled = obs_planes * obs_scale
            # PALMS heatmap creation Prep
            gaussian_kernel_config = (7, 3)   # kernel size, sigma

            obs_thetas = [0 - obs_oris[0], 0 - obs_oris[0] + np.pi/2,  0 - obs_oris[0] + np.pi,  0 - obs_oris[0] + 3 * np.pi/2]
            # obs_thetas = [2 * np.pi / orn_slice * i for i in range(orn_slice)] # 0 ~ 355 degrees, split into n slices

            CES = Conv_CES(obs_planes_scaled, resolution=resolution, mode='weighted', gaussian_kernel_config=gaussian_kernel_config)
            if obs_scale == scale_range[0]:
                CES.visualize_kernels()

            heatmaps = []
            pre_obs_theta = 0
            for ori_idx, obs_theta in enumerate(obs_thetas):
                # Rotate the kernels and create new heatmaps
                CES.rotate(obs_theta - pre_obs_theta)
                # CES.visualize_kernels()

                heatmap = CES.create_heatmap(binary_fp=binary_map, kernel='com', visualize_heatmap=False, alpha=alpha)
                masked_heatmap = np.where(map_mask, heatmap, 0)
                heatmaps.append(masked_heatmap)
                pre_obs_theta = obs_theta

            heatmaps = np.array(heatmaps)
            intermediate_heatmap = np.max(heatmaps, axis=0)

            # Debug only
            if False:
                labeling_tool.planes = obs_planes_scaled
                labeling_tool.output_file = label_path
                labeling_tool.view_label()
                plt.imshow(binary_map, cmap='gray')
                img = plt.imshow(intermediate_heatmap, cmap='viridis', origin='lower', alpha=0.5)
                plt.plot(label[0] / resolution, label[1] / resolution, 'o', color='green', alpha=0.5)
                plt.colorbar(img, label='Intensity')
                plt.show()

            scaled_heatmaps.append(intermediate_heatmap)
        
        final_heatmap = np.max(scaled_heatmaps, axis=0)
        # final_heatmap = np.sum(scaled_heatmaps, axis=0)
        final_heatmap = normalize_heatmap(final_heatmap, as_prob_distribution=True)

        # Get GT heatmap
        obs_1d = raycast_observation_torch(
            binary_map,
            origin=label,
            orientation=0,
            fov=to_radian(360),
            max_dist=max_dist,
            resolution=resolution,
            degree_interval= 360 / orn_slice
        )

        prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = localize(
            torch.tensor(desdf_map), torch.tensor(obs_1d), orn_slice=orn_slice, lambd=40
        )
        gt_heatmap = prob_dist_pred
        gt_heatmap = normalize_heatmap(gt_heatmap, as_prob_distribution=True)
       
        # Debug only
        if False:
            uniform_heatmap = np.ones_like(gt_heatmap) / gt_heatmap.size
            # in the uniform case, xent should equal to logN. Note: it could get worse than uniform
            uniform_xent = cross_entropy_score(uniform_heatmap, gt_heatmap)
            # in the best case, xent sould equal to ent. 
            best_xent = cross_entropy_score(gt_heatmap, gt_heatmap)
            uniform_kl = kl_div_score(uniform_heatmap, gt_heatmap)
            best_kl = kl_div_score(gt_heatmap, gt_heatmap)
            print(f'uniform = {uniform_xent}, best = {best_xent}')
            print(f'uniform = {uniform_kl}, best = {best_kl}')
            sys.exit()

        assert np.isclose(np.sum(gt_heatmap), 1.0) and np.isclose(np.sum(final_heatmap), 1.0), 'Heatmaps should be probability distributions'
        visualize_heatmaps_with_metrics(
            final_heatmap, gt_heatmap, label, resolution,
            binary_map, num_top=5
        )
        

if __name__ == '__main__':
    main_s3d()
    # data_dir = '/media/nvme1/ychen827/Structured3DData/data/Structured3D'
    # result_dir = './results/Structured3D'
    # model_name = 'dp'

    # Split all scenes into 4 groups so they can run on different GPUs
    # group = 3 # 0 ~ 3
    # all_scenes = [os.path.basename(scene_path) for scene_path in glob.glob(os.path.join(data_dir, 'scene*'))]
    # skip_scenes = all_scenes[:group * 125] + all_scenes[(group + 1) * 125:]

    # dset = Structure3DDataset(data_dir, step_size=360 / 8, FOV=60, num_images=8, pitch=-30, size=(1920, 1440), skip_scenes=skip_scenes)
    # intrinsics = dset._intrinsics
    
    # device = f'cuda:{group}'
    # mde = MDE(model_name=model_name, device=device)

    # for imgs, camViewDirections, sceneMap, label, obs_path in tqdm.tqdm(dset, desc='Estimating depths'):
    #     for i, img in enumerate(imgs):
    #         depth_dir = os.path.join(obs_path, 'dp_depth')
    #         if not os.path.exists(depth_dir):
    #             os.makedirs(depth_dir, exist_ok=True)
    #         depth_path = os.path.join(depth_dir, f'{i}_30.npy')
    #         if os.path.exists(depth_path):
    #             continue
    #         est_depth = mde.estimate_depth(image=img, intrinsics=intrinsics)
    #         if False:    
    #             plt.imshow(est_depth)
    #             plt.savefig(os.path.join(result_dir, 'depth.png'))
    #             plt.imshow(img)
    #             plt.savefig(os.path.join(result_dir, 'img.png'))
    #             sys.exit()

    #         np.save(depth_path, est_depth)
    #         print(f"{model_name} Depths saved to {depth_path}")
