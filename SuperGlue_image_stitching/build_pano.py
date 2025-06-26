import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

from pathlib import Path
import argparse
import random
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, frame2tensor,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def rotate_intrinsics_custom(K, original_size, direction='cw'):
    # rotate intrinsic 90 degree clockwise
    h, w = original_size
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    if direction == 'cw':
        K_new = np.array([
            [fy, 0, cy],
            [0, fx, w - cx],
            [0, 0, 1]
        ], dtype=np.float32)
    elif direction == 'ccw':
        K_new = np.array([
            [fy, 0, h - cy],
            [0, fx, cx],
            [0, 0, 1]
        ], dtype=np.float32)

    return K_new

def load_full_obs(obs_path, data_keys=None, return_numpy=True):
    # Load images, arkit_depths, confidence maps, intrinsics, poses, ARKit detected planes, and time stamps
    if data_keys is None:
        data_keys = ['images', 'arkit_depths', 'confidences', 'intrinsics', 
                     'poses', 'time_stamps', 'paths', 'frame_ids', 'metadata']

    # Images should be in .png format, arkit_depths, intrinsics, and poses should be in .json format
    images = []
    arkit_depths = []
    confidences = []
    intrinsics = []
    poses = []
    time_stamps = []
    metadata = None

    paths = {'image_paths': [], 
             'arkit_depth_paths': [],
             'confidence_paths': [], 
             'intrinsics_paths': [],
             'pose_paths': [],
             'planes_path': None,
             'time_stamps_path': None,
             'label_path': None,
             'metadata_path': None}

    planes_path = os.path.join(obs_path, 'detectedPlanes.json')
    paths['planes_path'] = planes_path
    time_stamps_path = os.path.join(obs_path, 'timeStamps.json')
    paths['time_stamps_path'] = time_stamps_path
    label_path = os.path.join(obs_path, 'label.txt')
    paths['label_path'] = label_path
    metadata_path = os.path.join(obs_path, 'metadata.json')
    paths['metadata_path']= metadata_path

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
        if 'images' in data_keys:
            image = cv2.imread(image_path)
            height, width, _ = image.shape  # Get the dimensions of the original image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            images.append(image)

        # Load ARKit depth
        if 'arkit_depths' in data_keys:
            with open(arkit_depth_path, 'r') as f:
                depth = json.load(f)
            depth = np.array(depth, dtype=np.float32)
            depth = depth.reshape((192, 256))  # Adjust to LiDAR depth resolution
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
            arkit_depths.append(depth)

        # Load confidence map
        if 'confidences' in data_keys:
            confidence = cv2.imread(confidence_path)
            confidences.append(confidence)
        
        # Load intrinsics
        if 'intrinsics' in data_keys:
            with open(intrinsics_path, 'r') as file:
                data = json.load(file)
                camera_intrinsic = np.array(data["data"], dtype=np.float32)
                if camera_intrinsic.shape != (3, 3):
                    raise ValueError("Loaded intrinsic matrix is not 3x3.")
                intrinsics.append(camera_intrinsic)

        # Load pose
        if 'poses' in data_keys:
            with open(pose_path, 'r') as file:
                data = json.load(file)
                pose = np.array(data["data"], dtype=np.float32)
                if pose.shape != (4, 4):
                    raise ValueError("Loaded transformation matrix is not 4x4.")
                poses.append(pose)

        # Load label
        if 'label' in data_keys:
            x, y, theta = load_label(label_path)
            theta = np.deg2rad(theta)
            label = np.array([x, y, theta])

        # Load metadata
        if 'metadata' in data_keys:
            with open(metadata_path, 'r') as file:
                data = json.load(file)
                metadata = data
    
    if return_numpy:
        images = np.array(images)
        arkit_depths = np.array(arkit_depths)
        confidences = np.array(confidences)
        intrinsics = np.array(intrinsics)
        poses = np.array(poses)
        time_stamps = np.array(time_stamps)
        frame_ids = np.array(frame_ids)

        for key in paths.keys():
            if type(paths[key]) == list:
                paths[key] = np.array(paths[key])

    full_data = {'images': images,
            'arkit_depths': arkit_depths,
            'confidences': confidences,
            'intrinsics': intrinsics,
            'poses': poses,
            'time_stamps': time_stamps,
            'paths': paths,
            'frame_ids': frame_ids,
            'metadata': metadata
    }

    data = {}
    for k in full_data.keys():
        if k in data_keys:
            data[k] = full_data[k]

    return data

def get_flip_matrix(x=False, y=False, z=False):
    x_val = -1 if x else 1
    y_val = -1 if y else 1
    z_val = -1 if z else 1

    # Flip the x and z axes by multiplying corresponding rows by -1
    flip_matrix = np.array([
        [x_val,  0,  0,  0],  # Flip x-axis
        [ 0,  y_val,  0,  0],  # Flip y-axis
        [ 0,  0, z_val,  0],  # Flip z-axis
        [ 0,  0,  0,  1]   # Keep homogeneous coordinate
    ], dtype=np.float64)

    return flip_matrix

def build_cylindrical_map_new(images, poses, intrinsics, all_mkpts, cyl_size=(512, 1024)):
    """
    Build a cylindrical map from multiple images and their poses.
    The panorama is uniformly scaled to fit inside the canvas without stretching.

    Inputs:
    - images: list of (H, W, 3) images
    - poses: list of (4, 4) camera extrinsics
    - intrinsics: list of (3, 3) camera intrinsic matrices
    - cyl_size: (height, width) of cylindrical map

    Output:
    - cylindrical_img (H, W, 3): final combined panorama
    - weight (H, W): number of images contributing to each pixel
    - individual_cylindrical_images: list of (H, W, 3) cylindrical warped images
    """

    cyl_h, cyl_w = cyl_size

    # First pass: find global min/max theta and h_cyl
    theta_all = []
    h_cyl_all = []

    for img, pose, K in zip(images, poses, intrinsics):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        h, w = img.shape[:2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = np.ones_like(x)

        dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        flip_matrix = get_flip_matrix(y=True, z=True)
        world_transform = pose @ flip_matrix

        pts_camera_homo = np.concatenate([dirs, np.ones((dirs.shape[0], 1))], axis=-1)
        pts_world_homo = pts_camera_homo @ world_transform.T
        pts_world = pts_world_homo[:, :3]

        theta = np.arctan2(pts_world[:, 0], pts_world[:, 2])
        h_cyl = pts_world[:, 1] / np.sqrt(pts_world[:, 0]**2 + pts_world[:, 2]**2)

        theta_all.append(theta)
        h_cyl_all.append(h_cyl)

    theta_all = np.concatenate(theta_all)
    h_cyl_all = np.concatenate(h_cyl_all)

    theta_min, theta_max = np.min(theta_all), np.max(theta_all)
    h_cyl_min, h_cyl_max = np.min(h_cyl_all), np.max(h_cyl_all)

    print(f'Theta range: {np.degrees(theta_min):.1f}° to {np.degrees(theta_max):.1f}°')
    print(f'Height range: {h_cyl_min:.2f} to {h_cyl_max:.2f}')

    theta_range = theta_max - theta_min
    h_cyl_range = h_cyl_max - h_cyl_min

    scale_w = cyl_w / theta_range
    scale_h = cyl_h / h_cyl_range
    scale = min(scale_w, scale_h)

    used_w = theta_range * scale
    used_h = h_cyl_range * scale

    pad_w = (cyl_w - used_w) / 2
    pad_h = (cyl_h - used_h) / 2

    # Second pass: map using uniform scaling and centering
    cylindrical_img = np.zeros((cyl_h, cyl_w, 3), dtype=np.float32)
    weight_flat = np.zeros((cyl_h * cyl_w), dtype=np.float32)
    individual_cylindrical_images = []

    all_mkpts_transformed = all_mkpts.copy()
    # This is saying for img0, its feature points are in the idx 1 in the last pair, and idx 0 in the first pair of mkpts
    # Another example, for img1, its feature points are in idx 1 in the first pair, and idx 0 in the second pair of mkpts
    mkpt_indices = [[[len(all_mkpts) - 1, 1], [0, 0]]]
    mkpt_indices += [[[i-1, 1], [i, 0]] for i in range(1, len(all_mkpts))]
    
    for img, pose, K, indices in zip(images, poses, intrinsics, mkpt_indices):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        h, w = img.shape[:2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = np.ones_like(x)

        dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        flip_matrix = get_flip_matrix(y=True, z=True)
        world_transform = pose @ flip_matrix

        pts_camera_homo = np.concatenate([dirs, np.ones((dirs.shape[0], 1))], axis=-1)
        pts_world_homo = pts_camera_homo @ world_transform.T
        pts_world = pts_world_homo[:, :3]

        theta = np.arctan2(pts_world[:, 0], pts_world[:, 2])
        h_cyl = pts_world[:, 1] / np.sqrt(pts_world[:, 0]**2 + pts_world[:, 2]**2)

        # Uniform mapping
        u_cyl = (theta - theta_min) * scale + pad_w
        v_cyl = (h_cyl - h_cyl_min) * scale + pad_h
        v_cyl = cyl_h - v_cyl  # flip vertically (optional depending on image convention)

        valid = (u_cyl >= 0) & (u_cyl < cyl_w) & (v_cyl >= 0) & (v_cyl < cyl_h)
        u_cyl = u_cyl[valid]
        v_cyl = v_cyl[valid]
        colors = img.reshape(-1, 3)[valid]

        u_cyl = np.clip(u_cyl, 0, cyl_w - 1)
        v_cyl = np.clip(v_cyl, 0, cyl_h - 1)

        x_pix = np.round(u_cyl).astype(np.int32)
        y_pix = np.round(v_cyl).astype(np.int32)
        flat_indices = y_pix * cyl_w + x_pix

        cylindrical_img_flat = cylindrical_img.reshape(-1, 3)

        # Create blank per-image cylindrical result
        single_cyl_img_flat = np.zeros_like(cylindrical_img_flat)
        single_weight_flat = np.zeros_like(weight_flat)

        # Accumulate
        np.add.at(cylindrical_img_flat, flat_indices, colors)
        np.add.at(weight_flat, flat_indices, 1)

        np.add.at(single_cyl_img_flat, flat_indices, colors)
        np.add.at(single_weight_flat, flat_indices, 1)

        # Reshape individual image
        single_cyl_img = single_cyl_img_flat.reshape(cyl_h, cyl_w, 3)
        single_weight = single_weight_flat.reshape(cyl_h, cyl_w)
        single_weight = np.clip(single_weight, 1e-5, None)
        single_cyl_img /= single_weight[..., None]
        single_cyl_img = np.clip(single_cyl_img, 0, 255).astype(np.uint8)

        individual_cylindrical_images.append(single_cyl_img)

        # Transform mkpts
        for i, j in indices:
            pts = all_mkpts[i][j] # ith pair, jth index

            # Step 1: Normalize to camera coordinates
            x = (pts[:, 0] - cx) / fx
            y = (pts[:, 1] - cy) / fy
            z = np.ones_like(x)

            dirs = np.stack([x, y, z], axis=-1)  # [n, 3]

            # Step 2: Camera to world transform
            flip_matrix = get_flip_matrix(y=True, z=True)
            world_transform = pose @ flip_matrix

            pts_camera_homo = np.concatenate([dirs, np.ones((dirs.shape[0], 1))], axis=-1)  # [n, 4]
            pts_world_homo = pts_camera_homo @ world_transform.T  # [n, 4]
            pts_world = pts_world_homo[:, :3]

            # Step 3: Cylindrical projection
            theta = np.arctan2(pts_world[:, 0], pts_world[:, 2])
            h_cyl = pts_world[:, 1] / np.sqrt(pts_world[:, 0]**2 + pts_world[:, 2]**2)

            # Step 4: Uniform mapping to cylindrical image coords
            u_cyl = (theta - theta_min) * scale + pad_w
            v_cyl = (h_cyl - h_cyl_min) * scale + pad_h
            v_cyl = cyl_h - v_cyl  # vertical flip (optional)

            # Step 5: Clip and round to pixel indices
            u_cyl = np.clip(u_cyl, 0, cyl_w - 1)
            v_cyl = np.clip(v_cyl, 0, cyl_h - 1)

            x_pix = np.round(u_cyl).astype(np.int32)
            y_pix = np.round(v_cyl).astype(np.int32)
            flat_indices = y_pix * cyl_w + x_pix  # optional if needed for indexing

            # Final transformed keypoints:
            all_mkpts_transformed[i][j] = np.stack([u_cyl, v_cyl], axis=-1)  # [n, 2]

    # Normalize final panorama
    weight = weight_flat.reshape(cyl_h, cyl_w)
    weight = np.clip(weight, 1e-5, None)
    cylindrical_img /= weight[..., None]
    cylindrical_img = np.clip(cylindrical_img, 0, 255).astype(np.uint8)

    return cylindrical_img, weight, individual_cylindrical_images, all_mkpts_transformed


def estimate_translation_ransac(pts1, pts2, iterations=5000, threshold=10.0, translation_penalty=0.2):
    """
    Estimate best translation using RANSAC, preferring smaller dx, dy.

    Args:
        pts1, pts2: matched points
        threshold: inlier distance threshold
        translation_penalty: penalize large translations (higher = stronger penalty)

    Returns:
        dx, dy
    """

    if pts1 is None or pts2 is None or len(pts1) < 4:
        return 0, 0

    best_score = -np.inf
    best_dxdy = np.array([0, 0])

    for _ in range(iterations):
        idx = np.random.choice(len(pts1), 2, replace=False)
        est = np.median(pts2[idx] - pts1[idx], axis=0)

        residuals = np.linalg.norm((pts2 - pts1) - est, axis=1)

        inliers = np.where(residuals < threshold)[0]
        num_inliers = len(inliers)

        # Compute score: number of inliers penalized by translation magnitude
        translation_magnitude = np.linalg.norm(est)
        score = num_inliers - translation_penalty * translation_magnitude

        if score > best_score:
            best_score = score
            best_dxdy = est

    dx, dy = best_dxdy
    return dx, dy


def compute_corrected_offsets(cylindrical_images, all_mkpts, vis=False):
    """
    Compute corrected global offsets for stitching.
    The first image stays still, we compute offsets according anchored to the first image
    But for the last offset, we calculate how much the first image should move in order to match with the last image
    So len(offsets) = len(cylindircal_images) + 1
    """

    offsets = [(0, 0)]  # Start with first image at (0,0)
    masks = []
    mask = cv2.cvtColor(cylindrical_images[0], cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255  # binary mask
    masks.append(mask)

    for i in range(len(cylindrical_images)):
        idx1 = i
        idx2 = (i + 1) % len(cylindrical_images)
        img1 = cylindrical_images[idx1]
        img2 = cylindrical_images[idx2]
        pts2, pts1 = all_mkpts[i]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        last_offset = offsets[-1]
        translation_matrix = np.float32([[1, 0, last_offset[0]], [0, 1, last_offset[1]]])
        warped_img1 = cv2.warpAffine(img1, translation_matrix, (w1, h1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_img2 = cv2.warpAffine(img2, translation_matrix, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # TODO: This is not giving the correct offset,
        # Apply the affine transform to the key points
        # pts_homo = np.concatenate([pts1, np.ones((pts1.shape[0], 1), dtype=np.float32)], axis=1)  # [n, 3]
        # pts1 = (translation_matrix @ pts_homo.T).T  # [n, 2]
        # pts_homo = np.concatenate([pts2, np.ones((pts1.shape[0], 1), dtype=np.float32)], axis=1)  # [n, 3]
        # pts2 = (translation_matrix @ pts_homo.T).T  # [n, 2]

        if pts1 is None or pts2 is None:
            dx = 0
            dy = 0
        else:
            # dx, dy = estimate_translation_ransac(pts1, pts2)
            # dx = -dx
            # dy = -dy

            # Check if the image is at the boarder
            x_range = list(np.where(img1 != 0))[1]
            if max(x_range) - min(x_range) == w1 - 1:
                # Shift the points towards the center of the image. so we can calculate the offset easier
                pts1[:, 0] += int(w1/2)
                pts1[:, 0] %= w1
                pts2[:, 0] += int(w1/2)
                pts2[:, 0] %= w1

            dx = pts1[:, 0] - pts2[:, 0]
            dx = -int(np.median(dx))
            dy = pts1[:, 1] - pts2[:, 1]
            dy = -int(np.median(dy))

        # visualize_feature_matches(warped_img1, warped_img2, pts1, pts2, dx=dx, dy=dy, threshold=10.0)
        new_offset = (last_offset[0] + dx, last_offset[1] + dy)
        offsets.append(new_offset)

        print(f"Image {idx2}: Estimated relative shift dx={dx:.2f}, dy={dy:.2f}")

        # === Visualization code ===
        # Warp img2 by estimated offset
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        final_warped_img2 = cv2.warpAffine(warped_img2, translation_matrix, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if vis: 
            # Create side-by-side comparison
            fig, axs = plt.subplots(2, 2, figsize=(18, 6))

            axs[0][0].imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
            axs[0][0].set_title(f'Image {idx1}')
            axs[0][0].axis('off')

            axs[0][1].imshow(cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB))
            axs[0][1].set_title(f'Image {idx2}')
            axs[0][1].axis('off')

            alpha = 0.5
            blended = cv2.addWeighted(warped_img1, alpha, warped_img2, 1 - alpha, 0)
            axs[1][0].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            axs[1][0].set_title(f'Blended before applying offset')
            axs[1][0].axis('off')

            # Blend img1 and warped img2
            alpha = 0.5
            blended = cv2.addWeighted(warped_img1, alpha, final_warped_img2, 1 - alpha, 0)
            axs[1][1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            axs[1][1].set_title(f'Blended after applying offset')
            axs[1][1].axis('off')

            plt.suptitle(f"Matching Image {idx1} → {idx2} | Estimated dx={dx:.1f}, dy={dy:.1f}")
            plt.tight_layout()
            plt.show()
            
    return np.array(offsets)


from types import SimpleNamespace
def get_superglue_opt(
    output_dir='./results',
    max_length=-1,
    resize=[-1],
    resize_float=True,
    superglue='outdoor',
    max_keypoints=2048,
    keypoint_threshold=0.05,
    nms_radius=5,
    sinkhorn_iterations=20,
    match_threshold=0.9,
    viz=True,
    eval=False,
    fast_viz=False,
    cache=False,
    show_keypoints=False,
    viz_extension='png',
    opencv_display=False,
    shuffle=False,
    force_cpu=False,
):
    # Construct argument namespace
    opt = SimpleNamespace(
        output_dir=output_dir,
        max_length=max_length,
        resize=resize,
        resize_float=resize_float,
        superglue=superglue,
        max_keypoints=max_keypoints,
        keypoint_threshold=keypoint_threshold,
        nms_radius=nms_radius,
        sinkhorn_iterations=sinkhorn_iterations,
        match_threshold=match_threshold,
        viz=viz,
        eval=eval,
        fast_viz=fast_viz,
        cache=cache,
        show_keypoints=show_keypoints,
        viz_extension=viz_extension,
        opencv_display=opencv_display,
        shuffle=shuffle,
        force_cpu=force_cpu
    )
    return opt


def extract_and_match_features(images, image_names):
    opt = get_superglue_opt()
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    
    pairs = [[i, i+1] for i in range(len(image_names) - 1)]
    pairs.append([len(image_names) - 1, 0])

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)

    all_mkpts = []
    for i, pair in enumerate(pairs):
        idx0, idx1 = pair[:2]
        name0, name1 = image_names[idx0], image_names[idx1]
        image0, image1 = images[idx0], images[idx1]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # Load matches if exists
        if os.path.exists(matches_path):
            npz = np.load(matches_path)
            matches = npz['matches']
            keypoints0 = npz['keypoints0']
            keypoints1 = npz['keypoints1']
            conf = npz['match_confidence']

            # Filter matched points
            matched_idx = matches > -1
            mkpts0 = keypoints0[matched_idx]
            mkpts1 = keypoints1[matches[matched_idx]]
            all_mkpts.append([mkpts0, mkpts1])
            continue

        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                do_eval = False
            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

        inp0 = frame2tensor(image0.astype('float32'), device)
        inp1 = frame2tensor(image1.astype('float32'), device)
        scales0, scales1 = (1, 1), (1, 1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        all_mkpts.append([mkpts0, mkpts1])

        if do_eval:
            # Estimate the pose and compute the pose error.
            assert len(pair) == 38, 'Pair does not have ground truth info'
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale the intrinsics to resized image.
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # Update the intrinsics + extrinsics if EXIF rotation was found.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0

            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            thresh = 1.  # In pixels relative to resized image size.
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        if do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        if do_viz_eval:
            # Visualize the evaluation results for the image pair.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info (only works with --fast_viz).
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)

            timer.update('viz_eval')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    if opt.eval:
        # Collate the results into a final table and print to terminal.
        pose_errors = []
        precisions = []
        matching_scores = []
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.*yy for yy in aucs]
        prec = 100.*np.mean(precisions)
        ms = 100.*np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))

    return all_mkpts


def apply_offsets_to_cylindrical_images(images, offsets):
    """
    Apply [dx, dy] offsets to a list of cylindrical images.
    Horizontal (x) shifts wrap around, vertical (y) shifts crop or pad with black.

    Args:
        images (List[np.ndarray]): List of HxWx3 images (uint8 or float).
        offsets (List[Tuple[int, int]]): List of (dx, dy) offsets for each image, the last offset indicates how much we should shift the first image such that it aligns with the last image and complte the loop

    Returns:
        List[np.ndarray]: List of offset-applied images, same size as input.
    """
    shifted_images = []

    for img, (dx, dy) in zip(images, offsets[:-1]):
        h, w = img.shape[:2]

        # Wrap horizontally (x)
        dx = dx % w  # normalize to [0, w)
        img_shifted_x = np.roll(img, shift=dx, axis=1)

        # Crop/pad vertically (y)
        if dy >= 0:
            # Crop from top, pad black at bottom
            img_shifted_y = np.zeros_like(img)
            cropped = img_shifted_x[:h - dy] if dy < h else np.zeros((0, w, 3), dtype=img.dtype)
            img_shifted_y[dy:dy + cropped.shape[0]] = cropped
        else:
            # Crop from bottom, pad black at top
            img_shifted_y = np.zeros_like(img)
            dy_abs = abs(dy)
            cropped = img_shifted_x[dy_abs:] if dy_abs < h else np.zeros((0, w, 3), dtype=img.dtype)
            img_shifted_y[:cropped.shape[0]] = cropped

        shifted_images.append(img_shifted_y)

    # Use the last offset to resize the image and move the part of the first image
    last_offset_x = offsets[-1][0]
    assert last_offset_x < 0
    first_image = shifted_images[0]
    empty_h_slice_idx = 0
    for i in range(first_image.shape[1]):
        h_slice = first_image[:, i:i-last_offset_x, :]
        if np.max(h_slice) == 0:
            empty_h_slice_idx = i
            break
    left = first_image[:, :empty_h_slice_idx, :]
    right = first_image[:, empty_h_slice_idx-last_offset_x:, :]
    first_image = np.concatenate([left, right], axis=1)
    
    shifted_images[0] = first_image
    for i in range(1, len(shifted_images)):
        shifted_images[i] = shifted_images[i][:, :last_offset_x, :]

    return shifted_images

def blend_cylindrical_images_alpha(images):
    """
    Alpha blends a list of cylindrical images by averaging overlapping pixels.

    Args:
        images (List[np.ndarray]): List of HxWx3 images (uint8 or float).

    Returns:
        np.ndarray: Blended image of shape HxWx3, same dtype as input.
    """
    if not images:
        raise ValueError("Input image list is empty.")

    h, w, c = images[0].shape
    dtype = images[0].dtype

    # Accumulation buffers
    accum = np.zeros((h, w, c), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for img in images:
        valid_mask = np.any(img > 0, axis=-1)  # assumes black (0) is padding
        img_float = img.astype(np.float32)
        
        accum[valid_mask] += img_float[valid_mask]
        weight[valid_mask] += 1.0

    # Avoid division by zero
    weight = np.clip(weight, 1e-5, None)
    blended = accum / weight[..., None]

    # Convert back to original dtype
    if np.issubdtype(dtype, np.integer):
        blended = np.clip(blended, 0, 255).astype(dtype)

    return blended



def compensate_exposure(warped_images, masks, corners):
    # compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    block_size = 128  # or 128 if images are really big
    compensator = cv2.detail_BlocksGainCompensator(block_size, block_size)
    compensator.feed(corners=corners, images=warped_images, masks=masks)

    for i in range(len(warped_images)):
        compensator.apply(i, corners[i], warped_images[i], masks[i])

def find_seams(warped_images, masks, corners):
    images_f32 = [img.astype(np.float32) for img in warped_images]
        # seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM) # Fastest but not good
    # seam_finder = cv2.detail_DpSeamFinder('COLOR')
    # seam_finder = cv2.detail_DpSeamFinder('COLOR_GRAD')
    seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD') # Best, but takes forever
    # seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR')
    refined_masks = seam_finder.find(images_f32, corners, masks)
    return refined_masks

def multiband_blend(warped_images, masks, corners, blend_strength=5):
    sizes = [(img.shape[1], img.shape[0]) for img in warped_images]
    dst_roi = cv2.detail.resultRoi(corners, sizes)

    blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * blend_strength / 100.
    # num_bands = min(5, max(1, int(np.log2(blend_width)) - 1))
    num_bands = (np.log(blend_width) / np.log(2.) - 1.).astype(np.int32)

    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(num_bands)
    blender.prepare(dst_roi)

    for img, mask, corner in zip(warped_images, masks, corners):
        blender.feed(cv2.UMat(img.astype(np.int16)), cv2.UMat(mask), corner)

    result, _ = blender.blend(None, None)
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_images_with_seams(images, fix_exposure=False, visualize=False):
    """
    Blend a list of cylindrical images using exposure compensation, seam finding, and multi-band blending.

    Args:
        images (List[np.ndarray]): List of HxWx3 cylindrical images (already on panorama canvas).
        visualize (bool): If True, show seam masks and final result.

    Returns:
        np.ndarray: Final blended image.
    """

    num_images = len(images)
    if num_images == 0:
        raise ValueError("Image list is empty.")

    # Step 1: Compute binary masks for each image
    masks = [(np.any(img != 0, axis=2).astype(np.uint8)) * 255 for img in images]

    # Step 2: All images are already aligned; corners are (0, 0)
    corners = [(0, 0)] * num_images

    # Step 3: Exposure Compensation
    if fix_exposure:
        compensate_exposure(images, masks, corners)

    # Step 4: Seam Finding (on downsampled versions)
    seam_megapix = 0.1
    is_seam_scale_set = False
    warped_images_seam = []
    masks_seam = []

    for img, mask in zip(images, masks):
        if not is_seam_scale_set:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (img.shape[0] * img.shape[1])))
            is_seam_scale_set = True

        img_seam = cv2.resize(img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        mask_seam = cv2.resize(mask, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)

        warped_images_seam.append(img_seam)
        masks_seam.append(mask_seam)

    # Step 5: Compute seam masks
    masks_small = find_seams(warped_images_seam, masks_seam, corners)

    # Step 6: Upscale and refine seam masks
    refined_masks = []
    h, w = images[0].shape[:2]
    for mask in masks_small:
        mask_up = cv2.resize(mask.get(), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.GaussianBlur(mask_up, (5, 5), sigmaX=2, sigmaY=2)
        _, mask_bin = cv2.threshold(mask_up, 128, 255, cv2.THRESH_BINARY)
        refined_masks.append(mask_bin)

    refined_masks = np.array(refined_masks)

    if visualize:
        print("Seam Masks")
        for mask in refined_masks:
            plt.imshow(mask, cmap='gray')
            plt.show()

    # Step 7: Multi-band blending
    blended = multiband_blend(images, refined_masks, corners)

    if visualize:
        print("Blended Panorama")
        plt.imshow(blended)
        plt.axis('off')
        plt.show()

    return blended


def main():
    obs_path = '/Users/cyqpp/Work/Research/Plane_Based_Localization/Dataset/PALMS+/BE/Session_1744229360'
    data = load_full_obs(obs_path, ['images', 'intrinsics', 'poses', 'paths'])
    images, intrinsics, poses = data['images'], data['intrinsics'], data['poses']
    image_paths = data['paths']['image_paths']
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    
    # Rotate images, intrinsics and poses for 90 degrees
    R_rotate = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    for i, pose in enumerate(poses):
        new_pose = pose.copy()
        new_pose[:3, :3] = pose[:3, :3] @ R_rotate
        poses[i] = new_pose

    original_img_size = images[0].shape[:2]
    for i in range(len(intrinsics)):
        intrinsics[i] = rotate_intrinsics_custom(intrinsics[i], original_img_size)

    new_images = []
    for i in range(len(images)):
        new_images.append(cv2.rotate(images[i], cv2.ROTATE_90_CLOCKWISE))
    images = np.array(new_images)

    # Run SuperGlue
    all_mkpts = extract_and_match_features(images, image_names)

    cylindrical_img, weight, individual_cylindrical_images, all_mkpts = build_cylindrical_map_new(images, poses, intrinsics, all_mkpts)

    offsets = compute_corrected_offsets(individual_cylindrical_images, all_mkpts)
    assert len(offsets) == len(individual_cylindrical_images) + 1, f'Error, got {len(offsets)} and {len(individual_cylindrical_images)}'

    individual_cylindrical_images = apply_offsets_to_cylindrical_images(individual_cylindrical_images, offsets)

    # combined_image = blend_cylindrical_images_alpha(individual_cylindrical_images)
    print("Blending images")
    combined_image = blend_images_with_seams(individual_cylindrical_images)
    plt.imshow(combined_image)
    plt.show()


if __name__ == '__main__':
    main()