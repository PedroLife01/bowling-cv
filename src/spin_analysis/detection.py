"""
Spin Detection - Core rotation estimation from optical flow.

Computes per-frame rotation of the bowling ball using:
1. ROI extraction around detected ball position
2. Lucas-Kanade optical flow with bidirectional (forward-backward) validation
3. 3D sphere projection of tracked surface points
4. Kabsch algorithm (SVD) for optimal rotation matrix estimation
5. Axis-angle decomposition of the rotation matrix

Input:  original video + ball trajectory CSV (frame, x, y, radius)
Output: rotation_data.csv (frame, x, y, radius, x_axis, y_axis, z_axis, angle)

Version: 1.0.0
Created: March 30, 2026
"""

import math
import os

import cv2
import numpy as np
import pandas as pd

from . import config as default_config


# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================


def roi_bounds(x, y, radius, margin, frame_shape):
    """
    Compute ROI bounding box around ball position.

    Parameters
    ----------
    x, y : int
        Ball center coordinates.
    radius : int
        Ball radius in pixels.
    margin : int
        Extra pixels beyond the radius.
    frame_shape : tuple
        (height, width, ...) of the video frame.

    Returns
    -------
    tuple : (x_min, x_max, y_min, y_max) clipped to frame bounds.
    """
    x_min = max(int(x) - radius - margin, 0)
    x_max = min(int(x) + radius + margin, frame_shape[1])
    y_min = max(int(y) - radius - margin, 0)
    y_max = min(int(y) + radius + margin, frame_shape[0])
    return x_min, x_max, y_min, y_max


def compute_optical_flow(prev_gray, curr_gray, roi_mask, lk_params, feature_params):
    """
    Compute Lucas-Kanade optical flow with bidirectional (forward-backward) check.

    Parameters
    ----------
    prev_gray : ndarray
        Grayscale ROI from the previous frame.
    curr_gray : ndarray
        Grayscale ROI from the current frame.
    roi_mask : ndarray
        Binary mask for feature detection (circular region inside ball).
    lk_params : dict
        Parameters for cv2.calcOpticalFlowPyrLK.
    feature_params : dict
        Parameters for cv2.goodFeaturesToTrack.

    Returns
    -------
    tuple : (p0, p1, p0r, status_forward, status_backward, fb_error)
        p0: detected features in prev_gray
        p1: tracked positions in curr_gray (forward pass)
        p0r: back-tracked positions in prev_gray (backward pass)
        status_forward/backward: tracking status arrays
        fb_error: forward-backward consistency error per point

    Raises
    ------
    ValueError
        If no features are detected in the ROI.
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=roi_mask, **feature_params)
    if p0 is None:
        raise ValueError("No features detected in ball ROI.")

    # Forward pass: prev -> curr
    p1, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None, **lk_params
    )
    # Backward pass: curr -> prev (for consistency check)
    p0r, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, p1, None, **lk_params
    )

    # Forward-backward error: distance between original and back-tracked points
    fb_error = np.linalg.norm(p0 - p0r, axis=2)

    return p0, p1, p0r, status_forward, status_backward, fb_error


def project_to_sphere(points_2d, center, radius):
    """
    Project 2D surface points onto a 3D sphere.

    Assumes the ball is a sphere and the camera is orthographic.
    For a point (px, py) relative to the ball center:
        z = sqrt(r^2 - px^2 - py^2)

    Parameters
    ----------
    points_2d : ndarray, shape (N, 2)
        2D point coordinates in ROI space.
    center : ndarray, shape (2,)
        Ball center in ROI coordinates.
    radius : float
        Ball radius in pixels.

    Returns
    -------
    ndarray, shape (N, 3)
        3D points on the sphere surface.
    """
    relative = points_2d - center
    px, py = relative[:, 0], relative[:, 1]
    z_sq = radius**2 - px**2 - py**2
    z = np.sqrt(np.maximum(z_sq, 0))
    return np.column_stack([px, py, z])


def compute_rotation(pts_3d_prev, pts_3d_curr):
    """
    Compute optimal rotation matrix using the Kabsch algorithm via SVD.

    Given two sets of corresponding 3D points P and Q:
        H = P^T @ Q
        U, S, Vt = SVD(H)
        R = Vt^T @ U^T

    Handles reflection: if det(R) < 0, flip the last row of Vt.

    Extracts axis-angle representation:
        angle = arccos((trace(R) - 1) / 2)
        axis  = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] / (2*sin(angle))

    Parameters
    ----------
    pts_3d_prev : ndarray, shape (N, 3)
        3D points from the previous frame.
    pts_3d_curr : ndarray, shape (N, 3)
        Corresponding 3D points from the current frame.

    Returns
    -------
    tuple : (axis, angle)
        axis: unit rotation axis (3,)
        angle: rotation angle in radians
    """
    P, Q = pts_3d_prev, pts_3d_curr
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection (improper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Extract angle from trace
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Extract axis
    if np.isclose(theta, 0):
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis /= norm

    return axis, theta


def _filter_3d_points(
    p0, p1, status_forward, status_backward, fb_error,
    center_roi1, center_roi2, ball_radius, cfg
):
    """
    Filter tracked points and project to 3D sphere surface.

    Applies forward-backward consistency check and displacement filtering,
    then projects valid points onto the sphere.

    Returns
    -------
    tuple : (old3d, new3d) arrays of shape (M, 3) each.
    """
    fb_threshold = cfg.FB_ERROR_THRESHOLD
    movement_threshold = ball_radius
    low_movement_threshold = ball_radius / cfg.LOW_MOVEMENT_FACTOR

    old3d, new3d = [], []
    for old_pt, new_pt, s1, s2, err in zip(
        p0.reshape(-1, 2),
        p1.reshape(-1, 2),
        status_forward.ravel(),
        status_backward.ravel(),
        fb_error.ravel(),
    ):
        if s1 and s2 and err < fb_threshold:
            displacement = np.linalg.norm(new_pt - old_pt)
            if low_movement_threshold < displacement < movement_threshold:
                ox, oy = old_pt - center_roi1
                nx, ny = new_pt - center_roi2
                oz = math.sqrt(max(ball_radius**2 - ox**2 - oy**2, 0))
                nz = math.sqrt(max(ball_radius**2 - nx**2 - ny**2, 0))
                old3d.append([ox, oy, oz])
                new3d.append([nx, ny, nz])

    return np.array(old3d), np.array(new3d)


def _filter_3d_points_lenient(
    p0, p1, status_forward, status_backward, fb_error,
    center_roi1, center_roi2, ball_radius, cfg
):
    """Filter with more lenient low-movement threshold (for retry)."""
    fb_threshold = cfg.FB_ERROR_THRESHOLD
    movement_threshold = ball_radius
    low_movement_threshold = ball_radius / cfg.LENIENT_LOW_MOVEMENT_FACTOR

    old3d, new3d = [], []
    for old_pt, new_pt, s1, s2, err in zip(
        p0.reshape(-1, 2),
        p1.reshape(-1, 2),
        status_forward.ravel(),
        status_backward.ravel(),
        fb_error.ravel(),
    ):
        if s1 and s2 and err < fb_threshold:
            displacement = np.linalg.norm(new_pt - old_pt)
            if low_movement_threshold < displacement < movement_threshold:
                ox, oy = old_pt - center_roi1
                nx, ny = new_pt - center_roi2
                oz = math.sqrt(max(ball_radius**2 - ox**2 - oy**2, 0))
                nz = math.sqrt(max(ball_radius**2 - nx**2 - ny**2, 0))
                old3d.append([ox, oy, oz])
                new3d.append([nx, ny, nz])

    return np.array(old3d), np.array(new3d)


def _fill_frames(video_path, csv_path):
    """
    Create a DataFrame with one row per video frame, filling in known ball positions.

    Frames without detections get NaN values for x, y, radius.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    df_coords = pd.read_csv(csv_path)

    full_df = pd.DataFrame({
        'frame': list(range(frame_count)),
        'x': [None] * frame_count,
        'y': [None] * frame_count,
        'radius': [None] * frame_count,
    })

    df_coords.set_index('frame', inplace=True)
    full_df.set_index('frame', inplace=True)
    full_df.update(df_coords)
    full_df.reset_index(inplace=True)

    return full_df


def _fill_frames_with_axis(video_path, csv_path):
    """
    Create a DataFrame with one row per video frame, filling in known rotation data.

    Frames without rotation data get NaN values for axis and angle columns.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    df_coords = pd.read_csv(csv_path)

    full_df = pd.DataFrame({
        'frame': list(range(frame_count)),
        'x': [None] * frame_count,
        'y': [None] * frame_count,
        'radius': [None] * frame_count,
        'x_axis': [None] * frame_count,
        'y_axis': [None] * frame_count,
        'z_axis': [None] * frame_count,
        'angle': [None] * frame_count,
    })

    df_coords.set_index('frame', inplace=True)
    full_df.set_index('frame', inplace=True)
    full_df.update(df_coords)
    full_df.reset_index(inplace=True)

    return full_df


# ==============================================================================
#                               MAIN PROCESSING FUNCTION
# ==============================================================================


def process_spin(video_path, ball_csv_path, config=None):
    """
    Main entry point for spin detection.

    Processes consecutive frame pairs to estimate per-frame ball rotation.

    Parameters
    ----------
    video_path : str
        Path to the original bowling video.
    ball_csv_path : str
        Path to CSV with ball trajectory (columns: frame, x, y, radius).
    config : module, optional
        Configuration module. Defaults to spin_analysis.config.

    Returns
    -------
    str
        Path to the output rotation_data.csv file.
    """
    if config is None:
        config = default_config

    cfg = config

    # Build output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(cfg.OUTPUT_DIR, video_name, 'spin_analysis')
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, 'rotation_data.csv')

    # Fill frames to get one row per video frame
    df = _fill_frames(video_path, ball_csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    margin = cfg.ROI_MARGIN

    # Optical flow parameters
    feature_params = dict(
        maxCorners=cfg.MAX_CORNERS,
        qualityLevel=cfg.QUALITY_LEVEL,
        minDistance=cfg.MIN_DISTANCE,
        blockSize=cfg.BLOCK_SIZE,
    )
    lk_params = dict(
        winSize=(cfg.WINDOW_SIZE, cfg.WINDOW_SIZE),
        maxLevel=cfg.MAX_PYRAMID_LEVEL,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            cfg.LK_MAX_ITER,
            cfg.LK_EPSILON,
        ),
    )

    output_data = []

    for frame_number in range(frame_count - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        if not (ret1 and ret2):
            continue

        # Check that both frames have valid ball data
        try:
            row1 = df.iloc[frame_number]
            row2 = df.iloc[frame_number + 1]
            if (
                pd.isna(row1['x']) or pd.isna(row1['y']) or pd.isna(row1['radius'])
                or pd.isna(row2['x']) or pd.isna(row2['y']) or pd.isna(row2['radius'])
            ):
                continue
        except IndexError:
            continue

        ball_center1 = np.array([int(row1['x']), int(row1['y'])])
        ball_center2 = np.array([int(row2['x']), int(row2['y'])])
        ball_radius = int(row1['radius'])

        # Extract ROIs
        x_min1, x_max1, y_min1, y_max1 = roi_bounds(
            ball_center1[0], ball_center1[1], ball_radius, margin, frame1.shape
        )
        x_min2, x_max2, y_min2, y_max2 = roi_bounds(
            ball_center2[0], ball_center2[1], ball_radius, margin, frame2.shape
        )

        roi1 = frame1[y_min1:y_max1, x_min1:x_max1]
        roi2 = frame2[y_min2:y_max2, x_min2:x_max2]

        if roi1.size == 0 or roi2.size == 0:
            continue

        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

        # Ball center in ROI coordinates
        center_roi1 = ball_center1 - np.array([x_min1, y_min1])
        center_roi2 = ball_center2 - np.array([x_min2, y_min2])

        # Circular mask for feature detection (slightly smaller than ball)
        mask = np.zeros_like(gray1)
        cv2.circle(
            mask,
            tuple(center_roi1.astype(int)),
            int(ball_radius * cfg.MASK_RADIUS_FACTOR),
            255, -1
        )

        # Compute optical flow
        try:
            p0, p1, p0r, s_f, s_b, fb_error = compute_optical_flow(
                gray1, gray2, mask, lk_params, feature_params
            )
        except ValueError:
            continue

        # Filter and project to 3D
        old3d, new3d = _filter_3d_points(
            p0, p1, s_f, s_b, fb_error,
            center_roi1, center_roi2, ball_radius, cfg
        )

        # Retry with lenient threshold if too few points
        if old3d.shape[0] < cfg.MIN_POINTS_FOR_ROTATION:
            old3d, new3d = _filter_3d_points_lenient(
                p0, p1, s_f, s_b, fb_error,
                center_roi1, center_roi2, ball_radius, cfg
            )

        if old3d.shape[0] < cfg.MIN_POINTS_FOR_ROTATION:
            continue

        # Kabsch rotation estimation
        axis, theta = compute_rotation(old3d, new3d)

        output_data.append({
            'frame': int(row1['frame']),
            'x': row1['x'],
            'y': row1['y'],
            'radius': row1['radius'],
            'x_axis': axis[0],
            'y_axis': axis[1],
            'z_axis': axis[2],
            'angle': theta,
        })

        if cfg.VERBOSE and frame_number % 30 == 0:
            print(f"  Frame {frame_number}: axis=({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}), "
                  f"angle={np.degrees(theta):.1f} deg")

    cap.release()

    # Save raw rotation data
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)

    # Fill missing frames with NaN rows
    output_df = _fill_frames_with_axis(video_path, output_csv_path)
    output_df.to_csv(output_csv_path, index=False)

    print(f"Saved rotation data to {output_csv_path}")
    return output_csv_path
