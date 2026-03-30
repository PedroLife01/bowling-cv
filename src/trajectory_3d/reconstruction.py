"""
Core trajectory reconstruction logic - Phase 3

Adapts the homography-based reconstruction from the worcester codebase
to work with bowling-cv's boundary_data.json and ball trajectory CSV formats.

The pipeline:
1. Read boundary_data.json -> extract 4 corner intersections
2. Build a per-frame lane corners CSV (static boundaries = same corners every frame)
3. Compute homography per frame mapping image corners to lane-space rectangle
4. Transform ball (x, y+radius) positions through the homography
5. Apply Savitzky-Golay smoothing to the transformed trajectory

Version: 1.0.0
Authors: Pedro Roriz
Created: March 30, 2026
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ==============================================================================
#                         BOUNDARY -> LANE CORNERS
# ==============================================================================


def convert_boundary_json_to_lane_csv(boundary_json_path, output_csv_path=None, num_frames=None):
    """
    Reads bowling-cv's boundary_data.json (which stores static lane boundaries)
    and produces a per-frame CSV with the 4 lane corner positions.

    The boundary_data.json already contains pre-computed intersection points:
        intersections.top_left     {x, y}
        intersections.top_right    {x, y}
        intersections.bottom_left  {x, y}
        intersections.bottom_right {x, y}

    Since boundaries are static (detected once from the whole video), every
    frame gets the same 4 corners.

    Parameters
    ----------
    boundary_json_path : str
        Path to boundary_data.json from Phase 1.
    output_csv_path : str, optional
        Where to write the CSV. If None, the CSV is not saved to disk.
    num_frames : int, optional
        Number of frames to generate rows for. If None, generates a single row
        with Frame=0 (sufficient when homography is constant).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Frame, tl_x, tl_y, tr_x, tr_y,
                                 bl_x, bl_y, br_x, br_y
    """
    with open(boundary_json_path, 'r') as f:
        boundary_data = json.load(f)

    intersections = boundary_data['intersections']

    # Extract the 4 corners from pre-computed intersections
    tl = intersections['top_left']
    tr = intersections['top_right']
    bl = intersections['bottom_left']
    br = intersections['bottom_right']

    corners = {
        'tl_x': tl['x'], 'tl_y': tl['y'],
        'tr_x': tr['x'], 'tr_y': tr['y'],
        'bl_x': bl['x'], 'bl_y': bl['y'],
        'br_x': br['x'], 'br_y': br['y'],
    }

    # Build DataFrame - one row per frame (or single row if num_frames is None)
    if num_frames is not None and num_frames > 0:
        rows = []
        for frame_idx in range(num_frames):
            row = {'Frame': frame_idx}
            row.update(corners)
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame([{'Frame': 0, **corners}])

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"  Lane corners CSV saved to {output_csv_path}")

    return df


# ==============================================================================
#                         HOMOGRAPHY COMPUTATION
# ==============================================================================


def compute_homographies_per_frame(lane_csv_path, width, height):
    """
    Reads the lane corners CSV and computes a homography matrix for each frame.

    The homography maps the 4 image-space lane corners to a destination
    rectangle of size (width x height) representing the lane in real-world
    coordinates (inches or cm).

    Parameters
    ----------
    lane_csv_path : str or pd.DataFrame
        Path to the lane corners CSV, or a DataFrame directly.
    width : int
        Destination width (e.g. LANE_WIDTH = 106).
    height : int
        Destination height (e.g. LANE_LENGTH = 1829).

    Returns
    -------
    dict
        {frame_number: 3x3 homography matrix}
    """
    if isinstance(lane_csv_path, pd.DataFrame):
        df = lane_csv_path
    else:
        df = pd.read_csv(lane_csv_path)

    homographies = {}

    for _, row in df.iterrows():
        frame = int(row['Frame'])

        # Source points: 4 lane corners in image space
        # Order: bottom-left, bottom-right, top-right, top-left
        pts_src = np.array([
            [row['bl_x'], row['bl_y']],
            [row['br_x'], row['br_y']],
            [row['tr_x'], row['tr_y']],
            [row['tl_x'], row['tl_y']],
        ], dtype=np.float32)

        # Destination points: rectangle in lane coordinates
        # Bottom of lane (foul line) = y=height, Top (pins) = y=0
        pts_dst = np.array([
            [0, height],
            [width, height],
            [width, 0],
            [0, 0],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(pts_src, pts_dst)
        if H is not None:
            homographies[frame] = H

    return homographies


# ==============================================================================
#                    BALL POSITION TRANSFORMATION
# ==============================================================================


def apply_homography_per_frame(ball_csv_path, homographies, output_csv_path=None):
    """
    Transforms ball positions from image space to lane coordinates using
    per-frame homography matrices.

    Uses the bottom of the ball (x, y + radius) as the contact point,
    matching the worcester reference implementation.

    Parameters
    ----------
    ball_csv_path : str
        Path to ball trajectory CSV with columns: frame, x, y, radius.
    homographies : dict
        {frame_number: 3x3 homography matrix} from compute_homographies_per_frame.
    output_csv_path : str, optional
        Where to save the transformed CSV. If None, not saved to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: frame, x, y (in lane coordinates).
    """
    df = pd.read_csv(ball_csv_path)
    transformed_data = []

    for _, row in df.iterrows():
        frame_id = int(row['frame'])
        x, y = row['x'], row['y']

        # Handle missing values
        if pd.isna(x) or pd.isna(y):
            transformed_data.append([frame_id, np.nan, np.nan])
            continue

        # Use radius if available for bottom-of-ball contact point
        r = row.get('radius', 0)
        if pd.isna(r):
            r = 0

        # Since boundaries are static, all frames share the same homography
        # Try the exact frame first, then fall back to frame 0
        H = homographies.get(frame_id)
        if H is None:
            H = homographies.get(0)
        if H is None:
            # Use any available homography (they're all the same for static boundaries)
            if homographies:
                H = next(iter(homographies.values()))
            else:
                continue

        # Transform bottom-of-ball point
        point = np.array([[[x, y + r]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, H)[0][0]
        transformed_data.append([frame_id, transformed_point[0], transformed_point[1]])

    transformed_df = pd.DataFrame(transformed_data, columns=['frame', 'x', 'y'])

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        transformed_df.to_csv(output_csv_path, index=False)
        print(f"  Transformed positions saved to {output_csv_path}")

    return transformed_df


# ==============================================================================
#                         TRAJECTORY SMOOTHING
# ==============================================================================


def smooth_trajectory(df, window_length=61, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth the transformed trajectory.

    Only smooths rows with valid (non-NaN) x and y values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: frame, x, y.
    window_length : int
        Window length for savgol_filter (must be odd).
    polyorder : int
        Polynomial order for savgol_filter.

    Returns
    -------
    pd.DataFrame
        Smoothed DataFrame with same columns.
    """
    df = df.copy()

    # Only smooth valid (non-NaN) rows
    valid_mask = df['x'].notna() & df['y'].notna()
    valid_count = valid_mask.sum()

    if valid_count < window_length:
        # Not enough points to smooth - adjust window or skip
        if valid_count >= 5:
            # Use a smaller window
            adjusted_window = valid_count if valid_count % 2 == 1 else valid_count - 1
            adjusted_window = max(adjusted_window, polyorder + 2)
            if adjusted_window % 2 == 0:
                adjusted_window -= 1
            print(f"  Warning: Only {valid_count} valid points. "
                  f"Adjusted smoothing window from {window_length} to {adjusted_window}.")
            window_length = adjusted_window
        else:
            print(f"  Warning: Only {valid_count} valid points. Skipping smoothing.")
            return df

    df.loc[valid_mask, 'x'] = savgol_filter(
        df.loc[valid_mask, 'x'].values,
        window_length=window_length,
        polyorder=polyorder
    )
    df.loc[valid_mask, 'y'] = savgol_filter(
        df.loc[valid_mask, 'y'].values,
        window_length=window_length,
        polyorder=polyorder
    )

    return df


# ==============================================================================
#                         MAIN ENTRY POINT
# ==============================================================================


def process_reconstruction(video_name, config=None):
    """
    Main entry point: runs the full trajectory reconstruction pipeline.

    Steps:
    1. Read boundary_data.json -> lane corners CSV
    2. Compute per-frame homography
    3. Transform ball positions to lane coordinates
    4. Smooth the trajectory
    5. Save outputs

    Parameters
    ----------
    video_name : str
        Name of the video (without extension), e.g. 'cropped_test3'.
    config : module, optional
        Configuration module. If None, uses this module's default config.

    Returns
    -------
    dict
        Dictionary with keys:
            'lane_corners_df': DataFrame of lane corners
            'transformed_df': raw transformed positions
            'smoothed_df': smoothed transformed positions
            'homographies': dict of homography matrices
            'output_dir': path to output directory
    """
    if config is None:
        from . import config

    output_base = os.path.join(config.OUTPUT_DIR, video_name)
    output_dir = os.path.join(output_base, config.OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 3: Trajectory Reconstruction - {video_name}")
    print(f"{'='*60}")

    # --- Step 1: Convert boundary data to lane corners ---
    boundary_json_path = os.path.join(output_base, config.BOUNDARY_DATA_FILENAME)
    if not os.path.exists(boundary_json_path):
        raise FileNotFoundError(
            f"Boundary data not found: {boundary_json_path}\n"
            f"Run Phase 1 (lane detection) first."
        )

    lane_corners_csv = os.path.join(output_dir, 'lane_corners.csv') if config.SAVE_LANE_CORNERS_CSV else None

    print(f"\nStep 1: Converting boundary data to lane corners...")
    lane_corners_df = convert_boundary_json_to_lane_csv(
        boundary_json_path,
        output_csv_path=lane_corners_csv
    )
    print(f"  Corners: TL=({lane_corners_df.iloc[0]['tl_x']:.0f}, {lane_corners_df.iloc[0]['tl_y']:.0f}), "
          f"TR=({lane_corners_df.iloc[0]['tr_x']:.0f}, {lane_corners_df.iloc[0]['tr_y']:.0f}), "
          f"BL=({lane_corners_df.iloc[0]['bl_x']:.0f}, {lane_corners_df.iloc[0]['bl_y']:.0f}), "
          f"BR=({lane_corners_df.iloc[0]['br_x']:.0f}, {lane_corners_df.iloc[0]['br_y']:.0f})")

    # --- Step 2: Compute homographies ---
    print(f"\nStep 2: Computing homographies...")
    print(f"  Lane dimensions: {config.LANE_WIDTH} x {config.LANE_LENGTH}")
    homographies = compute_homographies_per_frame(
        lane_corners_df, config.LANE_WIDTH, config.LANE_LENGTH
    )
    print(f"  Computed {len(homographies)} homography matrices")

    # --- Step 3: Transform ball positions ---
    ball_csv_path = os.path.join(output_base, config.BALL_TRAJECTORY_FILENAME)
    if not os.path.exists(ball_csv_path):
        raise FileNotFoundError(
            f"Ball trajectory not found: {ball_csv_path}\n"
            f"Run Phase 2 (ball detection) first."
        )

    transformed_csv = os.path.join(output_dir, 'transformed_positions.csv') if config.SAVE_TRANSFORMED_CSV else None

    print(f"\nStep 3: Transforming ball positions to lane coordinates...")
    transformed_df = apply_homography_per_frame(
        ball_csv_path, homographies, output_csv_path=transformed_csv
    )
    valid_count = transformed_df['x'].notna().sum()
    print(f"  Transformed {valid_count} valid positions out of {len(transformed_df)} frames")

    # --- Step 4: Smooth trajectory ---
    print(f"\nStep 4: Smoothing trajectory (window={config.SAVGOL_WINDOW}, order={config.SAVGOL_POLYORDER})...")
    smoothed_df = smooth_trajectory(
        transformed_df,
        window_length=config.SAVGOL_WINDOW,
        polyorder=config.SAVGOL_POLYORDER
    )

    if config.SAVE_SMOOTHED_CSV:
        smoothed_csv = os.path.join(output_dir, 'transformed_positions_smoothed.csv')
        smoothed_df.to_csv(smoothed_csv, index=False)
        print(f"  Smoothed trajectory saved to {smoothed_csv}")

    # --- Step 5: Generate visualization ---
    if config.SAVE_VISUALIZATION_VIDEO:
        print(f"\nStep 5: Generating overhead visualization video...")
        from .visualization import generate_overhead_video
        vis_path = generate_overhead_video(smoothed_df, output_dir, video_name, config)
        if vis_path:
            print(f"  Visualization saved to {vis_path}")

    print(f"\n{'='*60}")
    print(f"Phase 3 complete for {video_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    return {
        'lane_corners_df': lane_corners_df,
        'transformed_df': transformed_df,
        'smoothed_df': smoothed_df,
        'homographies': homographies,
        'output_dir': output_dir,
    }
