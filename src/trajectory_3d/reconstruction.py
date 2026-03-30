"""
Core trajectory reconstruction logic - Phase 3

Uses the exact same homography approach as bowling-analysis (worcester):
1. Convert boundary_data.json to per-frame Lane_points.csv (worcester format)
2. Convert ball trajectory CSV to worcester format (frame,x,y,radius)
3. Apply per-frame homography to map ball positions to lane coordinates (106x1829)
4. Post-process: outlier removal, median filter, Savitzky-Golay smoothing

Version: 2.0.0
"""

import json
import os
from collections import deque

import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter


# Lane dimensions in cm (standard bowling lane)
LANE_WIDTH = 106
LANE_LENGTH = 1829


# ==============================================================================
#                 ADAPTER: boundary_data.json -> Lane_points.csv
# ==============================================================================


def _line_y_at_x(line_data, x):
    """Compute Y coordinate of a master line at a given X."""
    x_top = line_data['x_top']
    y_top = line_data['y_top']
    x_bottom = line_data['x_bottom']
    y_bottom = line_data['y_bottom']
    if x_bottom == x_top:
        return y_top
    slope = (y_bottom - y_top) / (x_bottom - x_top)
    return y_top + slope * (x - x_top)


def _line_x_at_y(line_data, y):
    """Compute X coordinate of a master line at a given Y."""
    x_top = line_data['x_top']
    y_top = line_data['y_top']
    x_bottom = line_data['x_bottom']
    y_bottom = line_data['y_bottom']
    if y_bottom == y_top:
        return x_top
    t = (y - y_top) / (y_bottom - y_top)
    return x_top + t * (x_bottom - x_top)


def convert_boundary_json_to_lane_csv(boundary_json_path, output_csv_path=None, num_frames=None):
    """
    Convert bowling-cv's boundary_data.json to worcester's Lane_points.csv format.

    worcester format (per-frame, 9 columns):
        Frame,bottom_left_x,bottom_left_y,bottom_right_x,bottom_right_y,
        up_left_x,up_left_y,up_right_x,up_right_y

    The 4 corners are computed from the intersections of the master lines
    with the foul line (bottom) and top boundary.
    """
    with open(boundary_json_path, 'r') as f:
        bd = json.load(f)

    left = bd['master_left']
    right = bd['master_right']
    foul_y = bd['median_foul_params']['center_y']

    # Top boundary Y
    top_info = bd.get('top_boundary', {})
    top_y = top_info.get('y_position')
    if top_y is None:
        # Try line endpoints
        line_left = top_info.get('line_left')
        line_right = top_info.get('line_right')
        if line_left and line_right:
            top_y = int((line_left[1] + line_right[1]) / 2)
        else:
            top_y = int(foul_y * 0.12)  # estimate

    # Compute 4 corners: intersections of left/right lines with foul/top
    bl_x = _line_x_at_y(left, foul_y)
    bl_y = foul_y
    br_x = _line_x_at_y(right, foul_y)
    br_y = foul_y
    ul_x = _line_x_at_y(left, top_y)
    ul_y = top_y
    ur_x = _line_x_at_y(right, top_y)
    ur_y = top_y

    if num_frames is None or num_frames <= 0:
        num_frames = 1

    rows = []
    for i in range(num_frames):
        rows.append({
            'Frame': i,
            'bottom_left_x': bl_x, 'bottom_left_y': bl_y,
            'bottom_right_x': br_x, 'bottom_right_y': br_y,
            'up_left_x': ul_x, 'up_left_y': ul_y,
            'up_right_x': ur_x, 'up_right_y': ur_y,
        })

    df = pd.DataFrame(rows)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        df.to_csv(output_csv_path, index=False)

    return df


# ==============================================================================
#                 ADAPTER: ball CSV format conversion
# ==============================================================================


def convert_ball_csv_to_worcester(input_csv_path, output_csv_path=None):
    """
    Convert bowling-cv ball detection CSV to worcester format.

    bowling-cv format: frame,x,y,radius_x,radius_y,radius_fitted
    worcester format:  frame,x,y,radius
    """
    df = pd.read_csv(input_csv_path)

    # Map radius column
    if 'radius' not in df.columns:
        if 'radius_fitted' in df.columns:
            df['radius'] = df['radius_fitted']
        elif 'radius_x' in df.columns:
            df['radius'] = df['radius_x']
        else:
            raise ValueError(f"No radius column found in {input_csv_path}. "
                             f"Columns: {list(df.columns)}")

    out = df[['frame', 'x', 'y', 'radius']].copy()

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        out.to_csv(output_csv_path, index=False)

    return out


# ==============================================================================
#          HOMOGRAPHY (same logic as worcester Reconstruction.py)
# ==============================================================================


def compute_homographies_per_frame(lane_csv, width=LANE_WIDTH, height=LANE_LENGTH):
    """
    Compute a homography matrix for each frame from lane corner points.
    Identical to worcester's compute_homographies_per_frame.
    """
    if isinstance(lane_csv, pd.DataFrame):
        df = lane_csv
    else:
        df = pd.read_csv(lane_csv)

    homographies = {}

    for _, row in df.iterrows():
        frame = int(row['Frame'])

        pts_src = np.array([
            [row['bottom_left_x'], row['bottom_left_y']],
            [row['bottom_right_x'], row['bottom_right_y']],
            [row['up_right_x'], row['up_right_y']],
            [row['up_left_x'], row['up_left_y']],
        ], dtype=np.float32)

        pts_dst = np.array([
            [0, height], [width, height], [width, 0], [0, 0],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(pts_src, pts_dst)
        if H is not None:
            homographies[frame] = H

    return homographies


def apply_homography_per_frame(ball_csv, homographies, output_csv=None):
    """
    Transform ball positions using per-frame homography.
    Uses bottom-of-ball point (x, y+radius) as contact point.
    Identical to worcester's apply_homography_per_frame.
    """
    if isinstance(ball_csv, pd.DataFrame):
        df = ball_csv
    else:
        df = pd.read_csv(ball_csv)

    transformed = []

    for _, row in df.iterrows():
        frame_id = int(row['frame'])
        x, y, r = row['x'], row['y'], row['radius']

        if pd.isna(x) or pd.isna(y) or pd.isna(r):
            transformed.append([frame_id, np.nan, np.nan])
            continue

        H = homographies.get(frame_id)
        if H is None:
            H = homographies.get(0)
        if H is None and homographies:
            H = next(iter(homographies.values()))
        if H is None:
            continue

        point = np.array([[[x, y + r]]], dtype=np.float32)
        tp = cv2.perspectiveTransform(point, H)[0][0]
        transformed.append([frame_id, tp[0], tp[1]])

    out_df = pd.DataFrame(transformed, columns=['frame', 'x', 'y'])

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        out_df.to_csv(output_csv, index=False)

    return out_df


# ==============================================================================
#      POST-PROCESSING (same logic as worcester Post_processing_positions.py)
# ==============================================================================


def _remove_low_y(df):
    """Remove out-of-bounds y coordinates."""
    df = df.copy()
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    mask = (df['y'] > LANE_LENGTH - 80) | (df['y'] < 30)
    df.loc[mask, ['x', 'y']] = np.nan
    return df


def _remove_rolling_outliers(df, threshold=2.5, window_size=2):
    """Remove outliers using rolling median absolute deviation."""
    df = df.copy()
    initial_nan = df[['x', 'y']].isna().any(axis=1)

    def rolling_median_mad(values, ws):
        medians, mads = [], []
        window = deque(maxlen=ws)
        for v in values:
            window.append(v)
            if len(window) == ws:
                med = np.median(window)
                mad = np.median(np.abs(np.array(window) - med))
                medians.append(med)
                mads.append(mad)
            else:
                medians.append(np.nan)
                mads.append(np.nan)
        return np.array(medians), np.array(mads)

    x_med, _ = rolling_median_mad(df['x'].values, window_size)
    y_med, _ = rolling_median_mad(df['y'].values, window_size)

    distances = np.sqrt((df['x'].values - x_med) ** 2 + (df['y'].values - y_med) ** 2)
    med_dist = np.nanmedian(distances)
    mad_dist = np.nanmedian(np.abs(distances - med_dist))

    if mad_dist == 0:
        return df

    mod_z = 0.6745 * (distances - med_dist) / mad_dist
    outliers = (np.abs(mod_z) >= threshold) & ~initial_nan
    df.loc[outliers, ['x', 'y']] = np.nan
    return df


def _median_filter(df, kernel_size=3):
    """Apply median filter and remove non-positive values."""
    df = df.copy()
    df['x'] = medfilt(df['x'], kernel_size=kernel_size)
    df['y'] = medfilt(df['y'], kernel_size=kernel_size)
    df = df[df['x'] > 0]
    df = df[df['y'] > 0]
    return df


def _savgol_smooth(df, window_length=45, polyorder=3):
    """Apply Savitzky-Golay filter."""
    df = df.copy()
    n = len(df)
    if n < window_length:
        window_length = n if n % 2 == 1 else n - 1
    if window_length <= polyorder:
        return df
    df['x'] = savgol_filter(df['x'], window_length=window_length, polyorder=polyorder)
    df['y'] = savgol_filter(df['y'], window_length=window_length, polyorder=polyorder)
    return df


def _interpolate_missing(df):
    """Interpolate missing coordinates and apply final smoothing."""
    df = df.copy().set_index('frame')
    full_idx = range(df.index.min(), df.index.max() + 1)
    df = df.reindex(full_idx)
    df['x'] = df['x'].interpolate(method='linear').bfill().ffill()
    df['y'] = df['y'].interpolate(method='linear').bfill().ffill()
    df = df.reset_index().rename(columns={'index': 'frame'})
    df = _savgol_smooth(df)
    return df


def post_process_transformed(df):
    """Full post-processing pipeline (same as worcester)."""
    df = _remove_low_y(df)
    df = _remove_rolling_outliers(df)
    df = _median_filter(df)
    df = _interpolate_missing(df)
    return df


# ==============================================================================
#                           MAIN ENTRY POINT
# ==============================================================================


def process_reconstruction(video_name, config=None):
    """
    Full trajectory reconstruction pipeline.

    1. Convert boundary_data.json -> Lane_points.csv (worcester format)
    2. Convert ball CSV -> worcester format (frame,x,y,radius)
    3. Compute per-frame homographies
    4. Transform ball positions to lane coordinates
    5. Post-process (outlier removal, smoothing, interpolation)
    6. Generate visualization
    """
    if config is None:
        from . import config

    output_base = os.path.join(config.OUTPUT_DIR, video_name)
    output_dir = os.path.join(output_base, config.OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 3: Trajectory Reconstruction - {video_name}")
    print(f"{'='*60}")

    # --- Step 1: Convert boundary data to Lane_points.csv ---
    boundary_json = os.path.join(output_base, config.BOUNDARY_DATA_FILENAME)
    if not os.path.exists(boundary_json):
        raise FileNotFoundError(f"Not found: {boundary_json}. Run Phase 1 first.")

    # Get frame count from ball CSV
    ball_csv_orig = os.path.join(output_base, config.BALL_TRAJECTORY_FILENAME)
    if not os.path.exists(ball_csv_orig):
        raise FileNotFoundError(f"Not found: {ball_csv_orig}. Run Phase 2 first.")

    ball_df_orig = pd.read_csv(ball_csv_orig)
    num_frames = int(ball_df_orig['frame'].max()) + 1

    lane_csv = os.path.join(output_dir, 'Lane_points.csv')
    print(f"\nStep 1: Converting boundary data to lane corners ({num_frames} frames)...")
    lane_df = convert_boundary_json_to_lane_csv(boundary_json, lane_csv, num_frames)
    row0 = lane_df.iloc[0]
    print(f"  BL=({row0['bottom_left_x']:.0f},{row0['bottom_left_y']:.0f}) "
          f"BR=({row0['bottom_right_x']:.0f},{row0['bottom_right_y']:.0f}) "
          f"UL=({row0['up_left_x']:.0f},{row0['up_left_y']:.0f}) "
          f"UR=({row0['up_right_x']:.0f},{row0['up_right_y']:.0f})")

    # --- Step 2: Convert ball CSV to worcester format ---
    ball_csv = os.path.join(output_dir, 'ball_positions.csv')
    print(f"\nStep 2: Converting ball CSV to standard format...")
    ball_df = convert_ball_csv_to_worcester(ball_csv_orig, ball_csv)
    print(f"  {len(ball_df)} ball positions, radius column mapped")

    # --- Step 3: Compute homographies ---
    print(f"\nStep 3: Computing per-frame homographies (lane {LANE_WIDTH}x{LANE_LENGTH})...")
    homographies = compute_homographies_per_frame(lane_csv, LANE_WIDTH, LANE_LENGTH)
    print(f"  {len(homographies)} homography matrices computed")

    # --- Step 4: Transform positions ---
    raw_csv = os.path.join(output_dir, 'transformed_positions_raw.csv')
    print(f"\nStep 4: Transforming ball positions to lane coordinates...")
    raw_df = apply_homography_per_frame(ball_csv, homographies, raw_csv)
    valid = raw_df['x'].notna().sum()
    print(f"  {valid} valid transformed positions")

    # --- Step 5: Post-process ---
    print(f"\nStep 5: Post-processing (outlier removal + smoothing)...")
    processed_df = post_process_transformed(raw_df)
    processed_csv = os.path.join(output_dir, 'transformed_positions_smoothed.csv')
    processed_df.to_csv(processed_csv, index=False)
    print(f"  {len(processed_df)} final points saved to {processed_csv}")

    # --- Step 6: Visualization ---
    if config.SAVE_VISUALIZATION_VIDEO:
        print(f"\nStep 6: Generating overhead visualization...")
        from .visualization import generate_overhead_video
        vis_path = generate_overhead_video(processed_df, output_dir, video_name, config)
        if vis_path:
            print(f"  Saved to {vis_path}")

    valid_final = processed_df.dropna(subset=['x', 'y'])
    print(f"\n{'='*60}")
    print(f"Phase 3 complete for {video_name}")
    print(f"  Points: {len(valid_final)}")
    if len(valid_final) > 0:
        print(f"  X range: {valid_final['x'].min():.1f} - {valid_final['x'].max():.1f} (lane width: {LANE_WIDTH})")
        print(f"  Y range: {valid_final['y'].min():.1f} - {valid_final['y'].max():.1f} (lane length: {LANE_LENGTH})")
    print(f"{'='*60}\n")

    return {
        'lane_corners_df': lane_df,
        'transformed_df': raw_df,
        'smoothed_df': processed_df,
        'homographies': homographies,
        'output_dir': output_dir,
    }
