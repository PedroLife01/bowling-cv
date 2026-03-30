"""
Post-processing for spin rotation data.

Chains the following steps:
1. Sign correction: flip axis signs based on z_axis average direction
2. Outlier removal: linear regression residual-based filtering (multiple passes)
3. Interpolation: fill gaps using valid ball detection frames
4. Gaussian smoothing: smooth axis components over contiguous segments
5. Axis scaling and z_axis recomputation from unit-vector constraint
6. Angle processing: outlier removal, interpolation, rolling-window smoothing

Input:  rotation_data.csv (raw detection output)
Output: rotation_data_processed.csv (cleaned and smoothed)

Version: 1.0.0
Created: March 30, 2026
"""

import os

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

from . import config as default_config


# ==============================================================================
#                             OUTLIER REMOVAL
# ==============================================================================


def remove_outliers(df, threshold=0.50):
    """
    Remove outliers based on linear regression residuals.

    Fits a linear model to x_axis and y_axis as functions of frame number,
    then removes points whose absolute residual exceeds the threshold.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: frame, x_axis, y_axis, z_axis, angle.
    threshold : float
        Maximum allowed absolute residual.

    Returns
    -------
    DataFrame with outlier rows set to NaN for axis/angle columns.
    """
    df = df.copy()
    valid_mask = df['x_axis'].notna() & df['y_axis'].notna()
    valid_df = df[valid_mask]

    if len(valid_df) < 2:
        return df

    x_model = LinearRegression().fit(valid_df[['frame']], valid_df['x_axis'])
    y_model = LinearRegression().fit(valid_df[['frame']], valid_df['y_axis'])

    x_pred = x_model.predict(valid_df[['frame']])
    y_pred = y_model.predict(valid_df[['frame']])

    x_error = np.abs(valid_df['x_axis'] - x_pred)
    y_error = np.abs(valid_df['y_axis'] - y_pred)

    within_threshold = (x_error <= threshold) & (y_error <= threshold)
    outlier_indices = valid_df.index[~within_threshold]

    df.loc[outlier_indices, ['x_axis', 'y_axis', 'z_axis', 'angle']] = np.nan

    return df


def _remove_angle_outliers(series, threshold=1.0):
    """Remove angle outliers using z-score filtering."""
    if series.dropna().empty:
        return series
    z_scores = (series - series.mean()) / series.std()
    series = series.copy()
    series.loc[abs(z_scores) > threshold] = np.nan
    return series


# ==============================================================================
#                             SMOOTHING & INTERPOLATION
# ==============================================================================


def smooth_data(series, sigma=10):
    """
    Apply Gaussian smoothing to a pandas Series, handling NaN segments.

    Finds contiguous non-NaN segments and applies gaussian_filter1d
    independently to each segment. NaN values remain NaN.

    Parameters
    ----------
    series : Series
        Data to smooth (may contain NaN gaps).
    sigma : float
        Standard deviation for Gaussian kernel.

    Returns
    -------
    Series with smoothed values.
    """
    smoothed = series.copy()
    not_nan = series.notna()
    if not not_nan.any():
        return smoothed

    group = (not_nan != not_nan.shift()).cumsum()

    for grp_id, grp_data in group[not_nan].groupby(group):
        segment_indices = grp_data.index
        segment_values = series.loc[segment_indices]
        interpolated = segment_values.interpolate()
        smoothed_segment = gaussian_filter1d(interpolated, sigma=sigma, mode='nearest')
        smoothed.loc[segment_indices] = smoothed_segment

    return smoothed


def fill_frames(data, method='linear'):
    """
    Fill missing values using interpolation.

    Parameters
    ----------
    data : Series or DataFrame
        Data with NaN gaps.
    method : str
        Interpolation method ('linear', 'cubic', etc.).

    Returns
    -------
    Interpolated data.
    """
    return data.interpolate(method=method, limit_direction='both')


def _smooth_series(series, first_idx, last_idx, window=5):
    """Apply rolling mean smoothing to a segment of a series."""
    smoothed = series.copy()
    segment = (
        series[first_idx:last_idx + 1].rolling(window=window, center=True).mean()
    )
    segment = (
        segment.interpolate(method='cubic', limit_direction='both').bfill().ffill()
    )
    smoothed.loc[first_idx:last_idx + 1] = segment
    return smoothed


def _interpolate_axes_from_b(A, B):
    """Interpolate axis values in A using valid frames from B."""
    if 'frame' not in A.columns or 'frame' not in B.columns:
        raise ValueError("Both DataFrames must contain a 'frame' column.")

    valid_frames = B.dropna(subset=['x', 'y'])['frame'].unique()
    A_interp = A.copy()
    valid_mask = A_interp['frame'].isin(valid_frames)

    for axis in ['x_axis', 'y_axis', 'z_axis']:
        axis_series = A_interp.loc[valid_mask, axis]
        interpolated_values = axis_series.interpolate(
            method='linear', limit_direction='both'
        )
        A_interp.loc[valid_mask, axis] = interpolated_values

    return A_interp


def _enforce_non_decreasing_x_axis(df):
    """Enforce monotonically non-decreasing x_axis values."""
    if 'x_axis' not in df.columns:
        raise ValueError("DataFrame must contain 'x_axis' column.")
    last_valid = df.loc[0, 'x_axis']
    for i in range(1, len(df)):
        if df.loc[i, 'x_axis'] < last_valid:
            df.loc[i, 'x_axis'] = last_valid
        else:
            last_valid = df.loc[i, 'x_axis']
    return df


def _scale_x_axis(df):
    """Scale x_axis values linearly between first and last valid entries."""
    df = df.copy()
    valid_x = df['x_axis'].dropna()
    if valid_x.empty:
        return df
    first_idx = valid_x.index[0]
    last_idx = valid_x.index[-1]
    x_start = valid_x.iloc[0]
    x_end = 1 / valid_x.iloc[-1] if valid_x.iloc[-1] != 0 else 1.0
    num_rows = last_idx - first_idx + 1
    scale_factors = np.linspace(x_start, x_end, num_rows)
    df.loc[first_idx:last_idx, 'x_axis'] = (
        df.loc[first_idx:last_idx, 'x_axis'] * scale_factors
    )
    return df


def _scale_y_axis(df):
    """Scale y_axis values linearly from 1 to 0 between first and last valid entries."""
    df = df.copy()
    valid_y = df['y_axis'].dropna()
    if valid_y.empty:
        return df
    first_idx = valid_y.index[0]
    last_idx = valid_y.index[-1]
    num_rows = last_idx - first_idx + 1
    scale_factors = np.linspace(1, 0, num_rows)
    df.loc[first_idx:last_idx, 'y_axis'] = (
        df.loc[first_idx:last_idx, 'y_axis'] * scale_factors
    )
    return df


def _compute_z_axis_from_xy(df, z_axis_avg):
    """Recompute z_axis from unit-vector constraint: z = sqrt(1 - x^2 - y^2)."""
    df = df.copy()
    squared_sum = df['x_axis'] ** 2 + df['y_axis'] ** 2
    z_values = 1 - squared_sum
    z_values[z_values < 0] = np.nan
    if z_axis_avg < 0:
        df['z_axis'] = -np.sqrt(z_values)
    else:
        df['z_axis'] = np.sqrt(z_values)
    return df


# ==============================================================================
#                             MAIN PROCESSING FUNCTION
# ==============================================================================


def process_rotation_data(csv_path, config=None, ball_csv_path=None, video_path=None):
    """
    Full post-processing pipeline for rotation data.

    Parameters
    ----------
    csv_path : str
        Path to rotation_data.csv (raw detection output).
    config : module, optional
        Configuration module. Defaults to spin_analysis.config.
    ball_csv_path : str, optional
        Path to original ball trajectory CSV (for frame interpolation).
    video_path : str, optional
        Path to original video (for frame count).

    Returns
    -------
    str
        Path to the output rotation_data_processed.csv.
    """
    if config is None:
        config = default_config

    cfg = config
    df = pd.read_csv(csv_path)

    if df.empty or 'x_axis' not in df.columns:
        print("Warning: Empty or invalid rotation data. Skipping post-processing.")
        return csv_path

    output_dir = os.path.dirname(csv_path)
    output_csv_path = os.path.join(output_dir, 'rotation_data_processed.csv')

    # --- Step 1: Sign correction based on z_axis average ---
    z_axis_avg = df['z_axis'].mean()
    flip_condition = df['z_axis'] > cfg.Z_AXIS_FLIP_THRESHOLD if z_axis_avg < 0 \
        else df['z_axis'] < cfg.Z_AXIS_FLIP_THRESHOLD
    df.loc[flip_condition, ['x_axis', 'y_axis', 'z_axis']] *= -1

    x_axis_avg = df['x_axis'].mean()
    df_x = df.copy()
    x_condition = (
        df_x['x_axis'] > -x_axis_avg + 0.1 if x_axis_avg < 0
        else df_x['x_axis'] < -x_axis_avg + 0.1
    )
    df_x.loc[x_condition, ['x_axis', 'y_axis', 'z_axis', 'angle']] = np.nan

    y_axis_avg = df_x['y_axis'].mean()
    df_y = df_x.copy()
    y_condition = (
        df_y['y_axis'] > -y_axis_avg if y_axis_avg < 0
        else df_y['y_axis'] < -y_axis_avg
    )
    df_y.loc[y_condition, ['x_axis', 'y_axis', 'z_axis', 'angle']] = np.nan

    # --- Step 2: Outlier removal (multiple passes) ---
    filtered_df = df_y
    for threshold in cfg.OUTLIER_PASSES:
        filtered_df = remove_outliers(filtered_df, threshold=threshold)

    # --- Step 3: Interpolation using original ball detection frames ---
    if ball_csv_path and video_path:
        from .detection import _fill_frames
        df_original = _fill_frames(video_path, ball_csv_path)
        result_df = _interpolate_axes_from_b(filtered_df, df_original)
    else:
        result_df = filtered_df

    # --- Step 4: Gaussian smoothing ---
    smoothed_df = result_df.copy()
    for axis in ['x_axis', 'y_axis', 'z_axis']:
        smoothed_df[axis] = smooth_data(result_df[axis], sigma=cfg.SIGMA)

    smoothed_df = _enforce_non_decreasing_x_axis(smoothed_df)

    # --- Step 5: Scaling and z_axis recomputation ---
    df_scaled = _scale_x_axis(smoothed_df)
    df_scaled = _scale_y_axis(df_scaled)
    df_processed = _compute_z_axis_from_xy(df_scaled, z_axis_avg)

    # Fill position data from original if available
    if ball_csv_path and video_path:
        mask = (
            df_processed['x'].isna()
            & df_processed['y'].isna()
            & df_processed['radius'].isna()
            & df_original['x'].notna()
            & df_original['y'].notna()
            & df_original['radius'].notna()
        )
        df_processed.loc[mask, ['x', 'y', 'radius']] = df_original.loc[
            mask, ['x', 'y', 'radius']
        ]

    # --- Step 6: Angle post-processing ---
    df_angle = df.copy()
    df_angle['angle'] = _remove_angle_outliers(
        df_angle['angle'], threshold=cfg.ANGLE_OUTLIER_THRESHOLD
    )

    first_valid_index = df_processed['x'].first_valid_index()
    last_valid_index = df_processed['x'].last_valid_index()

    if first_valid_index is not None and last_valid_index is not None:
        df_angle.loc[first_valid_index:last_valid_index, 'angle'] = (
            df_angle.loc[first_valid_index:last_valid_index, 'angle']
            .interpolate(method='linear', limit_direction='both')
        )
        for _ in range(cfg.ANGLE_SMOOTH_PASSES):
            df_angle['angle'] = _smooth_series(
                df_angle['angle'].copy(),
                first_valid_index, last_valid_index,
                window=cfg.ANGLE_SMOOTH_WINDOW,
            )

    df_processed['angle'] = df_angle['angle']

    # Save
    df_processed.to_csv(output_csv_path, index=False)
    print(f"Saved processed rotation data to {output_csv_path}")

    return output_csv_path
