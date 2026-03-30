"""
Overhead trajectory visualization for Phase 3.

Generates a video showing the ball trajectory on a top-down lane diagram
using the transformed lane-space coordinates.

Version: 1.0.0
Authors: Pedro Roriz
Created: March 30, 2026
"""

import os

import cv2
import numpy as np
import pandas as pd


def _draw_lane_background(canvas, lane_w, lane_h, canvas_w, canvas_h, config):
    """
    Draw a top-down lane diagram on the canvas.

    Parameters
    ----------
    canvas : np.ndarray
        BGR image to draw on.
    lane_w : int
        Lane width in real units (e.g. 106).
    lane_h : int
        Lane height in real units (e.g. 1829).
    canvas_w : int
        Canvas width in pixels.
    canvas_h : int
        Canvas height in pixels.
    config : module
        Configuration with color settings.
    """
    # Fill background dark
    canvas[:] = (40, 40, 40)

    # Compute scale to fit lane into canvas with some margin
    margin = 20
    usable_w = canvas_w - 2 * margin
    usable_h = canvas_h - 2 * margin

    scale_x = usable_w / lane_w
    scale_y = usable_h / lane_h
    scale = min(scale_x, scale_y)

    # Lane rectangle in canvas pixels
    lw = int(lane_w * scale)
    lh = int(lane_h * scale)
    x_offset = (canvas_w - lw) // 2
    y_offset = (canvas_h - lh) // 2

    # Draw lane surface
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + lw, y_offset + lh),
                  config.VIS_LANE_COLOR, -1)

    # Draw gutter strips (5% on each side)
    gutter_w = max(int(lw * 0.05), 2)
    cv2.rectangle(canvas, (x_offset, y_offset),
                  (x_offset + gutter_w, y_offset + lh), config.VIS_GUTTER_COLOR, -1)
    cv2.rectangle(canvas, (x_offset + lw - gutter_w, y_offset),
                  (x_offset + lw, y_offset + lh), config.VIS_GUTTER_COLOR, -1)

    # Draw foul line at bottom
    foul_y = y_offset + lh
    cv2.line(canvas, (x_offset, foul_y), (x_offset + lw, foul_y), (0, 0, 255), 2)

    # Draw pin area marker at top
    cv2.line(canvas, (x_offset, y_offset), (x_offset + lw, y_offset), (0, 255, 0), 2)

    # Labels
    cv2.putText(canvas, "PINS", (x_offset + lw // 2 - 20, y_offset - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, "FOUL LINE", (x_offset + lw // 2 - 40, foul_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return x_offset, y_offset, scale


def _lane_to_canvas(x, y, x_offset, y_offset, scale, lane_h):
    """
    Convert lane coordinates to canvas pixel coordinates.

    In lane coords: y=0 is pins (top), y=LANE_LENGTH is foul line (bottom).
    On canvas: top is pins, bottom is foul line (same orientation).

    Parameters
    ----------
    x, y : float
        Position in lane coordinates.
    x_offset, y_offset : int
        Canvas offsets from _draw_lane_background.
    scale : float
        Scale factor from _draw_lane_background.
    lane_h : int
        Lane height in real units.

    Returns
    -------
    tuple
        (px, py) pixel coordinates on canvas.
    """
    px = int(x_offset + x * scale)
    py = int(y_offset + y * scale)
    return px, py


def generate_overhead_video(trajectory_df, output_dir, video_name, config):
    """
    Generate a video showing the ball trajectory on a top-down lane view.

    The ball moves from the foul line (bottom) toward the pins (top).
    Each frame of the video corresponds to a frame in the trajectory,
    progressively revealing the trajectory path.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        DataFrame with columns: frame, x, y (in lane coordinates).
    output_dir : str
        Directory to save the output video.
    video_name : str
        Name for the output file.
    config : module
        Configuration with visualization parameters.

    Returns
    -------
    str or None
        Path to the generated video, or None on failure.
    """
    # Filter to valid points only
    valid = trajectory_df.dropna(subset=['x', 'y']).reset_index(drop=True)

    if len(valid) < 2:
        print("  Warning: Not enough valid points for visualization.")
        return None

    canvas_w = config.VIS_CANVAS_WIDTH
    canvas_h = config.VIS_CANVAS_HEIGHT
    lane_w = config.LANE_WIDTH
    lane_h = config.LANE_LENGTH

    # Setup video writer
    output_path = os.path.join(output_dir, f'{video_name}_trajectory_overhead.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, config.VIS_FPS, (canvas_w, canvas_h))

    if not writer.isOpened():
        print(f"  Error: Could not create video writer at {output_path}")
        return None

    # Draw the static lane background once
    base_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x_offset, y_offset, scale = _draw_lane_background(
        base_canvas, lane_w, lane_h, canvas_w, canvas_h, config
    )

    # Pre-compute all canvas positions
    canvas_points = []
    for _, row in valid.iterrows():
        px, py = _lane_to_canvas(row['x'], row['y'], x_offset, y_offset, scale, lane_h)
        canvas_points.append((int(row['frame']), px, py))

    # Generate frames - progressively reveal trajectory
    for i in range(len(canvas_points)):
        frame = base_canvas.copy()

        # Draw trajectory trail up to current point
        if i > 0:
            for j in range(1, i + 1):
                pt1 = (canvas_points[j - 1][1], canvas_points[j - 1][2])
                pt2 = (canvas_points[j][1], canvas_points[j][2])
                cv2.line(frame, pt1, pt2, config.VIS_TRAJECTORY_COLOR, config.VIS_LINE_WIDTH)

        # Draw current ball position
        cx, cy = canvas_points[i][1], canvas_points[i][2]
        cv2.circle(frame, (cx, cy), config.VIS_BALL_RADIUS, config.VIS_BALL_COLOR, -1)

        # Frame info overlay
        frame_num = canvas_points[i][0]
        cv2.putText(frame, f"Frame: {frame_num}", (10, canvas_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pos: ({valid.iloc[i]['x']:.1f}, {valid.iloc[i]['y']:.1f})",
                    (10, canvas_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    print(f"  Generated {len(canvas_points)} frames")
    return output_path
