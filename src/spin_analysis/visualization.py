"""
Spin Visualization - 3D sphere video showing cumulative ball rotation.

Renders a rotating sphere with a checkerboard pattern whose north pole
tracks the estimated rotation axis. The sphere accumulates per-frame
rotations to show continuous spin. An arrow indicates angular velocity.

Uses matplotlib 3D surface plots and FFMpegWriter for video output.

Version: 1.0.0
Created: March 30, 2026
"""

import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from . import config as default_config


# ==============================================================================
#                            ROTATION HELPERS
# ==============================================================================


def _rotation_matrix(axis, theta):
    """
    Compute rotation matrix from axis-angle using Rodrigues' formula (quaternion form).

    Parameters
    ----------
    axis : ndarray, shape (3,)
        Unit rotation axis.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),       2*(b*d + a*c)],
        [2*(b*c + a*d),         a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),         2*(c*d + a*b),         a*a + d*d - b*b - c*c],
    ])


def _align_north_pole_to_vector(target_vector):
    """
    Compute rotation matrix that aligns the north pole [0,0,1] to target_vector.

    Parameters
    ----------
    target_vector : ndarray, shape (3,)
        Desired direction for the sphere's north pole.

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.
    """
    target_vector = target_vector / np.linalg.norm(target_vector)
    north_pole = np.array([0, 0, 1])

    if np.allclose(target_vector, north_pole):
        return np.eye(3)

    if np.allclose(target_vector, -north_pole):
        return _rotation_matrix(np.array([1, 0, 0]), np.pi)

    axis = np.cross(north_pole, target_vector)
    angle = np.arccos(np.clip(np.dot(north_pole, target_vector), -1.0, 1.0))
    return _rotation_matrix(axis, angle)


# ==============================================================================
#                            SPHERE GEOMETRY
# ==============================================================================


def _create_checkerboard_sphere(cfg):
    """
    Create sphere mesh with checkerboard color pattern.

    Returns
    -------
    tuple : (x0, y0, z0, colors) - mesh coordinates and face colors.
    """
    scaling = cfg.SPHERE_SCALING
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x0 = np.cos(u) * np.sin(v) * scaling
    y0 = np.sin(u) * np.sin(v) * scaling
    z0 = np.cos(v) * scaling

    colors = np.empty(u.shape, dtype=object)

    lat_step = np.pi / 50
    equator_thickness = 2 * lat_step
    pole_thickness = lat_step

    equator_lower = (np.pi / 2) - equator_thickness / 2
    equator_upper = (np.pi / 2) + equator_thickness / 2
    north_pole_upper = pole_thickness
    south_pole_lower = np.pi - pole_thickness

    band_colors = cfg.BAND_COLORS
    highlight_color = cfg.HIGHLIGHT_COLOR

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            v_val = v[i, j]
            if (equator_lower <= v_val <= equator_upper
                    or v_val <= north_pole_upper
                    or v_val >= south_pole_lower):
                colors[i, j] = highlight_color
            else:
                lat_index = int(v_val // (np.pi / cfg.NUM_LAT_BANDS))
                lon_index = int(u[i, j] // (2 * np.pi / cfg.NUM_LON_BANDS))
                colors[i, j] = band_colors[(lat_index + lon_index) % 2]

    return x0, y0, z0, colors


def _apply_rotation(R, x0, y0, z0):
    """Apply a 3x3 rotation matrix to sphere mesh coordinates."""
    x = R[0, 0] * x0 + R[0, 1] * y0 + R[0, 2] * z0
    y = R[1, 0] * x0 + R[1, 1] * y0 + R[1, 2] * z0
    z = R[2, 0] * x0 + R[2, 1] * y0 + R[2, 2] * z0
    return x, y, z


def _draw_axis_arrow(ax, point, angular_velocity, cfg):
    """Draw the rotation axis arrow indicator."""
    origin = -point * cfg.ARROW_ORIGIN_SCALE
    vector = -point * abs(angular_velocity) * cfg.ARROW_SCALE

    ax.plot(
        [origin[0], vector[0] + origin[0]],
        [origin[1], vector[1] + origin[1]],
        [origin[2], vector[2] + origin[2]],
        color='red', linewidth=4, zorder=10,
    )
    ax.scatter(
        [vector[0] + origin[0]],
        [vector[1] + origin[1]],
        [vector[2] + origin[2]],
        color='red', s=60, marker='o', zorder=10,
    )


def _draw_angular_velocity_text(ax, angular_velocity):
    """Draw angular velocity label below the sphere."""
    if np.isnan(angular_velocity):
        text = "N/D rad/s"
    elif angular_velocity > 0:
        text = f"{angular_velocity:.2f}rad/s"
    else:
        text = f"{angular_velocity:.2f}rad/s"

    ax.text2D(
        0.0, 0.09, text,
        ha='center', va='center', fontsize=30, color='black',
        bbox=dict(
            facecolor='none', edgecolor='black',
            boxstyle='round,pad=0.3', linewidth=2,
        ),
    )


# ==============================================================================
#                            VIDEO CREATION
# ==============================================================================


def create_sphere_video(rotation_csv_path, output_path=None, config=None):
    """
    Create a 3D sphere visualization video from processed rotation data.

    The sphere accumulates frame-by-frame rotations, with its north pole
    aligned to the estimated rotation axis. A red arrow shows the
    angular velocity direction and magnitude.

    Parameters
    ----------
    rotation_csv_path : str
        Path to CSV with columns: frame, x_axis, y_axis, z_axis, angle.
    output_path : str, optional
        Output video path. Defaults to sphere_video.mp4 in same directory.
    config : module, optional
        Configuration module. Defaults to spin_analysis.config.

    Returns
    -------
    str
        Path to the output video file.
    """
    if config is None:
        config = default_config

    cfg = config

    df = pd.read_csv(rotation_csv_path)

    if output_path is None:
        output_dir = os.path.dirname(rotation_csv_path)
        output_path = os.path.join(output_dir, 'sphere_video.mp4')

    # Get video FPS from the original video if possible, otherwise default to 30
    fps = 30

    # Create sphere geometry
    x0, y0, z0, colors = _create_checkerboard_sphere(cfg)

    # Initialize matplotlib figure
    fig = plt.figure(figsize=(cfg.SPHERE_FIGURE_SIZE, cfg.SPHERE_FIGURE_SIZE))
    fig.patch.set_facecolor(cfg.SPHERE_BG_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(cfg.SPHERE_BG_COLOR)

    metadata = dict(title='Bowling Ball Spin Visualization')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    first_valid_index = df['x_axis'].first_valid_index()
    last_valid_index = df['x_axis'].last_valid_index()
    mean_angle = df['angle'].mean()

    # Cumulative rotation state
    dtheta = 0

    with writer.saving(fig, output_path, cfg.SPHERE_DPI):
        for i, row in df.iterrows():
            ax.cla()

            # CASE 1: Before first valid axis data
            if pd.isna(row['x_axis']) and (first_valid_index is None or i <= first_valid_index):
                if first_valid_index is not None:
                    point = np.array([
                        df.loc[first_valid_index, 'x_axis'],
                        df.loc[first_valid_index, 'y_axis'],
                        df.loc[first_valid_index, 'z_axis'],
                    ])
                    angular_velocity = df.loc[first_valid_index, 'angle'] * fps
                    R_align = _align_north_pole_to_vector(point)
                    x, y, z = _apply_rotation(R_align, x0, y0, z0)

                    ax.plot_surface(x, y, z, facecolors=colors, edgecolor='gray',
                                    alpha=1, linewidth=0.2)
                    _draw_axis_arrow(ax, point, angular_velocity, cfg)
                else:
                    # No valid data at all - draw static sphere
                    ax.plot_surface(x0, y0, z0, facecolors=colors, edgecolor='gray',
                                    alpha=1, linewidth=0.2)

                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-2, 2])
                ax.view_init(elev=-90, azim=-90)
                ax.axis('off')
                _draw_angular_velocity_text(ax, np.nan)
                writer.grab_frame()
                continue

            # CASE 2: After last valid axis data
            if pd.isna(row['x_axis']) and last_valid_index is not None and i >= last_valid_index:
                point = np.array([
                    df.loc[last_valid_index, 'x_axis'],
                    df.loc[last_valid_index, 'y_axis'],
                    df.loc[last_valid_index, 'z_axis'],
                ])
                angular_velocity = df.loc[last_valid_index, 'angle'] * fps
                R_align = _align_north_pole_to_vector(point)
                x, y, z = _apply_rotation(R_align, x0, y0, z0)

                ax.plot_surface(x, y, z, facecolors=colors, edgecolor='gray',
                                alpha=1, linewidth=0.2)
                _draw_axis_arrow(ax, point, angular_velocity, cfg)

                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-2, 2])
                ax.view_init(elev=-90, azim=-90)
                ax.axis('off')
                _draw_angular_velocity_text(ax, np.nan)
                writer.grab_frame()
                continue

            # CASE 3: Valid axis data - accumulate rotation
            axis_vec = row[['x_axis', 'y_axis', 'z_axis']].values.astype(float)

            if np.linalg.norm(axis_vec) < 1e-6:
                writer.grab_frame()
                continue

            point = axis_vec.copy()
            R_align = _align_north_pole_to_vector(point)

            # Accumulate angle
            if pd.isna(row['angle']):
                dtheta -= mean_angle if not np.isnan(mean_angle) else 0
            else:
                dtheta -= row['angle']

            R_spin = _rotation_matrix(np.array([0, 0, 1]), dtheta)
            R_total = R_align @ R_spin

            x, y, z = _apply_rotation(R_total, x0, y0, z0)

            ax.plot_surface(x, y, z, facecolors=colors, edgecolor='gray',
                            alpha=1, linewidth=0.01)

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.view_init(elev=-90, azim=-90)
            ax.axis('off')

            angular_velocity = row['angle'] * fps if not pd.isna(row['angle']) else np.nan
            _draw_angular_velocity_text(ax, angular_velocity)
            _draw_axis_arrow(ax, point, angular_velocity if not np.isnan(angular_velocity) else 0, cfg)

            writer.grab_frame()

    plt.close(fig)
    print(f"Sphere video saved to {output_path}")
    return output_path
