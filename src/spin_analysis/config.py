"""
Configuration file for spin/rotation analysis - Phase 5
Edit parameters here without touching the main code.

Version: 1.0.0
Created: March 30, 2026
"""

import os

# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# ============================================
# OPTICAL FLOW PARAMETERS
# ============================================

# Shi-Tomasi corner detection (cv2.goodFeaturesToTrack)
MAX_CORNERS = 100           # Maximum number of corners to detect in ROI
QUALITY_LEVEL = 0.001       # Minimum quality of corners (lower = more features)
MIN_DISTANCE = 3            # Minimum distance between detected corners (pixels)
BLOCK_SIZE = 3              # Size of averaging block for corner detection

# Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK)
WINDOW_SIZE = 21            # Size of search window at each pyramid level
MAX_PYRAMID_LEVEL = 3       # Maximum number of pyramid levels
LK_MAX_ITER = 15            # Maximum number of iterations for LK refinement
LK_EPSILON = 0.01           # Convergence epsilon for LK iterations

# Bidirectional (forward-backward) error threshold
FB_ERROR_THRESHOLD = 10.0   # Maximum allowed forward-backward error (pixels)

# ============================================
# ROI PARAMETERS
# ============================================

ROI_MARGIN = 2              # Extra pixels around ball radius for ROI extraction
MASK_RADIUS_FACTOR = 0.8    # Fraction of ball radius used for circular feature mask

# ============================================
# POINT FILTERING
# ============================================

# Movement thresholds relative to ball radius
LOW_MOVEMENT_FACTOR = 5     # Minimum displacement = radius / LOW_MOVEMENT_FACTOR
LENIENT_LOW_MOVEMENT_FACTOR = 10  # More lenient factor for retry
MIN_POINTS_FOR_ROTATION = 3  # Minimum 3D point pairs needed for Kabsch

# ============================================
# KABSCH ALGORITHM
# ============================================

# Rotation is computed via SVD: H = P^T @ Q, then R = V @ U^T
# Reflection handling: if det(R) < 0, flip last row of Vt

# ============================================
# POST-PROCESSING PARAMETERS
# ============================================

# Outlier removal (linear regression residual-based)
OUTLIER_THRESHOLD = 0.50    # Residual threshold for axis outlier removal
OUTLIER_PASSES = [0.5, 0.3, 0.3]  # Multiple passes with decreasing thresholds

# Angle outlier removal (z-score based)
ANGLE_OUTLIER_THRESHOLD = 0.5  # Z-score threshold for angle outliers

# Gaussian smoothing
SIGMA = 10                  # Sigma for scipy.ndimage.gaussian_filter1d

# Axis flipping threshold
Z_AXIS_FLIP_THRESHOLD = 0.25  # Threshold for sign-flipping based on z_axis

# Smoothing window for angle series
ANGLE_SMOOTH_WINDOW = 25    # Rolling window size for angle smoothing
ANGLE_SMOOTH_PASSES = 2     # Number of smoothing passes for angle

# ============================================
# VISUALIZATION PARAMETERS
# ============================================

# Sphere video output
SPHERE_DPI = 150            # DPI for rendered sphere frames
SPHERE_FIGURE_SIZE = 6      # Figure size (inches) for matplotlib
SPHERE_SCALING = 1.5        # Scale factor for sphere radius in visualization
SPHERE_BG_COLOR = (255/255, 219/255, 133/255)  # Light brown background

# Checkerboard pattern
NUM_LAT_BANDS = 1           # Number of latitude band divisions
NUM_LON_BANDS = 4           # Number of longitude band divisions
BAND_COLORS = [
    (0/255, 0/255, 0/255, 1.0),        # Black
    (255/255, 255/255, 255/255, 1.0),   # White
]
HIGHLIGHT_COLOR = (1.0, 0.0, 0.0, 1.0)  # Red for equator and poles

# Angular velocity arrow
ARROW_SCALE = 0.05          # Scaling factor for angular velocity arrow length
ARROW_ORIGIN_SCALE = 1.6    # How far from center the arrow starts

# ============================================
# OUTPUT CONFIGURATION
# ============================================

# Output folder structure:
# output/<video_name>/spin_analysis/
#   ├── rotation_data.csv          (raw detection output)
#   ├── rotation_data_processed.csv (after post-processing)
#   └── sphere_video.mp4           (3D visualization)

SAVE_RAW_CSV = True
SAVE_PROCESSED_CSV = True
SAVE_SPHERE_VIDEO = True

# ============================================
# DEBUG & LOGGING
# ============================================

DEBUG_MODE = False
VERBOSE = True
