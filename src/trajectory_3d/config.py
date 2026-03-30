"""
Configuration file for trajectory 3D reconstruction - Phase 3
Edit parameters here without touching the main code

Version: 1.0.0
Authors: Pedro Roriz
Created: March 30, 2026
"""

import os

# ============================================
# PROJECT PATHS
# ============================================

# Base paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# List of videos to process (same as Phase 1 & 2)
VIDEO_FILES = [
    'cropped_test3.mp4',
    'cropped_test6.mp4',
    'cropped_test7.mp4',
    'cropped_test9.mp4',
    'cropped_test10.mp4'
]

# ============================================
# LANE DIMENSIONS (inches)
# ============================================

# Official bowling lane dimensions
LANE_WIDTH = 106    # 41.5 inches gutter-to-gutter, but using full width with gutters
LANE_LENGTH = 1829  # ~60 feet from foul line to pin deck (in centimeters, kept from reference)

# ============================================
# HOMOGRAPHY PARAMETERS
# ============================================

# Destination rectangle in lane coordinates
# Maps to: (0, 0) top-left to (LANE_WIDTH, LANE_LENGTH) bottom-left
# Convention: top of lane = pins (y=0), bottom = foul line (y=LANE_LENGTH)

# ============================================
# SMOOTHING PARAMETERS
# ============================================

# Savitzky-Golay filter for trajectory smoothing
SAVGOL_WINDOW = 61       # Window length (must be odd, larger = smoother)
SAVGOL_POLYORDER = 2     # Polynomial order (2 = quadratic fit)

# ============================================
# INPUT FILE NAMES (relative to output/<video_name>/)
# ============================================

# Boundary data from Phase 1 (lane detection)
BOUNDARY_DATA_FILENAME = 'boundary_data.json'

# Ball trajectory from Phase 2 (ball detection)
BALL_TRAJECTORY_FILENAME = os.path.join('ball_detection', 'trajectory_processed_original.csv')

# ============================================
# OUTPUT CONFIGURATION
# ============================================

# Output subfolder name within output/<video_name>/
OUTPUT_SUBFOLDER = 'trajectory_3d'

# Save intermediate lane corners CSV (for debugging)
SAVE_LANE_CORNERS_CSV = True

# Save transformed positions CSV
SAVE_TRANSFORMED_CSV = True

# Save smoothed trajectory CSV
SAVE_SMOOTHED_CSV = True

# ============================================
# VISUALIZATION PARAMETERS
# ============================================

# Generate overhead trajectory visualization video
SAVE_VISUALIZATION_VIDEO = True

# Video settings
VIS_FPS = 30
VIS_LANE_COLOR = (200, 200, 200)       # Light grey lane background (BGR)
VIS_GUTTER_COLOR = (120, 120, 120)     # Dark grey gutters (BGR)
VIS_TRAJECTORY_COLOR = (255, 0, 255)   # Magenta trajectory line (BGR)
VIS_BALL_COLOR = (0, 255, 255)         # Yellow current ball position (BGR)
VIS_LINE_WIDTH = 2
VIS_BALL_RADIUS = 8
VIS_TRAIL_ALPHA = 0.6                  # Opacity for trajectory trail

# Canvas dimensions for visualization (pixels)
VIS_CANVAS_WIDTH = 400     # Width of the overhead view canvas
VIS_CANVAS_HEIGHT = 1200   # Height of the overhead view canvas

# ============================================
# DEBUG OPTIONS
# ============================================

DEBUG_MODE = False
VERBOSE = True
