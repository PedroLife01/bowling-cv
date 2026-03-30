"""
Configuration for real-time bowling analysis.
"""

import os

# ============================================
# PROJECT PATHS
# ============================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, 'assets', 'Template_lane_1.png')

# ============================================
# CAMERA
# ============================================
CAMERA_INDEX = 0          # 0 = default webcam, 1 = Continuity Camera (usually)
CAMERA_WIDTH = 1280       # Requested resolution width
CAMERA_HEIGHT = 720       # Requested resolution height
CAMERA_FPS = 30           # Requested FPS

# ============================================
# CALIBRATION (Lane Detection)
# ============================================
LANE_CALIBRATION_FRAMES = 100  # Frames to collect for lane boundary detection
CALIBRATION_TIMEOUT = 10  # Seconds before giving up on calibration

# ============================================
# BALL TRACKING (reuse Phase 2 defaults)
# ============================================
# MOG2 Background Subtraction
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 40
MOG2_DETECT_SHADOWS = True
SHADOW_THRESHOLD = 200

# Morphological filtering
MORPH_KERNEL_SIZE = 3
MORPH_KERNEL_SHAPE = 'ellipse'
USE_SHADOW_SEPARATION = True
SHADOW_SEPARATION_ITERATIONS = 2

# ROI / Tracking
B_MIN = 30
K_SCALE = 0.15
MIN_BALL_RADIUS = 5
MAX_BALL_RADIUS = 150
FOUL_LINE_PRIORITY_ZONE = 100
KALMAN_PROCESS_NOISE = 1.0
KALMAN_MEASUREMENT_NOISE = 10.0
CONFIRMATION_FRAMES = 20
CONFIRMATION_DISTANCE = 240
MAX_LOST_FRAMES = 2

# Blob Analysis (must match ball_detection attribute names)
CIRCULARITY_THRESHOLD = 0.70
ASPECT_RATIO_MAX = 1.8              # BlobAnalyzer expects ASPECT_RATIO_MAX
AUTO_CALIBRATE_AREA = True
AREA_MAX_AT_FOUL = 5000
AREA_MIN_AT_FOUL = 100
AREA_MAX_AT_PINS = 1500
AREA_MIN_AT_PINS = 30
CALIBRATION_FRAMES = 30
CALIBRATION_MIN_CIRCULARITY = 0.7
ENABLE_COLOR_FILTER = False
BALL_HSV_MIN = (100, 50, 50)
BALL_HSV_MAX = (130, 255, 255)
COLOR_MATCH_THRESHOLD = 0.7

# ROI / Tracking (additional attributes needed by BallTracker)
CONFIRMATION_THRESHOLD = 20         # BallTracker expects CONFIRMATION_THRESHOLD
SPATIAL_CONFIRMATION_DISTANCE = 240 # BallTracker expects SPATIAL_CONFIRMATION_DISTANCE
FOUL_LINE_EXCLUSION_FACTOR = 0.6
REACTIVATION_SEARCH_MARGIN = 20
REACTIVATION_TIMEOUT = 100
MIN_VELOCITY_Y = -2
SEARCH_BUFFER = 50

# Stop condition
ENABLE_STOP_CONDITION = True
STOP_THRESHOLD_PCT = -0.03
INTERPOLATE_TO_BOUNDARY = False
NUM_KALMAN_PREDICTIONS_AFTER_STOP = 5

# Masking
MASK_TOP_BOUNDARY = False

# ============================================
# PIN DETECTION
# ============================================
PIN_SETTLE_FRAMES = 45    # Wait this many frames after ball reaches pins
DIFFERENCE_THRESHOLD = 30
MIN_PIN_AREA = 150
MAX_PIN_AREA = 8000
MIN_PIN_ASPECT_RATIO = 0.2
MAX_PIN_ASPECT_RATIO = 1.5
MIN_PIN_SOLIDITY = 0.5
PIN_MORPH_KERNEL_SIZE = 5

# ============================================
# DISPLAY
# ============================================
WINDOW_NAME = 'Bowling CV - Real-time Analysis'
SHOW_TRAJECTORY_TRAIL = True
TRAIL_LENGTH = 50         # Number of past positions to draw
TRAIL_COLOR = (255, 0, 255)  # Magenta
BOUNDARY_COLOR = (0, 255, 0)  # Green
BALL_COLOR = (0, 0, 255)      # Red
TEXT_COLOR = (255, 255, 255)   # White

# ============================================
# SESSION
# ============================================
SAVE_SESSION_DATA = True
SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'realtime_sessions')

# ============================================
# GENERAL
# ============================================
VERBOSE = True
DEBUG_MODE = False
