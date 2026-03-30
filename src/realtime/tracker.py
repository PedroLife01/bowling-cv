"""
Real-time ball tracker.

Wraps the existing MOG2 + Kalman filter pipeline for frame-by-frame use
with a live camera stream.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from ball_detection.blob_analysis import BlobAnalyzer
from ball_detection.roi_logic import BallTracker


class RealtimeTracker:
    """
    Frame-by-frame ball tracker for real-time use.

    Manages MOG2 background model and BallTracker state,
    processing one frame at a time from a live stream.
    """

    def __init__(self, config, frame_width, frame_height, foul_line_y, top_boundary_y=None):
        """
        Args:
            config: Configuration module
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            foul_line_y: Y-coordinate of the foul line
            top_boundary_y: Y-coordinate of the top boundary (optional)
        """
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.foul_line_y = foul_line_y
        self.top_boundary_y = top_boundary_y

        # MOG2 background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=getattr(config, 'MOG2_DETECT_SHADOWS', True)
        )

        # Morphological kernel
        ksize = getattr(config, 'MORPH_KERNEL_SIZE', 3)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # Shadow separation kernel
        self.use_shadow_sep = getattr(config, 'USE_SHADOW_SEPARATION', True)
        self.shadow_sep_iters = getattr(config, 'SHADOW_SEPARATION_ITERATIONS', 2)

        # Ball tracker (Stage C+D+E)
        self.analyzer = BlobAnalyzer(config)
        self.tracker = BallTracker(config, frame_width, frame_height, foul_line_y,
                                   blob_analyzer=self.analyzer)
        if top_boundary_y is not None:
            self.tracker.set_boundaries(top_boundary_y)

        # State
        self.frame_idx = 0
        self.trajectory = deque(maxlen=500)
        self.throw_complete = False
        self._calibrated = False

    def process_frame(self, masked_frame):
        """
        Process a single masked frame.

        Args:
            masked_frame: BGR frame with lane mask applied

        Returns:
            dict with keys:
                - detection: {center, radius} or None
                - prediction: {x, y, vx, vy} or None
                - mode: 'global' or 'local'
                - trajectory: list of (x, y) recent positions
                - throw_complete: bool
                - frame_idx: int
        """
        # Stage B: Motion detection
        fg_mask = self.bg_subtractor.apply(masked_frame)

        # Shadow removal
        shadow_thresh = getattr(self.config, 'SHADOW_THRESHOLD', 200)
        _, clean_mask = cv2.threshold(fg_mask, shadow_thresh, 255, cv2.THRESH_BINARY)

        # Morphological opening (noise removal)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, self.kernel)

        # Shadow separation
        if self.use_shadow_sep:
            clean_mask = cv2.erode(clean_mask, self.kernel, iterations=self.shadow_sep_iters)

        # Auto-calibrate blob analyzer
        if not self.analyzer.is_calibrated:
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.analyzer.auto_calibrate(masked_frame, contours, self.frame_height, self.foul_line_y)

        # Stage C+D+E: Integrated tracking
        result = self.tracker.process_frame(clean_mask, self.frame_idx, masked_frame)

        # Update trajectory
        if result.get('detection'):
            pos = result['detection']['center']
            self.trajectory.append(pos)

        # Check if throw is complete
        if result.get('trajectory_complete', False):
            self.throw_complete = True

        result['trajectory'] = list(self.trajectory)
        result['throw_complete'] = self.throw_complete
        result['frame_idx'] = self.frame_idx
        result['mask'] = clean_mask

        self.frame_idx += 1
        return result

    def reset(self):
        """Reset tracker for a new throw."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.MOG2_HISTORY,
            varThreshold=self.config.MOG2_VAR_THRESHOLD,
            detectShadows=getattr(self.config, 'MOG2_DETECT_SHADOWS', True)
        )
        self.analyzer = BlobAnalyzer(self.config)
        self.tracker = BallTracker(
            self.config, self.frame_width, self.frame_height,
            self.foul_line_y, blob_analyzer=self.analyzer
        )
        if self.top_boundary_y is not None:
            self.tracker.set_boundaries(self.top_boundary_y)

        self.trajectory.clear()
        self.throw_complete = False
        self.frame_idx = 0
