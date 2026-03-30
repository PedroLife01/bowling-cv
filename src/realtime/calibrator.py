"""
Lane calibration for real-time mode.

Uses the existing LaneDetector to calibrate boundaries from collected frames,
then provides a mask for real-time frame processing.
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lane_detection.detection_functions import detect_horizontal_line, detect_vertical_boundaries_approach1
from lane_detection.master_line_computation import compute_master_line_from_collection
from lane_detection import config as lane_config


class LaneCalibrator:
    """
    Calibrates lane boundaries from live camera frames.

    Collects frames, detects boundaries using the same algorithms as Phase 1,
    then provides a mask for real-time processing.
    """

    def __init__(self, config=None):
        self.config = config or lane_config
        self.calibrated = False
        self.boundaries = None
        self.mask = None
        self._collected_frames = []

    def add_frame(self, frame):
        """
        Add a frame for calibration.

        Returns:
            bool: True if calibration is complete
        """
        if self.calibrated:
            return True

        self._collected_frames.append(frame.copy())

        target = getattr(self.config, 'NUM_COLLECTION_FRAMES', 100)
        if len(self._collected_frames) >= target:
            return self._run_calibration()

        return False

    @property
    def progress(self):
        """Return calibration progress as fraction 0.0-1.0."""
        if self.calibrated:
            return 1.0
        target = getattr(self.config, 'NUM_COLLECTION_FRAMES', 100)
        return len(self._collected_frames) / target

    def _run_calibration(self):
        """Run lane detection on collected frames using Phase 1 algorithms."""
        print("\n[Calibrator] Running lane detection on collected frames...")
        frames = self._collected_frames
        height, width = frames[0].shape[:2]

        # --- Step 1: Detect foul line in each frame ---
        foul_params_list = []
        for frame in frames:
            _, _, foul_params = detect_horizontal_line(frame)
            if foul_params is not None:
                foul_params_list.append(foul_params)

        if not foul_params_list:
            print("[Calibrator] ERROR: Could not detect foul line in any frame")
            self._collected_frames = []
            return False

        # Compute median foul line parameters
        median_foul_params = {
            'center_y': int(np.median([fp['center_y'] for fp in foul_params_list])),
            'slope': float(np.median([fp['slope'] for fp in foul_params_list])),
            'center_x': foul_params_list[0]['center_x'],
            'width': foul_params_list[0]['width'],
            'height': foul_params_list[0]['height'],
        }

        foul_y = median_foul_params['center_y']
        print(f"  Foul line detected: Y={foul_y}")

        # --- Step 2: Detect side boundaries in each frame ---
        angle_mode = 'from_vertical' if getattr(self.config, 'USE_ABSOLUTE_ANGLES', True) else 'from_horizontal'

        left_lines = []
        right_lines = []
        for frame in frames:
            result = detect_vertical_boundaries_approach1(frame, median_foul_params, angle_mode)
            if result is not None:
                _, _, _, left_lines_frame, right_lines_frame = result
                if left_lines_frame:
                    left_lines.extend(left_lines_frame)
                if right_lines_frame:
                    right_lines.extend(right_lines_frame)

        if not left_lines or not right_lines:
            print("[Calibrator] ERROR: Could not detect side boundaries")
            self._collected_frames = []
            return False

        # Compute master side lines using voting system
        bin_width = getattr(self.config, 'BIN_WIDTH', 10)
        vote_thresh = getattr(self.config, 'VOTE_THRESHOLD', 0.15)
        angle_tol = getattr(self.config, 'ANGLE_TOLERANCE', 3)

        left_result, _ = compute_master_line_from_collection(
            left_lines, median_foul_params,
            bin_width=bin_width, vote_threshold=vote_thresh,
            angle_tolerance=angle_tol, side='left', angle_mode=angle_mode
        )
        right_result, _ = compute_master_line_from_collection(
            right_lines, median_foul_params,
            bin_width=bin_width, vote_threshold=vote_thresh,
            angle_tolerance=angle_tol, side='right', angle_mode=angle_mode
        )

        if left_result is None or right_result is None:
            print("[Calibrator] ERROR: Voting system failed for side boundaries")
            self._collected_frames = []
            return False

        left_x = int(left_result['x_intersect'])
        right_x = int(right_result['x_intersect'])
        print(f"  Side boundaries: Left X={left_x}, Right X={right_x}")

        # --- Step 3: Detect top boundary ---
        # The Phase 1 top boundary detector expects a video file path, not frames.
        # We write frames to a temp video, detect, then clean up.
        top_y = None
        try:
            from lane_detection.top_boundary_detection import detect_top_boundary_all_frames

            import tempfile
            tmp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tmp_path = tmp_video.name
            tmp_video.close()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp_path, fourcc, 30.0, (width, height))
            for f in frames[:50]:  # Use subset for speed
                writer.write(f)
            writer.release()

            top_data = detect_top_boundary_all_frames(tmp_path, self.config)
            os.unlink(tmp_path)

            if top_data and len(top_data) > 0:
                y_values = [d['y'] for d in top_data if d is not None and d.get('y') is not None]
                if y_values:
                    top_y = int(np.median(y_values))
                    print(f"  Top boundary: Y={top_y}")
        except Exception as e:
            print(f"  Warning: Top boundary detection failed: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if top_y is None:
            # Estimate top boundary as 15% from top of frame
            top_y = int(height * 0.15)
            print(f"  Top boundary estimated: Y={top_y}")

        # --- Build boundary data (same format as Phase 1 boundary_data.json) ---
        self.boundaries = {
            'foul_line_y': foul_y,
            'left_x': left_x,
            'right_x': right_x,
            'top_y': top_y,
            'frame_width': width,
            'frame_height': height,
            # Phase 1 compatible fields
            'median_foul_params': {'center_y': foul_y},
            'top_boundary': {'y_position': top_y} if top_y else {},
            'master_left': left_result,
            'master_right': right_result,
        }

        # --- Create lane mask ---
        self.mask = np.zeros((height, width), dtype=np.uint8)

        # Use the actual boundary lines for a more accurate mask
        # Left boundary: from (left_x, foul_y) to top
        # Right boundary: from (right_x, foul_y) to top
        pts = np.array([
            [left_x, foul_y],
            [right_x, foul_y],
            [right_x, top_y],
            [left_x, top_y],
        ])
        cv2.fillPoly(self.mask, [pts], 255)

        self.calibrated = True
        self._collected_frames = []  # Free memory

        print(f"[Calibrator] Calibration complete!")
        print(f"  Lane: {left_x}-{right_x} x {top_y}-{foul_y}")
        print(f"  Lane width: {right_x - left_x}px, Lane height: {foul_y - top_y}px")

        return True

    def apply_mask(self, frame):
        """Apply lane mask to a frame. Returns masked BGR frame."""
        if self.mask is None:
            return frame
        return cv2.bitwise_and(frame, frame, mask=self.mask)

    def draw_boundaries(self, frame):
        """Draw lane boundaries on a frame for visualization."""
        if self.boundaries is None:
            return frame

        b = self.boundaries

        # Foul line (red)
        cv2.line(frame, (0, b['foul_line_y']), (b['frame_width'], b['foul_line_y']),
                 (0, 0, 255), 2)

        # Side lines (blue)
        cv2.line(frame, (b['left_x'], 0), (b['left_x'], b['frame_height']),
                 (255, 0, 0), 2)
        cv2.line(frame, (b['right_x'], 0), (b['right_x'], b['frame_height']),
                 (255, 0, 0), 2)

        # Top boundary (green)
        if b.get('top_y'):
            cv2.line(frame, (0, b['top_y']), (b['frame_width'], b['top_y']),
                     (0, 255, 0), 2)

        return frame

    def save_boundaries(self, path):
        """Save boundary data to JSON file."""
        if self.boundaries is None:
            return

        # Make JSON-serializable copy
        data = {}
        for k, v in self.boundaries.items():
            if isinstance(v, dict):
                data[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                           for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                data[k] = float(v)
            else:
                data[k] = v

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Calibrator] Boundaries saved to {path}")
