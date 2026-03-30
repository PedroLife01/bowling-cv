"""
Real-time display overlay.

Draws lane boundaries, ball trajectory, pin count, and stats
on the live camera feed.
"""

import cv2
import numpy as np
import time


class RealtimeDisplay:
    """Draws analysis overlay on live frames."""

    def __init__(self, config):
        self.config = config
        self._fps_times = []
        self._throw_count = 0
        self._pin_counts = []
        self._last_pin_count = None

    def draw_overlay(self, frame, calibrator, tracker_result=None, state='calibrating'):
        """
        Draw complete overlay on frame.

        Args:
            frame: BGR frame (modified in-place)
            calibrator: LaneCalibrator instance
            tracker_result: Result from RealtimeTracker.process_frame()
            state: 'calibrating', 'waiting', 'tracking', 'pin_count'

        Returns:
            Frame with overlay
        """
        # FPS calculation
        now = time.time()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        fps = len(self._fps_times)

        # Draw lane boundaries
        if calibrator.calibrated:
            calibrator.draw_boundaries(frame)

        # Draw based on state
        if state == 'calibrating':
            self._draw_calibration(frame, calibrator)
        elif state == 'waiting':
            self._draw_waiting(frame)
        elif state == 'tracking' and tracker_result:
            self._draw_tracking(frame, tracker_result)
        elif state == 'pin_count':
            self._draw_pin_count(frame)

        # HUD
        self._draw_hud(frame, fps, state)

        return frame

    def _draw_calibration(self, frame, calibrator):
        """Draw calibration progress."""
        h, w = frame.shape[:2]
        progress = calibrator.progress

        # Progress bar
        bar_w = 300
        bar_h = 20
        x = (w - bar_w) // 2
        y = h // 2

        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + int(bar_w * progress), y + bar_h),
                      (0, 255, 0), -1)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 1)

        text = f"Calibrating lane... {progress*100:.0f}%"
        self._put_text_centered(frame, text, x + bar_w // 2, y - 10)

    def _draw_waiting(self, frame):
        """Draw waiting-for-throw indicator."""
        h, w = frame.shape[:2]
        self._put_text_centered(frame, "Waiting for throw...", w // 2, h - 40,
                                color=(0, 255, 255))

    def _draw_tracking(self, frame, result):
        """Draw ball tracking visualization."""
        trajectory = result.get('trajectory', [])
        detection = result.get('detection')

        # Draw trajectory trail
        if len(trajectory) > 1:
            trail_color = self.config.TRAIL_COLOR
            max_len = self.config.TRAIL_LENGTH
            points = trajectory[-max_len:]

            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(frame, pt1, pt2, trail_color, thickness)

        # Draw current detection
        if detection:
            cx, cy = int(detection['center'][0]), int(detection['center'][1])
            radius = int(detection.get('radius', 10))
            cv2.circle(frame, (cx, cy), radius, self.config.BALL_COLOR, 2)
            cv2.circle(frame, (cx, cy), 3, self.config.BALL_COLOR, -1)

        # Draw ROI box
        roi_box = result.get('roi_box')
        if roi_box:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Mode indicator
        mode = result.get('mode', 'unknown')
        h = frame.shape[0]
        cv2.putText(frame, f"Mode: {mode}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _draw_pin_count(self, frame):
        """Draw pin count result."""
        if self._last_pin_count is not None:
            h, w = frame.shape[:2]
            text = f"Pins down: {self._last_pin_count}"
            self._put_text_centered(frame, text, w // 2, h // 2,
                                    scale=1.5, color=(0, 255, 0))

    def _draw_hud(self, frame, fps, state):
        """Draw heads-up display (FPS, throw count, stats)."""
        # FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # State
        state_text = {
            'calibrating': 'CALIBRATING',
            'waiting': 'READY',
            'tracking': 'TRACKING',
            'pin_count': 'PIN COUNT',
        }.get(state, state.upper())

        cv2.putText(frame, state_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Throw count and average
        if self._throw_count > 0:
            cv2.putText(frame, f"Throws: {self._throw_count}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if self._pin_counts:
                avg = sum(self._pin_counts) / len(self._pin_counts)
                cv2.putText(frame, f"Avg pins: {avg:.1f}", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls help
        h = frame.shape[0]
        cv2.putText(frame, "Q=quit  R=reset  SPACE=new throw", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def record_throw(self, pins_down):
        """Record a completed throw."""
        self._throw_count += 1
        self._pin_counts.append(pins_down)
        self._last_pin_count = pins_down

    def _put_text_centered(self, frame, text, cx, cy, scale=0.7, color=(255, 255, 255)):
        """Put text centered at given position."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        x = cx - tw // 2
        y = cy + th // 2
        # Background
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, scale, color, 1)
