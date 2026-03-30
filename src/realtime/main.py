"""
Real-time Bowling Analysis - Main Entry Point

Runs a live camera session that:
1. Calibrates lane boundaries from first ~100 frames
2. Tracks ball trajectory in real-time using MOG2 + Kalman filter
3. Detects pin count after each throw
4. Displays overlay with trajectory, boundaries, and stats

Usage:
    python -m src.realtime.main                     # Default camera (index 0)
    python -m src.realtime.main --camera 1          # Continuity Camera (try 0 or 1)
    python -m src.realtime.main --camera "rtsp://..." # IP camera

Controls:
    Q      - Quit
    R      - Recalibrate lane
    SPACE  - Reset for new throw
    S      - Save current session data
"""

import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime import config
from realtime.stream_capture import StreamCapture
from realtime.calibrator import LaneCalibrator
from realtime.tracker import RealtimeTracker
from realtime.display import RealtimeDisplay


def list_cameras():
    """Try to detect available cameras."""
    print("\nDetecting cameras...")
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            name = cap.getBackendName()
            available.append(i)
            print(f"  Camera {i}: {w}x{h} @ {fps:.0f}fps ({name})")
            cap.release()
        else:
            cap.release()

    if not available:
        print("  No cameras found!")
    return available


def run_session(camera_source=0, camera_width=1280, camera_height=720):
    """
    Run a real-time bowling analysis session.

    Args:
        camera_source: Camera index or stream URL
        camera_width: Requested camera width
        camera_height: Requested camera height
    """
    print("\n" + "=" * 60)
    print("BOWLING CV - REAL-TIME ANALYSIS")
    print("=" * 60)

    # Initialize components
    display = RealtimeDisplay(config)
    calibrator = LaneCalibrator()

    # State machine
    state = 'calibrating'
    tracker = None
    before_frame = None  # Frame before throw (for pin detection)
    pin_settle_start = None

    # Open camera
    print(f"\nOpening camera source: {camera_source}")
    try:
        stream = StreamCapture(
            source=camera_source,
            width=camera_width,
            height=camera_height,
            fps=config.CAMERA_FPS
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nAvailable cameras:")
        list_cameras()
        return

    with stream:
        print(f"Camera opened: {stream}")
        print(f"\nControls: Q=quit, R=recalibrate, SPACE=new throw, S=save\n")

        while True:
            frame_idx, frame = stream.read()
            if frame is None:
                time.sleep(0.001)
                continue

            # === STATE: CALIBRATING ===
            if state == 'calibrating':
                done = calibrator.add_frame(frame)
                vis_frame = frame.copy()
                display.draw_overlay(vis_frame, calibrator, state='calibrating')

                if done:
                    print("[Session] Calibration complete! Waiting for throw...")
                    b = calibrator.boundaries
                    tracker = RealtimeTracker(
                        config,
                        b['frame_width'], b['frame_height'],
                        b['foul_line_y'],
                        top_boundary_y=b.get('top_y')
                    )
                    # Capture "before" frame for pin detection
                    before_frame = frame.copy()
                    state = 'waiting'

            # === STATE: WAITING FOR THROW ===
            elif state == 'waiting':
                masked = calibrator.apply_mask(frame)
                result = tracker.process_frame(masked)

                vis_frame = frame.copy()

                # If we detect ball movement, switch to tracking
                if result.get('detection') is not None:
                    state = 'tracking'
                    print("[Session] Ball detected! Tracking...")

                display.draw_overlay(vis_frame, calibrator, result, state='waiting')

            # === STATE: TRACKING ===
            elif state == 'tracking':
                masked = calibrator.apply_mask(frame)
                result = tracker.process_frame(masked)

                vis_frame = frame.copy()
                display.draw_overlay(vis_frame, calibrator, result, state='tracking')

                # Check if throw is complete
                if result.get('throw_complete', False):
                    print(f"[Session] Throw complete at frame {result['frame_idx']}")
                    pin_settle_start = time.time()
                    state = 'pin_settle'

            # === STATE: WAITING FOR PINS TO SETTLE ===
            elif state == 'pin_settle':
                vis_frame = frame.copy()
                elapsed = time.time() - pin_settle_start
                settle_time = config.PIN_SETTLE_FRAMES / config.CAMERA_FPS

                display.draw_overlay(vis_frame, calibrator, state='tracking')

                # Draw settle progress
                cv2.putText(vis_frame, f"Pins settling... {elapsed:.1f}s / {settle_time:.1f}s",
                            (10, vis_frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                if elapsed >= settle_time:
                    # Simple pin detection: frame differencing
                    pins = _detect_pins_simple(before_frame, frame, calibrator)
                    display.record_throw(pins)
                    print(f"[Session] Pins down: {pins}")
                    state = 'pin_count'
                    pin_count_start = time.time()

            # === STATE: SHOWING PIN COUNT ===
            elif state == 'pin_count':
                vis_frame = frame.copy()
                display.draw_overlay(vis_frame, calibrator, state='pin_count')

                # Show result for 3 seconds then reset
                if time.time() - pin_count_start > 3.0:
                    tracker.reset()
                    before_frame = frame.copy()
                    state = 'waiting'
                    print("[Session] Ready for next throw...")

            # Show frame
            cv2.imshow(config.WINDOW_NAME, vis_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Recalibrate
                calibrator = LaneCalibrator()
                state = 'calibrating'
                print("[Session] Recalibrating...")
            elif key == ord(' '):
                # Reset for new throw
                if tracker:
                    tracker.reset()
                before_frame = frame.copy()
                state = 'waiting'
                print("[Session] Reset - waiting for throw...")
            elif key == ord('s'):
                _save_session(display, calibrator)

    cv2.destroyAllWindows()
    print("\n[Session] Session ended.")


def _detect_pins_simple(before_frame, after_frame, calibrator):
    """
    Simple pin detection via frame differencing.

    Args:
        before_frame: Frame before throw
        after_frame: Frame after throw
        calibrator: LaneCalibrator with boundary data

    Returns:
        int: Estimated number of pins knocked down
    """
    if before_frame is None or calibrator.boundaries is None:
        return 0

    b = calibrator.boundaries
    top_y = b.get('top_y', 0) or 0

    # Crop to pin area (top portion of lane)
    pin_region_h = max(1, int((b['foul_line_y'] - top_y) * 0.2))
    y1 = max(0, top_y - 20)
    y2 = top_y + pin_region_h
    x1 = b['left_x']
    x2 = b['right_x']

    if y2 <= y1 or x2 <= x1:
        return 0

    before_crop = before_frame[y1:y2, x1:x2]
    after_crop = after_frame[y1:y2, x1:x2]

    # Convert to grayscale and compute difference
    gray_before = cv2.cvtColor(before_crop, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_before, gray_after)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Count contours as proxy for pins
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pin_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if config.MIN_PIN_AREA <= area <= config.MAX_PIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / max(w, 1)
            if config.MIN_PIN_ASPECT_RATIO <= aspect <= config.MAX_PIN_ASPECT_RATIO:
                pin_count += 1

    return min(pin_count, 10)  # Cap at 10


def _save_session(display, calibrator):
    """Save session data to disk."""
    import json
    import os

    session_dir = config.SESSION_OUTPUT_DIR
    os.makedirs(session_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = os.path.join(session_dir, f"session_{timestamp}.json")

    data = {
        'timestamp': timestamp,
        'throws': display._throw_count,
        'pin_counts': display._pin_counts,
        'average_pins': sum(display._pin_counts) / max(len(display._pin_counts), 1),
        'boundaries': calibrator.boundaries if calibrator.calibrated else None,
    }

    with open(session_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[Session] Saved to {session_file}")


def main():
    parser = argparse.ArgumentParser(description='Real-time Bowling CV Analysis')
    parser.add_argument('--camera', default=0,
                        help='Camera index (0, 1) or stream URL')
    parser.add_argument('--width', type=int, default=config.CAMERA_WIDTH,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=config.CAMERA_HEIGHT,
                        help='Camera height')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available cameras and exit')
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    # Parse camera source
    try:
        camera_source = int(args.camera)
    except ValueError:
        camera_source = args.camera  # URL string

    run_session(camera_source, args.width, args.height)


if __name__ == '__main__':
    main()
