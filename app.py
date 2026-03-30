"""
Streamlit web interface for Bowling CV Analysis Pipeline.

Two modes:
  1. Recorded Video: Upload a bowling video and run each phase interactively
  2. Real-Time: Connect a camera (iPhone via Continuity Camera, USB webcam)
     for live lane calibration, ball tracking, and pin detection

Usage:
    streamlit run app.py
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_INPUT = PROJECT_ROOT / "assets" / "input"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _save_uploaded_video(uploaded) -> Path:
    """Save uploaded file to assets/input/ and return the path."""
    ASSETS_INPUT.mkdir(parents=True, exist_ok=True)
    dest = ASSETS_INPUT / uploaded.name
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    return dest


def _reencode_for_web(input_mp4: str) -> str:
    """Re-encode mp4v video to H.264 baseline for browser playback."""
    output_mp4 = input_mp4.replace(".mp4", "_web.mp4")
    if os.path.exists(output_mp4):
        return output_mp4
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_mp4,
                "-c:v", "libx264", "-profile:v", "baseline",
                "-level", "3.0", "-movflags", "+faststart",
                output_mp4,
            ],
            check=True,
            capture_output=True,
        )
        return output_mp4
    except (subprocess.CalledProcessError, FileNotFoundError):
        return input_mp4


def _display_video(path: str, max_height: int = 400):
    """Display a local video file with CSS max-height."""
    if not os.path.exists(path):
        st.warning(f"Video not found: {path}")
        return
    video_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(video_bytes).decode()
    html = f"""
    <video controls style="max-height:{max_height}px; width:auto; display:block; margin:auto;">
      <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    </video>
    """
    st.markdown(html, unsafe_allow_html=True)


def _run_module(module_args: list, label: str) -> bool:
    """Run a python module as subprocess and show output."""
    cmd = [sys.executable, "-m"] + module_args
    with st.spinner(f"Running {label}..."):
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
    if result.returncode != 0:
        st.error(f"{label} failed (exit code {result.returncode})")
        if result.stderr:
            st.code(result.stderr[-2000:], language="text")
        return False
    if result.stdout:
        with st.expander(f"{label} output", expanded=False):
            st.code(result.stdout[-3000:], language="text")
    return True


def _find_output_files(video_name: str, subfolder: str, pattern: str) -> list:
    """Find files matching a glob pattern in an output subfolder."""
    search_dir = OUTPUT_DIR / video_name / subfolder
    if not search_dir.exists():
        return []
    return sorted(search_dir.glob(pattern))


# ─── Streamlit App ────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Bowling CV Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Bowling CV Analysis Pipeline")

    tab_recorded, tab_realtime = st.tabs(["Recorded Video", "Real-Time Camera"])

    with tab_recorded:
        _page_recorded_video()

    with tab_realtime:
        _page_realtime_camera()


# ==============================================================================
#                          RECORDED VIDEO PAGE
# ==============================================================================


def _page_recorded_video():
    """Recorded video analysis with 5 phases."""

    for key in ("video_path", "video_name"):
        st.session_state.setdefault(key, None)

    # Upload
    uploaded = st.file_uploader("Upload bowling video", type=["mp4", "avi", "mov"])
    if uploaded:
        path = _save_uploaded_video(uploaded)
        st.session_state.video_path = str(path)
        st.session_state.video_name = path.stem
        st.success(f"Saved: {path.name}")

    if not st.session_state.video_path:
        st.info("Upload a bowling video to get started.")
        return

    video_path = st.session_state.video_path
    video_name = st.session_state.video_name
    video_file = Path(video_path).name

    # Pipeline status
    out = OUTPUT_DIR / video_name
    with st.sidebar:
        st.subheader("Pipeline Status")
        phases = {
            "Phase 1 - Lane": (out / "boundary_data.json").exists(),
            "Phase 2 - Ball": (out / "ball_detection").exists(),
            "Phase 3 - Trajectory": (out / "trajectory_3d").exists(),
            "Phase 4 - Pins": (out / "pin_detection").exists(),
            "Phase 5 - Spin": (out / "spin_analysis").exists(),
        }
        for name, done in phases.items():
            st.write(f"{'[x]' if done else '[ ]'} {name}")

    # ── Phase 1 ──
    with st.expander("Phase 1: Lane Detection", expanded=True):
        st.write("Detects lane boundaries (foul line, left/right gutters, pin area).")
        if st.button("Run Phase 1", key="run_p1"):
            _run_module(
                ["src.lane_detection.main", "--video", video_file],
                "Phase 1 - Lane Detection",
            )
            st.rerun()

        boundary_json = OUTPUT_DIR / video_name / "boundary_data.json"
        if boundary_json.exists():
            st.success("Lane boundaries detected.")
            with open(boundary_json) as f:
                data = json.load(f)
            with st.expander("Boundary Data", expanded=False):
                st.json(data)

    # ── Phase 2 ──
    with st.expander("Phase 2: Ball Detection"):
        st.write("Tracks the bowling ball from foul line to pins using MOG2 + Kalman filter.")
        if st.button("Run Phase 2", key="run_p2"):
            _run_module(
                ["src.ball_detection.main", "--video", video_file],
                "Phase 2 - Ball Detection",
            )
            st.rerun()

        ball_dir = OUTPUT_DIR / video_name / "ball_detection"
        if ball_dir.exists():
            st.success("Ball detection complete.")
            traj_csv = ball_dir / "trajectory_processed_original.csv"
            if traj_csv.exists():
                import pandas as pd
                df = pd.read_csv(traj_csv)
                st.write(f"Trajectory: {len(df)} frames, {df['x'].notna().sum()} valid detections")
                with st.expander("Trajectory Data", expanded=False):
                    st.dataframe(df.head(50))
            videos = list(ball_dir.glob("*overlay*.mp4")) + list(ball_dir.glob("*Stage_H*.mp4"))
            for v in videos[:1]:
                web = _reencode_for_web(str(v))
                _display_video(web)

    # ── Phase 3 ──
    with st.expander("Phase 3: Trajectory Reconstruction"):
        st.write("Maps ball positions to real-world lane coordinates via homography.")
        if st.button("Run Phase 3", key="run_p3"):
            _run_module(
                ["src.trajectory_3d.main", "--video", video_file],
                "Phase 3 - Trajectory Reconstruction",
            )
            st.rerun()

        traj_dir = OUTPUT_DIR / video_name / "trajectory_3d"
        if traj_dir.exists():
            st.success("Trajectory reconstruction complete.")
            smoothed_csv = traj_dir / "transformed_positions_smoothed.csv"
            if smoothed_csv.exists():
                import pandas as pd
                df = pd.read_csv(smoothed_csv)
                valid = df.dropna(subset=["x", "y"])
                st.write(f"Transformed: {len(valid)} valid points")
                if len(valid) > 0:
                    st.write(f"X range: {valid['x'].min():.1f} - {valid['x'].max():.1f}")
                    st.write(f"Y range: {valid['y'].min():.1f} - {valid['y'].max():.1f}")
                with st.expander("Transformed Data", expanded=False):
                    st.dataframe(df.head(50))
            overhead_vids = list(traj_dir.glob("*overhead*.mp4"))
            for v in overhead_vids[:1]:
                web = _reencode_for_web(str(v))
                _display_video(web, max_height=600)

    # ── Phase 4 ──
    with st.expander("Phase 4: Pin Detection"):
        st.write("Counts toppled pins using frame differencing at impact moment.")
        if st.button("Run Phase 4", key="run_p4"):
            _run_module(
                ["src.pin_detection.main", "--video", video_file],
                "Phase 4 - Pin Detection",
            )
            st.rerun()

        pin_dir = OUTPUT_DIR / video_name / "pin_detection"
        if pin_dir.exists():
            st.success("Pin detection complete.")
            pin_results = list(pin_dir.glob("*.json"))
            if pin_results:
                with open(pin_results[0]) as f:
                    pins = json.load(f)
                toppled = pins.get("toppled_pins", "?")
                result = pins.get("result", "?")
                st.metric("Toppled Pins", f"{toppled}/10", delta=result)

    # ── Phase 5 ──
    with st.expander("Phase 5: Spin Analysis"):
        st.write("Analyzes ball rotation via optical flow + Kabsch algorithm.")
        if st.button("Run Phase 5", key="run_p5"):
            _run_module(
                ["src.spin_analysis.main", "--video", video_path],
                "Phase 5 - Spin Analysis",
            )
            st.rerun()

        spin_dir = OUTPUT_DIR / video_name / "spin_analysis"
        if spin_dir.exists():
            st.success("Spin analysis complete.")
            processed_csv = spin_dir / "rotation_data_processed.csv"
            if processed_csv.exists():
                import pandas as pd
                df = pd.read_csv(processed_csv)
                valid = df.dropna(subset=["x_axis"])
                st.write(f"Rotation data: {len(valid)} valid frames")
                if len(valid) > 0 and "angle" in df.columns:
                    mean_angle = valid["angle"].mean()
                    st.metric("Avg Angular Velocity", f"{mean_angle * 30:.2f} rad/s")
                with st.expander("Rotation Data", expanded=False):
                    st.dataframe(df.head(50))
            sphere_vids = list(spin_dir.glob("*sphere*.mp4"))
            for v in sphere_vids[:1]:
                web = _reencode_for_web(str(v))
                _display_video(web)


# ==============================================================================
#                          REAL-TIME CAMERA PAGE
# ==============================================================================


def _detect_cameras() -> list:
    """Probe camera indices 0-4 and return available ones."""
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append({"index": i, "resolution": f"{w}x{h}"})
            cap.release()
        else:
            cap.release()
    return available


def _page_realtime_camera():
    """Real-time camera analysis page."""

    # Init session state for realtime
    for key in (
        "rt_running", "rt_camera_idx", "rt_state", "rt_calibrator",
        "rt_tracker", "rt_throws", "rt_before_frame", "rt_pin_settle_start",
    ):
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("rt_running", False)
    st.session_state.setdefault("rt_throws", [])

    st.subheader("Real-Time Bowling Analysis")
    st.write(
        "Connect your iPhone via USB (Continuity Camera) or a USB webcam. "
        "The system will calibrate the lane automatically, then track each throw."
    )

    # Camera selection
    col1, col2 = st.columns([2, 1])
    with col1:
        camera_idx = st.number_input(
            "Camera index", min_value=0, max_value=10, value=0, step=1,
            help="0 = built-in webcam, 1 = iPhone Continuity Camera (usually). "
                 "Try different indices if your camera doesn't appear.",
        )
    with col2:
        if st.button("Detect cameras"):
            cams = _detect_cameras()
            if cams:
                for c in cams:
                    st.write(f"Camera {c['index']}: {c['resolution']}")
            else:
                st.warning("No cameras detected.")

    st.divider()

    # Controls
    col_start, col_stop, col_reset = st.columns(3)
    with col_start:
        start = st.button("Start", key="rt_start", type="primary")
    with col_stop:
        stop = st.button("Stop", key="rt_stop")
    with col_reset:
        reset_throw = st.button("New Throw", key="rt_reset")

    if stop:
        st.session_state.rt_running = False

    if start:
        st.session_state.rt_running = True
        st.session_state.rt_camera_idx = camera_idx
        st.session_state.rt_state = "calibrating"
        st.session_state.rt_throws = []
        st.session_state.rt_before_frame = None

    if reset_throw and st.session_state.rt_running:
        st.session_state.rt_state = "waiting"

    # Session stats
    if st.session_state.rt_throws:
        st.sidebar.subheader("Session Stats")
        throws = st.session_state.rt_throws
        st.sidebar.write(f"Throws: {len(throws)}")
        st.sidebar.write(f"Pins: {', '.join(str(t) for t in throws)}")
        st.sidebar.write(f"Avg: {sum(throws) / len(throws):.1f}")

    # Main feed
    if not st.session_state.rt_running:
        st.info(
            "Press **Start** to begin the live analysis session.\n\n"
            "**Controls:**\n"
            "- **Start**: Open camera and begin calibration\n"
            "- **Stop**: End session\n"
            "- **New Throw**: Reset tracker for next throw\n\n"
            "**Setup tips:**\n"
            "- Position camera behind the bowler, elevated and centered on the lane\n"
            "- iPhone: connect via USB, it should appear as Continuity Camera (index 1)\n"
            "- Keep the camera steady during calibration (~3 seconds)"
        )
        return

    _run_realtime_feed()


def _run_realtime_feed():
    """Run the real-time camera feed with lane calibration and ball tracking."""

    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from realtime import config as rt_config
    from realtime.calibrator import LaneCalibrator
    from realtime.tracker import RealtimeTracker
    from lane_detection import config as lane_config

    # Override lane config for faster real-time calibration (30 frames instead of 100)
    REALTIME_CALIBRATION_FRAMES = 30
    original_num = lane_config.NUM_COLLECTION_FRAMES
    lane_config.NUM_COLLECTION_FRAMES = REALTIME_CALIBRATION_FRAMES

    camera_idx = st.session_state.rt_camera_idx
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        st.error(
            f"Cannot open camera {camera_idx}. "
            "Try a different index or check your USB connection."
        )
        st.session_state.rt_running = False
        lane_config.NUM_COLLECTION_FRAMES = original_num
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Layout: video feed on left (large), info panel on right
    col_feed, col_info = st.columns([3, 1])

    with col_info:
        st.markdown(f"**Camera:** {w}x{h} (idx {camera_idx})")
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        state_placeholder = st.empty()
        throws_placeholder = st.empty()

    with col_feed:
        frame_placeholder = st.empty()

    calibrator = LaneCalibrator()
    tracker = None
    state = "calibrating"
    before_frame = None
    pin_settle_start = None
    result_start = None
    frame_count = 0

    try:
        while st.session_state.rt_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame_count += 1
            vis_frame = frame.copy()

            # === CALIBRATING ===
            if state == "calibrating":
                done = calibrator.add_frame(frame)
                progress = calibrator.progress

                # Progress bar
                progress_placeholder.progress(
                    progress,
                    text=f"Calibrating: {int(progress * 100)}% ({len(calibrator._collected_frames)}/{REALTIME_CALIBRATION_FRAMES} frames)"
                )

                # Draw progress bar on frame
                bar_w = int(vis_frame.shape[1] * 0.6)
                bar_h = 30
                bar_x = (vis_frame.shape[1] - bar_w) // 2
                bar_y = vis_frame.shape[0] - 60
                cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * progress)
                cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 255), -1)
                cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

                _draw_text_with_bg(vis_frame, "CALIBRATING - Keep camera steady on the lane",
                                   (10, 40), scale=0.9, color=(0, 200, 255))
                _draw_text_with_bg(vis_frame, f"{int(progress * 100)}%",
                                   (bar_x + bar_w // 2 - 20, bar_y + 22), scale=0.7, color=(255, 255, 255))

                if done:
                    b = calibrator.boundaries
                    try:
                        tracker = RealtimeTracker(
                            rt_config,
                            b["frame_width"], b["frame_height"],
                            b["foul_line_y"],
                            top_boundary_y=b.get("top_y"),
                        )
                        before_frame = frame.copy()
                        state = "waiting"
                        progress_placeholder.empty()
                        status_placeholder.success("Lane calibrated!")
                    except Exception as e:
                        status_placeholder.error(f"Tracker init failed: {e}")
                        state = "calibrating"
                        calibrator = LaneCalibrator()

            # === WAITING ===
            elif state == "waiting":
                try:
                    masked = calibrator.apply_mask(frame)
                    result = tracker.process_frame(masked)
                except Exception:
                    result = {}

                _draw_boundaries(vis_frame, calibrator)
                _draw_text_with_bg(vis_frame, "READY - Roll the ball",
                                   (10, 40), scale=0.9, color=(0, 255, 0))
                state_placeholder.info("Waiting for throw...")

                if result.get("detection") is not None:
                    state = "tracking"

            # === TRACKING ===
            elif state == "tracking":
                try:
                    masked = calibrator.apply_mask(frame)
                    result = tracker.process_frame(masked)
                except Exception:
                    result = {}

                _draw_boundaries(vis_frame, calibrator)
                _draw_ball(vis_frame, result)
                _draw_text_with_bg(vis_frame, "TRACKING",
                                   (10, 40), scale=0.9, color=(0, 100, 255))
                state_placeholder.warning("Tracking ball...")

                if result.get("throw_complete", False):
                    pin_settle_start = time.time()
                    state = "pin_settle"

            # === PIN SETTLE ===
            elif state == "pin_settle":
                _draw_boundaries(vis_frame, calibrator)
                settle_time = getattr(rt_config, 'PIN_SETTLE_FRAMES', 60) / getattr(rt_config, 'CAMERA_FPS', 30)
                elapsed = time.time() - pin_settle_start
                pct = min(elapsed / settle_time, 1.0)

                _draw_text_with_bg(vis_frame,
                                   f"Pins settling... {elapsed:.1f}s / {settle_time:.1f}s",
                                   (10, 40), scale=0.9, color=(0, 255, 255))
                state_placeholder.info(f"Waiting for pins to settle... {int(pct * 100)}%")

                if elapsed >= settle_time:
                    pins = _detect_pins_realtime(before_frame, frame, calibrator, rt_config)
                    st.session_state.rt_throws.append(pins)
                    state = "result"
                    result_start = time.time()

            # === SHOWING RESULT ===
            elif state == "result":
                _draw_boundaries(vis_frame, calibrator)
                throws = st.session_state.rt_throws
                last = throws[-1] if throws else 0

                _draw_text_with_bg(vis_frame, f"PINS DOWN: {last}/10",
                                   (10, 50), scale=1.4, color=(0, 255, 0), thickness=3)
                metrics_placeholder.metric("Last Throw", f"{last} pins")
                state_placeholder.success(f"Result: {last} pins down!")

                # Update throws display
                if throws:
                    throws_placeholder.write(
                        f"**Session:** {len(throws)} throws | "
                        f"Pins: {', '.join(str(t) for t in throws)} | "
                        f"Avg: {sum(throws)/len(throws):.1f}"
                    )

                if time.time() - result_start > 3.0:
                    if tracker:
                        tracker.reset()
                    before_frame = frame.copy()
                    state = "waiting"

            # Display frame (BGR -> RGB)
            frame_placeholder.image(
                cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            time.sleep(0.03)

    finally:
        cap.release()
        lane_config.NUM_COLLECTION_FRAMES = original_num


def _draw_text_with_bg(frame, text, pos, scale=0.8, color=(255, 255, 255),
                       thickness=2, bg_color=(0, 0, 0), bg_alpha=0.6):
    """Draw text with a semi-transparent background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad + baseline),
                  bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_boundaries(frame, calibrator):
    """Draw lane boundaries on the frame."""
    if not calibrator.calibrated or calibrator.boundaries is None:
        return
    b = calibrator.boundaries
    h, w = frame.shape[:2]

    # Foul line
    foul_y = b.get("foul_line_y")
    if foul_y:
        cv2.line(frame, (0, foul_y), (w, foul_y), (0, 0, 255), 2)

    # Left/right boundaries
    left_x = b.get("left_x")
    right_x = b.get("right_x")
    top_y = b.get("top_y", 0) or 0
    if left_x:
        cv2.line(frame, (left_x, top_y), (left_x, foul_y or h), (255, 0, 0), 2)
    if right_x:
        cv2.line(frame, (right_x, top_y), (right_x, foul_y or h), (255, 0, 0), 2)

    # Top boundary
    if top_y:
        cv2.line(frame, (left_x or 0, top_y), (right_x or w, top_y), (0, 255, 0), 2)


def _draw_ball(frame, result):
    """Draw ball detection circle on the frame."""
    det = result.get("detection")
    if det is None:
        return
    x, y, r = int(det.get("x", 0)), int(det.get("y", 0)), int(det.get("radius", 10))
    cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def _detect_pins_realtime(before_frame, after_frame, calibrator, cfg):
    """Simple pin detection via frame differencing for real-time mode."""
    if before_frame is None or calibrator.boundaries is None:
        return 0

    b = calibrator.boundaries
    top_y = b.get("top_y", 0) or 0
    foul_y = b.get("foul_line_y", before_frame.shape[0])

    pin_region_h = max(1, int((foul_y - top_y) * 0.2))
    y1 = max(0, top_y - 20)
    y2 = top_y + pin_region_h
    x1 = b.get("left_x", 0) or 0
    x2 = b.get("right_x", before_frame.shape[1]) or before_frame.shape[1]

    if y2 <= y1 or x2 <= x1:
        return 0

    before_crop = before_frame[y1:y2, x1:x2]
    after_crop = after_frame[y1:y2, x1:x2]

    gray_before = cv2.cvtColor(before_crop, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_before, gray_after)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pin_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if hasattr(cfg, "MIN_PIN_AREA") and hasattr(cfg, "MAX_PIN_AREA"):
            if cfg.MIN_PIN_AREA <= area <= cfg.MAX_PIN_AREA:
                pin_count += 1
        elif 50 <= area <= 3000:
            pin_count += 1

    return min(pin_count, 10)


if __name__ == "__main__":
    main()
