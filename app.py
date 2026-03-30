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
    """
    Real-time camera analysis page.

    Flow:
    1. SETUP     - Select camera, press Connect
    2. CALIBRATE - Auto lane detection (~1 second)
    3. READY     - Shows calibrated lane. Press "Arm Throw" to start listening
    4. ARMED     - Waiting for ball detection, auto-transitions to tracking
    5. TRACKING  - Following the ball down the lane
    6. REVIEW    - Shows result (pins, replay of captured frames). Save or New Throw.
    """

    # Session state init
    defaults = {
        "rt_phase": "setup",       # setup, calibrate, ready, armed, tracking, review
        "rt_camera_idx": 0,
        "rt_throws": [],           # list of dicts: {pins, frames, trajectory}
        "rt_captured_frames": [],   # frames captured during current throw
        "rt_trajectory": [],        # ball positions during current throw
        "rt_last_pins": 0,
        "rt_calibrator": None,
        "rt_tracker": None,
        "rt_before_frame": None,
        "rt_boundaries": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    phase = st.session_state.rt_phase

    # ── SETUP ──
    if phase == "setup":
        st.subheader("Real-Time Bowling Analysis")

        st.markdown("""
**How it works:**
1. Connect your camera (iPhone via USB / Continuity Camera, or webcam)
2. Point it at the lane from behind, elevated and centered
3. The system calibrates the lane boundaries automatically
4. Press **Arm Throw** when you're ready to bowl
5. The system tracks the ball and counts pins
6. Review the result, save it, or start a new throw
        """)

        col1, col2 = st.columns([2, 1])
        with col1:
            camera_idx = st.number_input(
                "Camera index", min_value=0, max_value=10, value=0, step=1,
                help="0 = built-in webcam, 1 = iPhone Continuity Camera (usually)",
            )
        with col2:
            if st.button("Detect cameras"):
                with st.spinner("Scanning..."):
                    cams = _detect_cameras()
                if cams:
                    for c in cams:
                        st.success(f"Camera {c['index']}: {c['resolution']}")
                else:
                    st.warning("No cameras found.")

        if st.button("Connect Camera", type="primary"):
            st.session_state.rt_camera_idx = camera_idx
            st.session_state.rt_phase = "calibrate"
            st.rerun()

    # ── CALIBRATE ──
    elif phase == "calibrate":
        _run_calibration_phase()

    # ── READY ──
    elif phase == "ready":
        _show_ready_phase()

    # ── ARMED / TRACKING ──
    elif phase in ("armed", "tracking"):
        _run_throw_tracking()

    # ── REVIEW ──
    elif phase == "review":
        _show_review_phase()


def _run_calibration_phase():
    """Calibrate lane boundaries from live camera."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from realtime.calibrator import LaneCalibrator
    from lane_detection import config as lane_config

    CAL_FRAMES = 30
    original_num = lane_config.NUM_COLLECTION_FRAMES
    lane_config.NUM_COLLECTION_FRAMES = CAL_FRAMES

    st.subheader("Calibrating Lane...")
    st.write("Keep the camera steady and pointed at the lane.")

    frame_placeholder = st.empty()
    progress_bar = st.progress(0, text="Collecting frames...")

    cap = cv2.VideoCapture(st.session_state.rt_camera_idx)
    if not cap.isOpened():
        st.error("Cannot open camera. Go back and try a different index.")
        if st.button("Back to Setup"):
            st.session_state.rt_phase = "setup"
            st.rerun()
        lane_config.NUM_COLLECTION_FRAMES = original_num
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    calibrator = LaneCalibrator()
    calibrated = False

    try:
        while not calibrated:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            done = calibrator.add_frame(frame)
            progress = calibrator.progress

            progress_bar.progress(progress, text=f"Collecting frames: {int(progress * 100)}%")

            # Draw on frame
            vis = frame.copy()
            bar_w = int(vis.shape[1] * 0.5)
            bar_x = (vis.shape[1] - bar_w) // 2
            bar_y = vis.shape[0] - 50
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 25), (40, 40, 40), -1)
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 25), (0, 200, 255), -1)
            _draw_text_with_bg(vis, "CALIBRATING - Keep camera steady",
                               (10, 35), scale=0.8, color=(0, 200, 255))

            frame_placeholder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_container_width=True)

            if done:
                calibrated = True
                st.session_state.rt_calibrator = calibrator
                st.session_state.rt_boundaries = calibrator.boundaries
                st.session_state.rt_before_frame = frame.copy()

                # Save a snapshot of the calibrated view
                vis_cal = frame.copy()
                calibrator.draw_boundaries(vis_cal)
                _draw_text_with_bg(vis_cal, "CALIBRATED", (10, 35), scale=1.0, color=(0, 255, 0))
                frame_placeholder.image(cv2.cvtColor(vis_cal, cv2.COLOR_BGR2RGB),
                                        channels="RGB", use_container_width=True)

            time.sleep(0.03)

    finally:
        cap.release()
        lane_config.NUM_COLLECTION_FRAMES = original_num

    if calibrated:
        b = calibrator.boundaries
        st.success(
            f"Lane calibrated! "
            f"Foul line: Y={b['foul_line_y']}, "
            f"Left: X={b['left_x']}, Right: X={b['right_x']}"
        )
        st.session_state.rt_phase = "ready"
        time.sleep(1)
        st.rerun()


def _show_ready_phase():
    """Show calibrated lane and controls."""
    st.subheader("Lane Calibrated - Ready to Bowl")

    # Show boundaries info
    b = st.session_state.rt_boundaries
    if b:
        col1, col2, col3 = st.columns(3)
        col1.metric("Foul Line", f"Y={b['foul_line_y']}")
        col2.metric("Lane Left", f"X={b['left_x']}")
        col3.metric("Lane Right", f"X={b['right_x']}")

    st.divider()

    # Show session history
    throws = st.session_state.rt_throws
    if throws:
        st.markdown("### Session History")
        for i, t in enumerate(throws, 1):
            pins = t["pins"]
            label = "STRIKE!" if pins == 10 else f"{pins} pins"
            st.write(f"**Throw {i}:** {label}")
        total = sum(t["pins"] for t in throws)
        st.write(f"**Total:** {total} pins in {len(throws)} throws (avg: {total/len(throws):.1f})")
        st.divider()

    # Main action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Arm Throw", type="primary", help="Start listening for the ball"):
            st.session_state.rt_captured_frames = []
            st.session_state.rt_trajectory = []
            st.session_state.rt_phase = "armed"
            st.rerun()
    with col2:
        if st.button("Recalibrate", help="Re-detect lane boundaries"):
            st.session_state.rt_phase = "calibrate"
            st.rerun()
    with col3:
        if throws and st.button("Save Session"):
            _save_realtime_session()

    st.info(
        "Press **Arm Throw** when you're ready to bowl. "
        "The system will detect the ball automatically and track it down the lane."
    )


def _run_throw_tracking():
    """Armed/Tracking: capture frames until throw completes."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from realtime import config as rt_config
    from realtime.calibrator import LaneCalibrator
    from realtime.tracker import RealtimeTracker

    calibrator = st.session_state.rt_calibrator
    b = st.session_state.rt_boundaries

    if calibrator is None or b is None:
        st.error("Lost calibration data. Recalibrating...")
        st.session_state.rt_phase = "calibrate"
        st.rerun()
        return

    cap = cv2.VideoCapture(st.session_state.rt_camera_idx)
    if not cap.isOpened():
        st.error("Cannot open camera.")
        st.session_state.rt_phase = "setup"
        st.rerun()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Init tracker
    try:
        tracker = RealtimeTracker(
            rt_config,
            b["frame_width"], b["frame_height"],
            b["foul_line_y"],
            top_boundary_y=b.get("top_y"),
        )
    except Exception as e:
        cap.release()
        st.error(f"Failed to initialize tracker: {e}")
        st.session_state.rt_phase = "ready"
        st.rerun()
        return

    is_armed = st.session_state.rt_phase == "armed"

    col_feed, col_info = st.columns([3, 1])
    with col_info:
        state_text = st.empty()
        if is_armed:
            state_text.info("ARMED - Waiting for ball...")
        else:
            state_text.warning("TRACKING ball...")
        cancel_placeholder = st.empty()
        if cancel_placeholder.button("Cancel", key="cancel_throw"):
            cap.release()
            st.session_state.rt_phase = "ready"
            st.rerun()
            return

    with col_feed:
        frame_placeholder = st.empty()

    captured_frames = st.session_state.rt_captured_frames
    trajectory = st.session_state.rt_trajectory
    before_frame = st.session_state.rt_before_frame
    tracking = not is_armed
    pin_settle_start = None
    max_frames = 600  # 20 second timeout

    try:
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            vis = frame.copy()
            _draw_boundaries(vis, calibrator)

            try:
                masked = calibrator.apply_mask(frame)
                result = tracker.process_frame(masked)
            except Exception:
                result = {}

            det = result.get("detection")

            if not tracking:
                # ARMED - waiting for ball
                _draw_text_with_bg(vis, "ARMED - Waiting for ball...",
                                   (10, 40), scale=0.9, color=(0, 200, 255))

                if det is not None:
                    tracking = True
                    state_text.warning("TRACKING ball!")
                    before_frame = frame.copy()
                    st.session_state.rt_before_frame = before_frame
            else:
                # TRACKING
                _draw_ball(vis, result)
                _draw_text_with_bg(vis, "TRACKING",
                                   (10, 40), scale=0.9, color=(0, 100, 255))

                # Record frame and position
                captured_frames.append(frame.copy())
                if det:
                    trajectory.append({
                        "frame": len(captured_frames) - 1,
                        "x": det.get("x"), "y": det.get("y"),
                        "radius": det.get("radius"),
                    })

                    # Draw trajectory trail
                    for j in range(1, len(trajectory)):
                        p1 = trajectory[j - 1]
                        p2 = trajectory[j]
                        if p1.get("x") and p2.get("x"):
                            cv2.line(vis,
                                     (int(p1["x"]), int(p1["y"])),
                                     (int(p2["x"]), int(p2["y"])),
                                     (255, 0, 255), 2)

                # Check if throw complete
                if result.get("throw_complete", False):
                    _draw_text_with_bg(vis, "THROW COMPLETE!",
                                       (10, 80), scale=1.0, color=(0, 255, 0))
                    frame_placeholder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                            channels="RGB", use_container_width=True)

                    # Wait for pins to settle
                    state_text.info("Pins settling...")
                    settle_frames = getattr(rt_config, 'PIN_SETTLE_FRAMES', 60)
                    for _ in range(settle_frames):
                        ret, frame = cap.read()
                        if ret:
                            captured_frames.append(frame.copy())
                        time.sleep(0.03)

                    # Detect pins
                    after_frame = frame if ret else captured_frames[-1]
                    pins = _detect_pins_realtime(before_frame, after_frame, calibrator, rt_config)
                    st.session_state.rt_last_pins = pins
                    st.session_state.rt_captured_frames = captured_frames
                    st.session_state.rt_trajectory = trajectory

                    # Record throw
                    st.session_state.rt_throws.append({
                        "pins": pins,
                        "num_frames": len(captured_frames),
                        "trajectory_points": len(trajectory),
                    })

                    cap.release()
                    st.session_state.rt_phase = "review"
                    st.rerun()
                    return

            frame_placeholder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_container_width=True)
            time.sleep(0.03)

        # Timeout
        state_text.warning("Timeout - no throw detected in 20 seconds.")

    finally:
        cap.release()

    st.session_state.rt_phase = "ready"
    st.rerun()


def _show_review_phase():
    """Show throw results with replay and save options."""
    pins = st.session_state.rt_last_pins
    captured = st.session_state.rt_captured_frames
    trajectory = st.session_state.rt_trajectory
    throws = st.session_state.rt_throws

    # Header with result
    if pins == 10:
        st.subheader("STRIKE!")
    elif pins == 0:
        st.subheader("Gutter Ball")
    else:
        st.subheader(f"Result: {pins} Pins Down")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pins Down", f"{pins}/10")
    col2.metric("Frames Captured", len(captured))
    col3.metric("Trajectory Points", len(trajectory))

    st.divider()

    # Replay
    if captured:
        st.markdown("### Throw Replay")
        frame_idx = st.slider("Frame", 0, len(captured) - 1, len(captured) // 2, key="replay_slider")

        vis = captured[frame_idx].copy()
        calibrator = st.session_state.rt_calibrator
        if calibrator:
            _draw_boundaries(vis, calibrator)

        # Draw trajectory trail up to this frame
        for j in range(1, len(trajectory)):
            t = trajectory[j]
            if t["frame"] > frame_idx:
                break
            p1 = trajectory[j - 1]
            if p1.get("x") and t.get("x"):
                cv2.line(vis,
                         (int(p1["x"]), int(p1["y"])),
                         (int(t["x"]), int(t["y"])),
                         (255, 0, 255), 2)

        # Draw current ball position
        for t in trajectory:
            if t["frame"] == frame_idx and t.get("x"):
                cv2.circle(vis, (int(t["x"]), int(t["y"])),
                           int(t.get("radius", 10)), (0, 255, 255), 2)
                break

        _draw_text_with_bg(vis, f"Frame {frame_idx}/{len(captured)-1}  |  {pins} pins",
                           (10, 35), scale=0.7, color=(255, 255, 255))

        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    # Trajectory data
    if trajectory:
        st.markdown("### Trajectory Data")
        import pandas as pd
        df = pd.DataFrame(trajectory)
        st.dataframe(df, use_container_width=True)

    st.divider()

    # Session summary
    if throws:
        st.markdown("### Session Summary")
        for i, t in enumerate(throws, 1):
            p = t["pins"]
            label = "STRIKE!" if p == 10 else f"{p} pins"
            st.write(f"Throw {i}: {label} ({t['trajectory_points']} tracked points)")
        total = sum(t["pins"] for t in throws)
        st.metric("Session Total", f"{total} pins in {len(throws)} throws")

    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("New Throw", type="primary"):
            st.session_state.rt_captured_frames = []
            st.session_state.rt_trajectory = []
            st.session_state.rt_phase = "ready"
            st.rerun()
    with col2:
        if st.button("Save Session"):
            _save_realtime_session()
    with col3:
        if st.button("End Session"):
            st.session_state.rt_phase = "setup"
            st.rerun()


def _save_realtime_session():
    """Save the current session data to disk."""
    from datetime import datetime
    import pandas as pd

    throws = st.session_state.rt_throws
    if not throws:
        st.warning("No throws to save.")
        return

    session_dir = OUTPUT_DIR / "realtime_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = session_dir / f"session_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "num_throws": len(throws),
        "total_pins": sum(t["pins"] for t in throws),
        "average": sum(t["pins"] for t in throws) / len(throws),
        "throws": throws,
        "boundaries": st.session_state.rt_boundaries,
    }

    # Make JSON-serializable
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(session_file, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)

    # Save trajectory CSV if available
    trajectory = st.session_state.rt_trajectory
    if trajectory:
        csv_file = session_dir / f"trajectory_{timestamp}.csv"
        pd.DataFrame(trajectory).to_csv(csv_file, index=False)
        st.success(f"Session saved to {session_file} and {csv_file}")
    else:
        st.success(f"Session saved to {session_file}")


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
