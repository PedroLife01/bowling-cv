"""
Streamlit web interface for Bowling CV Analysis Pipeline.

Upload a bowling video and run each analysis phase interactively:
  Phase 1: Lane Detection
  Phase 2: Ball Detection
  Phase 3: Trajectory Reconstruction
  Phase 4: Pin Detection
  Phase 5: Spin Analysis

Usage:
    streamlit run app.py
"""

import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.title("Bowling CV Analysis Pipeline")

    # Session state
    for key in ("video_path", "video_name"):
        st.session_state.setdefault(key, None)

    # ── Sidebar: Upload & Status ──
    with st.sidebar:
        st.header("Video Input")
        uploaded = st.file_uploader("Upload bowling video", type=["mp4", "avi", "mov"])
        if uploaded:
            path = _save_uploaded_video(uploaded)
            st.session_state.video_path = str(path)
            st.session_state.video_name = path.stem
            st.success(f"Saved: {path.name}")

        if st.session_state.video_name:
            st.divider()
            st.subheader("Pipeline Status")
            vn = st.session_state.video_name
            out = OUTPUT_DIR / vn

            phases = {
                "Phase 1 - Lane": (out / "boundary_data.json").exists(),
                "Phase 2 - Ball": (out / "ball_detection").exists(),
                "Phase 3 - Trajectory": (out / "trajectory_3d").exists(),
                "Phase 4 - Pins": (out / "pin_detection").exists(),
                "Phase 5 - Spin": (out / "spin_analysis").exists(),
            }
            for name, done in phases.items():
                icon = "done" if done else "pending"
                st.write(f"{'[x]' if done else '[ ]'} {name}")

    if not st.session_state.video_path:
        st.info("Upload a bowling video to get started.")
        return

    video_path = st.session_state.video_path
    video_name = st.session_state.video_name
    video_file = Path(video_path).name

    # ── Phase 1: Lane Detection ──
    with st.expander("Phase 1: Lane Detection", expanded=True):
        st.write("Detects lane boundaries (foul line, left/right gutters, pin area).")

        if st.button("Run Phase 1", key="run_p1"):
            _run_module(
                ["src.lane_detection.main", "--video", video_file],
                "Phase 1 - Lane Detection",
            )
            st.rerun()

        # Show results
        boundary_json = OUTPUT_DIR / video_name / "boundary_data.json"
        if boundary_json.exists():
            st.success("Lane boundaries detected.")
            import json
            with open(boundary_json) as f:
                data = json.load(f)
            with st.expander("Boundary Data", expanded=False):
                st.json(data)

    # ── Phase 2: Ball Detection ──
    with st.expander("Phase 2: Ball Detection"):
        st.write("Tracks the bowling ball from foul line to pins using MOG2 + Kalman filter.")

        if st.button("Run Phase 2", key="run_p2"):
            _run_module(
                ["src.ball_detection.main", "--video", video_file],
                "Phase 2 - Ball Detection",
            )
            st.rerun()

        # Show results
        ball_dir = OUTPUT_DIR / video_name / "ball_detection"
        if ball_dir.exists():
            st.success("Ball detection complete.")

            # Show trajectory CSV
            traj_csv = ball_dir / "trajectory_processed_original.csv"
            if traj_csv.exists():
                import pandas as pd
                df = pd.read_csv(traj_csv)
                st.write(f"Trajectory: {len(df)} frames, {df['x'].notna().sum()} valid detections")
                with st.expander("Trajectory Data", expanded=False):
                    st.dataframe(df.head(50))

            # Show overlay video
            videos = list(ball_dir.glob("*overlay*.mp4")) + list(ball_dir.glob("*Stage_H*.mp4"))
            for v in videos[:1]:
                web = _reencode_for_web(str(v))
                _display_video(web)

    # ── Phase 3: Trajectory Reconstruction ──
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

            # Show overhead video
            overhead_vids = list(traj_dir.glob("*overhead*.mp4"))
            for v in overhead_vids[:1]:
                web = _reencode_for_web(str(v))
                _display_video(web, max_height=600)

    # ── Phase 4: Pin Detection ──
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

            # Show pin results
            pin_results = list(pin_dir.glob("*.json"))
            if pin_results:
                import json
                with open(pin_results[0]) as f:
                    pins = json.load(f)
                toppled = pins.get("toppled_pins", "?")
                result = pins.get("result", "?")
                st.metric("Toppled Pins", f"{toppled}/10", delta=result)

    # ── Phase 5: Spin Analysis ──
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
                    fps = 30
                    st.metric("Avg Angular Velocity", f"{mean_angle * fps:.2f} rad/s")

                with st.expander("Rotation Data", expanded=False):
                    st.dataframe(df.head(50))

            # Show sphere video
            sphere_vids = list(spin_dir.glob("*sphere*.mp4"))
            for v in sphere_vids[:1]:
                web = _reencode_for_web(str(v))
                _display_video(web)


if __name__ == "__main__":
    main()
