"""
Final composite video creation - Phase 6

Combines all analysis outputs into a single side-by-side video:
  [Original + sphere overlay in top-right] | [Top-down lane trajectory]

Uses FFmpeg for video composition.

Usage:
    python -m src.final_video --video Recording_3.mp4
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def get_video_dimensions(video_path):
    """Get video width and height using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'default=noprint_wrappers=1', video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    dims = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=')
            dims[k] = int(v)
    return dims.get('width', 0), dims.get('height', 0)


def create_final_video(original_video, trajectory_lane_video, sphere_video, output_path):
    """
    Compose all analysis results into a single side-by-side video.

    Layout:
        [Original video + sphere in top-right corner] | [Top-down lane trajectory]

    Parameters
    ----------
    original_video : str
        Path to the original bowling video (or trajectory overlay video).
    trajectory_lane_video : str
        Path to the top-down lane trajectory reconstruction video.
    sphere_video : str
        Path to the 3D sphere rotation visualization video.
    output_path : str
        Path for the final composite output video.
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    _, left_height = get_video_dimensions(original_video)
    if left_height == 0:
        print(f"ERROR: Cannot read video dimensions from {original_video}")
        return None

    # Use temp directory for intermediate files
    tmp_dir = tempfile.mkdtemp(prefix='bowling_final_')
    resized_right = os.path.join(tmp_dir, 'resized_right.mp4')
    padded_right = os.path.join(tmp_dir, 'padded_right.mp4')
    resized_sphere = os.path.join(tmp_dir, 'resized_sphere.mp4')
    padded_sphere = os.path.join(tmp_dir, 'padded_sphere.mp4')
    overlayed_left = os.path.join(tmp_dir, 'overlayed_left.mp4')

    ffmpeg_base = ['ffmpeg', '-y', '-loglevel', 'warning']

    try:
        # Step 1: Resize lane trajectory video to match original height
        print("  Resizing lane trajectory video...")
        subprocess.run(
            ffmpeg_base + [
                '-i', trajectory_lane_video,
                '-vf', f'scale=-1:{left_height}',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                resized_right,
            ], check=True,
        )

        # Step 2: Add left border to lane video
        subprocess.run(
            ffmpeg_base + [
                '-i', resized_right,
                '-vf', 'pad=width=iw+5:height=ih:x=5:y=0:color=black',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                padded_right,
            ], check=True,
        )

        # Step 3: Resize sphere video
        print("  Resizing sphere video...")
        sphere_width = 300
        subprocess.run(
            ffmpeg_base + [
                '-i', sphere_video,
                '-vf', f'scale={sphere_width}:-1',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                resized_sphere,
            ], check=True,
        )

        # Step 4: Add border to sphere
        subprocess.run(
            ffmpeg_base + [
                '-i', resized_sphere,
                '-vf', 'pad=width=iw+5:height=ih+5:x=5:y=0:color=black',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                padded_sphere,
            ], check=True,
        )

        # Step 5: Overlay sphere on top-right of original
        print("  Overlaying sphere on original...")
        subprocess.run(
            ffmpeg_base + [
                '-i', original_video,
                '-i', padded_sphere,
                '-filter_complex', '[0:v][1:v]overlay=W-w:0',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                overlayed_left,
            ], check=True,
        )

        # Step 6: Stack side by side
        print("  Composing final video...")
        subprocess.run(
            ffmpeg_base + [
                '-i', overlayed_left,
                '-i', padded_right,
                '-filter_complex', '[0:v][1:v]hstack=inputs=2',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                output_path,
            ], check=True,
        )

        print(f"  Final video saved to: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: FFmpeg failed: {e}")
        return None

    finally:
        # Cleanup temp files
        for f in [resized_right, padded_right, resized_sphere, padded_sphere, overlayed_left]:
            if os.path.exists(f):
                os.remove(f)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def compose_for_video(video_name, output_dir=None):
    """
    Auto-detect analysis outputs for a video and compose final video.

    Looks for:
    - Original video or trajectory overlay in output/<video_name>/
    - Lane trajectory video in output/<video_name>/trajectory_3d/
    - Sphere video in output/<video_name>/spin_analysis/
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    base = os.path.join(output_dir, video_name)

    # Find original video (prefer overlay if available)
    original = None
    for candidate in [
        os.path.join(base, 'ball_detection', 'Ball_detected_processed.mp4'),
        os.path.join(base, 'ball_detection', 'ball_tracking_overlay_ransac.mp4'),
        os.path.join(PROJECT_ROOT, 'assets', 'input', f'{video_name}.mp4'),
    ]:
        if os.path.exists(candidate):
            original = candidate
            break

    # Find lane trajectory video
    lane_video = None
    for candidate in [
        os.path.join(base, 'trajectory_3d', f'{video_name}_trajectory_overhead.mp4'),
    ]:
        if os.path.exists(candidate):
            lane_video = candidate
            break

    # Find sphere video
    sphere_video = None
    for candidate in [
        os.path.join(base, 'spin_analysis', 'sphere_video.mp4'),
    ]:
        if os.path.exists(candidate):
            sphere_video = candidate
            break

    if not original:
        print(f"ERROR: No original/overlay video found for {video_name}")
        return None
    if not lane_video:
        print(f"WARNING: No lane trajectory video found for {video_name}")
    if not sphere_video:
        print(f"WARNING: No sphere video found for {video_name}")

    # If we have all 3, do full composition
    if original and lane_video and sphere_video:
        final_path = os.path.join(base, f'{video_name}_final.mp4')
        return create_final_video(original, lane_video, sphere_video, final_path)

    # If just original + lane (no sphere), do side-by-side without overlay
    if original and lane_video:
        final_path = os.path.join(base, f'{video_name}_final.mp4')
        _, h = get_video_dimensions(original)
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        try:
            subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'warning',
                '-i', lane_video,
                '-vf', f'scale=-1:{h}',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', tmp,
            ], check=True)
            subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'warning',
                '-i', original, '-i', tmp,
                '-filter_complex', '[0:v][1:v]hstack=inputs=2',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                final_path,
            ], check=True)
            return final_path
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    return None


def main():
    parser = argparse.ArgumentParser(description='Create final composite video')
    parser.add_argument('--video', type=str, required=True, help='Video name (e.g. Recording_3)')
    args = parser.parse_args()

    video_name = Path(args.video).stem
    result = compose_for_video(video_name)
    if result:
        print(f"\nFinal video: {result}")
    else:
        print("\nFailed to create final video.")


if __name__ == '__main__':
    main()
