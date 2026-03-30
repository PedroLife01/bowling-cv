"""
Main entry point for spin/rotation analysis - Phase 5.

Pipeline:
1. Spin detection: optical flow + Kabsch algorithm -> rotation_data.csv
2. Post-processing: outlier removal, smoothing -> rotation_data_processed.csv
3. Visualization: 3D sphere video -> sphere_video.mp4

Usage:
    python -m src.spin_analysis.main --video path/to/video.mp4

    Required: ball trajectory CSV must exist at:
        output/<video_name>/ball_detection/trajectory_processed_original.csv

Version: 1.0.0
Created: March 30, 2026
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spin_analysis import config
from spin_analysis.detection import process_spin
from spin_analysis.post_processing import process_rotation_data
from spin_analysis.visualization import create_sphere_video


def main():
    """
    Main entry point for spin analysis pipeline.

    Steps:
    1. Detect per-frame rotation from optical flow (Kabsch algorithm)
    2. Post-process rotation data (outlier removal, smoothing, interpolation)
    3. Generate 3D sphere visualization video
    """
    parser = argparse.ArgumentParser(
        description='Bowling Ball Spin Analysis - Phase 5'
    )
    parser.add_argument(
        '--video', type=str, required=True,
        help='Path to input bowling video file'
    )
    parser.add_argument(
        '--ball-csv', type=str, default=None,
        help='Path to ball trajectory CSV (default: auto-detect from output dir)'
    )
    parser.add_argument(
        '--skip-detection', action='store_true',
        help='Skip Step 1 (spin detection), use existing rotation_data.csv'
    )
    parser.add_argument(
        '--skip-postprocess', action='store_true',
        help='Skip Step 2 (post-processing)'
    )
    parser.add_argument(
        '--skip-visualization', action='store_true',
        help='Skip Step 3 (sphere video generation)'
    )
    args = parser.parse_args()

    video_path = args.video
    video_name = Path(video_path).stem

    # Auto-detect ball trajectory CSV (prefer worcester Adjusted_positions)
    if args.ball_csv:
        ball_csv_path = args.ball_csv
    else:
        ball_dir = os.path.join(config.OUTPUT_DIR, video_name, 'ball_detection')
        candidates = [
            os.path.join(ball_dir, 'Adjusted_positions.csv'),
            os.path.join(ball_dir, 'ball_positions.csv'),
            os.path.join(ball_dir, 'Circle_positions_cleaned.csv'),
            os.path.join(ball_dir, 'trajectory_processed_original.csv'),
        ]
        ball_csv_path = None
        for c in candidates:
            if os.path.exists(c):
                ball_csv_path = c
                break
        if ball_csv_path is None:
            ball_csv_path = candidates[-1]  # default for error message

    # Output directory
    output_dir = os.path.join(config.OUTPUT_DIR, video_name, 'spin_analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#' * 80}")
    print(f"# BOWLING BALL SPIN ANALYSIS - PHASE 5")
    print(f"# Step 1: Spin Detection (Optical Flow + Kabsch)")
    print(f"# Step 2: Post-Processing (Outlier Removal + Smoothing)")
    print(f"# Step 3: Visualization (3D Sphere Video)")
    print(f"# Video: {video_name}")
    print(f"{'#' * 80}\n")

    rotation_csv_path = os.path.join(output_dir, 'rotation_data.csv')
    processed_csv_path = os.path.join(output_dir, 'rotation_data_processed.csv')
    sphere_video_path = os.path.join(output_dir, 'sphere_video.mp4')

    # Step 1: Spin Detection
    if not args.skip_detection:
        print(f"\n{'=' * 60}")
        print(f"Step 1: Spin Detection")
        print(f"{'=' * 60}")

        if not os.path.exists(ball_csv_path):
            print(f"ERROR: Ball trajectory CSV not found: {ball_csv_path}")
            print("  Run ball detection (Phase 2) first, or specify --ball-csv")
            sys.exit(1)

        if not os.path.exists(video_path):
            print(f"ERROR: Video not found: {video_path}")
            sys.exit(1)

        rotation_csv_path = process_spin(video_path, ball_csv_path, config)
        print(f">>> Spin detection complete")
    else:
        print("Skipping spin detection (using existing rotation_data.csv)")
        if not os.path.exists(rotation_csv_path):
            print(f"ERROR: rotation_data.csv not found: {rotation_csv_path}")
            sys.exit(1)

    # Step 2: Post-Processing
    if not args.skip_postprocess:
        print(f"\n{'=' * 60}")
        print(f"Step 2: Post-Processing")
        print(f"{'=' * 60}")

        processed_csv_path = process_rotation_data(
            rotation_csv_path, config,
            ball_csv_path=ball_csv_path,
            video_path=video_path,
        )
        print(f">>> Post-processing complete")
    else:
        print("Skipping post-processing")
        if not os.path.exists(processed_csv_path):
            processed_csv_path = rotation_csv_path

    # Step 3: Visualization
    if not args.skip_visualization and config.SAVE_SPHERE_VIDEO:
        print(f"\n{'=' * 60}")
        print(f"Step 3: 3D Sphere Visualization")
        print(f"{'=' * 60}")

        csv_for_viz = processed_csv_path if os.path.exists(processed_csv_path) \
            else rotation_csv_path

        sphere_video_path = create_sphere_video(
            csv_for_viz, sphere_video_path, config
        )
        print(f">>> Visualization complete")
    else:
        print("Skipping sphere video generation")

    print(f"\n{'#' * 80}")
    print(f"# SPIN ANALYSIS COMPLETE")
    print(f"#")
    print(f"# Output directory: {output_dir}")
    if os.path.exists(rotation_csv_path):
        print(f"#   rotation_data.csv           (raw detection)")
    if os.path.exists(processed_csv_path):
        print(f"#   rotation_data_processed.csv  (post-processed)")
    if os.path.exists(sphere_video_path):
        print(f"#   sphere_video.mp4             (3D visualization)")
    print(f"{'#' * 80}\n")


if __name__ == '__main__':
    main()
