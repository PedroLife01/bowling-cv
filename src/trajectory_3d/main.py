"""
Main entry point for trajectory 3D reconstruction - Phase 3.

Reconstructs the ball trajectory in real-world lane coordinates by
applying homography-based perspective transformation to ball positions
detected in Phase 2.

Requires:
- Phase 1 output: boundary_data.json (lane boundaries)
- Phase 2 output: trajectory_processed_original.csv (ball positions)

Version: 1.0.0
Authors: Pedro Roriz
Created: March 30, 2026

Usage:
    python -m src.trajectory_3d.main
    python -m src.trajectory_3d.main --video cropped_test3.mp4
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_3d import config
from trajectory_3d.reconstruction import process_reconstruction


def main():
    """
    Main entry point for trajectory reconstruction pipeline.

    Steps:
    1. Convert boundary data to lane corners
    2. Compute per-frame homography
    3. Transform ball positions to lane coordinates
    4. Smooth trajectory with Savitzky-Golay filter
    5. Generate overhead visualization video
    """
    parser = argparse.ArgumentParser(
        description='Trajectory 3D Reconstruction Pipeline - Phase 3'
    )
    parser.add_argument(
        '--video', type=str,
        help='Process a single video file (e.g. cropped_test3.mp4)'
    )
    parser.add_argument(
        '--skip-visualization', action='store_true',
        help='Skip overhead visualization video generation'
    )
    parser.add_argument(
        '--skip-smoothing', action='store_true',
        help='Skip Savitzky-Golay smoothing step'
    )
    args = parser.parse_args()

    # Determine which videos to process
    if args.video:
        videos = [args.video]
    else:
        videos = config.VIDEO_FILES

    # Apply CLI overrides to config
    if args.skip_visualization:
        config.SAVE_VISUALIZATION_VIDEO = False

    print(f"\n{'#'*80}")
    print(f"# TRAJECTORY 3D RECONSTRUCTION - PHASE 3")
    print(f"# Step 1: Boundary -> Lane Corners")
    print(f"# Step 2: Compute Homographies")
    print(f"# Step 3: Transform Ball Positions")
    print(f"# Step 4: Smooth Trajectory")
    print(f"# Step 5: Overhead Visualization")
    print(f"# Processing {len(videos)} video(s)")
    print(f"{'#'*80}\n")

    results = {}

    for video_file in videos:
        video_name = Path(video_file).stem

        try:
            result = process_reconstruction(video_name, config)
            results[video_name] = result

            # Print summary
            transformed = result['transformed_df']
            smoothed = result['smoothed_df']
            valid_raw = transformed['x'].notna().sum()
            valid_smooth = smoothed['x'].notna().sum()

            print(f"\nSummary for {video_name}:")
            print(f"  Transformed points: {valid_raw}")
            print(f"  Smoothed points:    {valid_smooth}")

            if valid_smooth > 0:
                x_range = (smoothed['x'].min(), smoothed['x'].max())
                y_range = (smoothed['y'].min(), smoothed['y'].max())
                print(f"  X range: {x_range[0]:.1f} - {x_range[1]:.1f} (lane width: {config.LANE_WIDTH})")
                print(f"  Y range: {y_range[0]:.1f} - {y_range[1]:.1f} (lane length: {config.LANE_LENGTH})")

        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print(f"Skipping {video_name}.\n")
            continue
        except Exception as e:
            print(f"\nERROR processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'#'*80}")
    print(f"# TRAJECTORY RECONSTRUCTION COMPLETE")
    print(f"# Processed {len(results)}/{len(videos)} videos successfully")
    print(f"{'#'*80}\n")

    return results


if __name__ == '__main__':
    main()
