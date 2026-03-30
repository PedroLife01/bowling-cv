"""
Adapter to use bowling-analysis (worcester) ball detection pipeline.

Runs worcester as a subprocess to avoid module name collisions,
since both projects have a `ball_detection` package.

Usage:
    from ball_detection.worcester_adapter import run_worcester_ball_detection
    results = run_worcester_ball_detection(video_path, lane_csv_path, output_dir)
"""

import os
import subprocess
import sys
import textwrap

WORCESTER_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'bowling-analysis', 'worcester', 'src'
))


def run_worcester_ball_detection(video_path, lane_csv_path, output_dir):
    """
    Run the full worcester ball detection pipeline via subprocess.

    Returns dict with paths to output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.abspath(video_path)
    lane_csv_path = os.path.abspath(lane_csv_path)
    output_dir = os.path.abspath(output_dir)

    raw_video = os.path.join(output_dir, 'Ball_detected_raw.mp4')
    raw_csv = os.path.join(output_dir, 'Circle_positions_raw.csv')
    cleaned_csv = os.path.join(output_dir, 'Circle_positions_cleaned.csv')
    transformed_raw_csv = os.path.join(output_dir, 'Transformed_positions_raw.csv')
    transformed_processed_csv = os.path.join(output_dir, 'Transformed_positions_processed.csv')
    adjusted_csv = os.path.join(output_dir, 'Adjusted_positions.csv')
    processed_video = os.path.join(output_dir, 'Ball_detected_processed.mp4')

    script = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {WORCESTER_SRC!r})

        from ball_detection.Detection import process_video_with_roi
        from ball_detection.Post_processing_outliers import process_data
        from ball_detection.Post_processing_smoothing import process_coordinates_final
        from reconstruction.Reconstruction import process_reconstruction
        from reconstruction.Post_processing_positions import process_data_transformed

        print("[Worcester] Step 1: Ball detection (Hough Circle)...")
        process_video_with_roi({video_path!r}, {lane_csv_path!r}, {raw_video!r}, {raw_csv!r})

        print("[Worcester] Step 2: Outlier removal...")
        process_data({raw_csv!r}, {cleaned_csv!r})

        print("[Worcester] Step 3: Homography transformation...")
        process_reconstruction({lane_csv_path!r}, {cleaned_csv!r}, {transformed_raw_csv!r})
        process_data_transformed({transformed_raw_csv!r}, {transformed_processed_csv!r})

        print("[Worcester] Step 4: Radius smoothing + adjusted positions...")
        process_coordinates_final({video_path!r}, {cleaned_csv!r}, {transformed_processed_csv!r}, {adjusted_csv!r}, {processed_video!r})

        print("[Worcester] Ball detection complete!")
    """)

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True,
        cwd=WORCESTER_SRC,
    )

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  ERROR: Worcester ball detection failed:")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-10:]:
                print(f"    {line}")
        return None

    return {
        'raw_csv': raw_csv,
        'cleaned_csv': cleaned_csv,
        'transformed_raw_csv': transformed_raw_csv,
        'transformed_processed_csv': transformed_processed_csv,
        'adjusted_csv': adjusted_csv,
        'overlay_video': raw_video,
        'processed_video': processed_video,
    }


def check_worcester_available():
    """Check if worcester source is available."""
    return os.path.isdir(WORCESTER_SRC) and os.path.exists(
        os.path.join(WORCESTER_SRC, 'ball_detection', 'Detection.py')
    )
