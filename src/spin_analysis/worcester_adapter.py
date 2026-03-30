"""
Adapter to use bowling-analysis (worcester) spin detection pipeline.

Runs worcester spin detection as a subprocess to avoid import conflicts.
"""

import os
import subprocess
import sys
import textwrap

WORCESTER_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'bowling-analysis', 'worcester', 'src'
))


def run_worcester_spin(video_path, ball_csv_path, output_dir):
    """
    Run worcester spin detection + post-processing + visualization.

    Parameters
    ----------
    video_path : str
        Path to the input bowling video.
    ball_csv_path : str
        Path to Adjusted_positions.csv (frame,x,y,radius).
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict or None
        Paths to output files, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.abspath(video_path)
    ball_csv_path = os.path.abspath(ball_csv_path)
    output_dir = os.path.abspath(output_dir)

    rotation_csv = os.path.join(output_dir, 'rotation_data.csv')
    rotation_processed_csv = os.path.join(output_dir, 'rotation_data_processed.csv')
    sphere_raw_video = os.path.join(output_dir, 'sphere_raw.mp4')
    sphere_video = os.path.join(output_dir, 'sphere_video.mp4')

    script = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {WORCESTER_SRC!r})

        from spin.Detection import process_spin
        from spin.Post_processing import spin_post_processing
        from spin.Video_creation import spin_video_creation

        print("[Worcester Spin] Step 1: Optical flow + Kabsch rotation detection...")
        process_spin({video_path!r}, {ball_csv_path!r}, {rotation_csv!r})

        print("[Worcester Spin] Step 2: Post-processing (outlier removal + smoothing)...")
        spin_post_processing({rotation_csv!r}, {rotation_processed_csv!r}, {ball_csv_path!r}, {video_path!r})

        print("[Worcester Spin] Step 3: 3D sphere visualization...")
        spin_video_creation({video_path!r}, {sphere_raw_video!r}, {sphere_video!r}, {rotation_processed_csv!r})

        print("[Worcester Spin] Complete!")
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
        print(f"  ERROR: Worcester spin analysis failed:")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-15:]:
                print(f"    {line}")
        return None

    return {
        'rotation_csv': rotation_csv,
        'rotation_processed_csv': rotation_processed_csv,
        'sphere_video': sphere_video,
    }


def check_worcester_available():
    """Check if worcester spin source is available."""
    return os.path.exists(os.path.join(WORCESTER_SRC, 'spin', 'Detection.py'))
