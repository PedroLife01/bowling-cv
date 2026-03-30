"""
Main entry point for Bowling Computer Vision Pipeline.

Orchestrates all phases of the bowling analysis system:
- Phase 1: Lane Detection (boundary detection)
- Phase 2: Ball Detection (trajectory tracking)
- Phase 3: Trajectory Reconstruction (homography-based 2D lane mapping)
- Phase 4: Pin Detection (toppled pin counting)
- Phase 5: Spin/Rotation Analysis (optical flow + Kabsch algorithm)

Can run individual phases or the complete end-to-end pipeline.

Version: 2.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Last Updated: February 6, 2026

Usage:
    # Run complete pipeline (all phases)
    python main.py
    
    # Run individual phases
    python main.py --phase 1              # Lane detection only
    python main.py --phase 2              # Ball detection only
    python main.py --phase 3              # Trajectory reconstruction only
    python main.py --phase 4              # Pin detection only
    python main.py --phase 5              # Spin analysis only

    # Run specific combinations
    python main.py --phase 1 --phase 2    # Lane + Ball
    python main.py --phase 2 --phase 4    # Ball + Pin (requires Phase 1 data)
    python main.py --phase 3 --phase 5    # Trajectory + Spin (requires Phase 1+2 data)
    
    # Process single video
    python main.py --video cropped_test3.mp4
    python main.py --video cropped_test3.mp4 --phase 2
    
    # Run individual module mains directly
    python -m src.lane_detection.main --video cropped_test3.mp4
    python -m src.ball_detection.main --video cropped_test3.mp4
    python -m src.pin_detection.main --video cropped_test3.mp4
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    """
    Main orchestrator for the complete bowling CV pipeline.
    
    Coordinates execution of Phase 1, 2, and 4 based on user arguments.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Bowling Computer Vision Pipeline - Complete Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run all phases on all videos
  python main.py --phase 1                 # Run only Phase 1 (lane detection)
  python main.py --phase 2 --phase 4       # Run Phase 2 and 4
  python main.py --video cropped_test3.mp4 # Process single video (all phases)
  python main.py --video test3.mp4 --phase 2  # Single video, Phase 2 only

Individual Module Entry Points:
  python -m src.lane_detection.main --video cropped_test3.mp4
  python -m src.ball_detection.main --video cropped_test3.mp4
  python -m src.pin_detection.main --video cropped_test3.mp4
        """
    )
    parser.add_argument('--video', type=str, 
                        help='Process single video file (default: process all configured videos)')
    parser.add_argument('--phase', type=int, action='append', choices=[1, 2, 3, 4, 5],
                        help='Run specific phase(s). Can be specified multiple times. Default: run all phases')
    args = parser.parse_args()

    # Determine which phases to run
    if args.phase:
        phases = sorted(set(args.phase))  # Remove duplicates and sort
    else:
        phases = [1, 2, 3, 4, 5]  # Run all phases by default
    
    # Determine which videos to process
    videos = [args.video] if args.video else None
    
    # Print header
    print(f"\n{'#'*80}")
    print(f"# BOWLING COMPUTER VISION PIPELINE v2.0.0")
    print(f"# Phases: {', '.join(f'P{p}' for p in phases)}")
    if args.video:
        print(f"# Video: {args.video}")
    else:
        print(f"# Videos: All configured videos")
    print(f"{'#'*80}\n")
    
    # Run each phase in sequence
    for phase in phases:
        if phase == 1:
            run_phase_1(videos)
        elif phase == 2:
            run_phase_2(videos)
        elif phase == 3:
            run_phase_3(videos)
        elif phase == 4:
            run_phase_4(videos)
        elif phase == 5:
            run_phase_5(videos)
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETE")
    print(f"# Executed Phases: {', '.join(f'Phase {p}' for p in phases)}")
    print(f"{'#'*80}\n")


def run_phase_1(videos=None):
    """
    Run Phase 1: Lane Detection
    
    Parameters:
    -----------
    videos : list, optional
        List of video files to process. If None, uses configured videos.
    """
    print(f"\n{'='*80}")
    print(f"PHASE 1: LANE DETECTION")
    print(f"{'='*80}\n")
    
    from lane_detection.main import main as lane_main
    import sys
    
    # Build command line arguments
    original_argv = sys.argv
    args = ['main.py']
    if videos:
        args.extend(['--video', videos[0]])  # Lane detection processes one video at a time
    
    sys.argv = args
    try:
        lane_main()
    finally:
        sys.argv = original_argv


def run_phase_2(videos=None):
    """
    Run Phase 2: Ball Detection
    
    Parameters:
    -----------
    videos : list, optional
        List of video files to process. If None, uses configured videos.
    """
    print(f"\n{'='*80}")
    print(f"PHASE 2: BALL DETECTION")
    print(f"{'='*80}\n")
    
    from ball_detection.main import main as ball_main
    import sys
    
    # Build command line arguments
    original_argv = sys.argv
    args = ['main.py']
    if videos:
        args.extend(['--video', videos[0]])  # Ball detection processes one video at a time
    
    sys.argv = args
    try:
        ball_main()
    finally:
        sys.argv = original_argv


def run_phase_3(videos=None):
    """
    Run Phase 3: Trajectory Reconstruction

    Parameters:
    -----------
    videos : list, optional
        List of video files to process. If None, uses configured videos.
    """
    print(f"\n{'='*80}")
    print(f"PHASE 3: TRAJECTORY RECONSTRUCTION")
    print(f"{'='*80}\n")

    from trajectory_3d.main import main as trajectory_main
    import sys

    original_argv = sys.argv
    args = ['main.py']
    if videos:
        args.extend(['--video', videos[0]])

    sys.argv = args
    try:
        trajectory_main()
    finally:
        sys.argv = original_argv


def run_phase_4(videos=None):
    """
    Run Phase 4: Pin Detection
    
    Parameters:
    -----------
    videos : list, optional
        List of video files to process. If None, uses configured videos.
    """
    print(f"\n{'='*80}")
    print(f"PHASE 4: PIN DETECTION")
    print(f"{'='*80}\n")
    
    from pin_detection.main import main as pin_main
    import sys
    
    # Build command line arguments
    original_argv = sys.argv
    args = ['main.py']
    if videos:
        args.extend(['--video', videos[0]])  # Pin detection processes one video at a time
    
    sys.argv = args
    try:
        pin_main()
    finally:
        sys.argv = original_argv


def run_phase_5(videos=None):
    """
    Run Phase 5: Spin/Rotation Analysis

    Parameters:
    -----------
    videos : list, optional
        List of video files to process. If None, uses configured videos.
    """
    print(f"\n{'='*80}")
    print(f"PHASE 5: SPIN/ROTATION ANALYSIS")
    print(f"{'='*80}\n")

    from spin_analysis.main import main as spin_main
    import sys

    original_argv = sys.argv
    args = ['main.py']
    if videos:
        args.extend(['--video', videos[0]])

    sys.argv = args
    try:
        spin_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
