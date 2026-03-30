"""
Bowling Analysis Project

Computer vision system for analyzing bowling videos using traditional CV techniques.
No machine learning required - pure algorithmic approach using OpenCV.

Modules:
--------
- lane_detection: Phase 1 - Detect lane boundaries (top, bottom, left, right)
- ball_detection: Phase 2 - Track ball trajectory from foul line to pins
- trajectory_3d: Phase 3 - Reconstruct trajectory in lane coordinates via homography
- pin_detection: Phase 4 - Count toppled pins using frame differencing
- spin_analysis: Phase 5 - Analyze ball spin/rotation via optical flow + Kabsch

Version: 1.0.0
Authors: Mohammad Umayr Romshoo, Mohammad Ammar Mughees
Course: Image Analysis and Computer Vision
Date: February 2026
"""

__version__ = "1.0.0"
__authors__ = ["Mohammad Umayr Romshoo", "Mohammad Ammar Mughees"]
__course__ = "Image Analysis and Computer Vision"

# Import phase modules
try:
    from . import lane_detection
except ImportError:
    lane_detection = None

try:
    from . import ball_detection
except ImportError:
    ball_detection = None

try:
    from . import trajectory_3d
except ImportError:
    trajectory_3d = None

try:
    from . import pin_detection
except ImportError:
    pin_detection = None

try:
    from . import spin_analysis
except ImportError:
    spin_analysis = None

# Export public API
__all__ = [
    'lane_detection',
    'ball_detection',
    'trajectory_3d',
    'pin_detection',
    'spin_analysis',
    '__version__',
]


def get_project_info():
    """
    Get project information summary.
    
    Returns:
    --------
    dict : Project metadata and module status
    """
    return {
        'name': 'Bowling Analysis Project',
        'version': __version__,
        'authors': __authors__,
        'course': __course__,
        'modules': {
            'lane_detection': {
                'available': lane_detection is not None,
                'status': 'Complete' if lane_detection else 'Not installed',
                'description': 'Phase 1: Lane boundary detection'
            },
            'ball_detection': {
                'available': ball_detection is not None,
                'status': 'Complete' if ball_detection else 'Not installed',
                'description': 'Phase 2: Ball trajectory tracking'
            },
            'pin_detection': {
                'available': pin_detection is not None,
                'status': 'Complete' if pin_detection else 'Not installed',
                'description': 'Phase 4: Pin counting via frame differencing'
            }
        }
    }


def print_project_info():
    """Print formatted project information."""
    info = get_project_info()
    
    print("\n" + "="*80)
    print(f"{info['name']} v{info['version']}")
    print("="*80)
    print(f"\nAuthors: {', '.join(info['authors'])}")
    print(f"Course:  {info['course']}")
    
    print("\n📦 Available Modules:")
    print("-" * 80)
    
    for module_name, module_info in info['modules'].items():
        status_icon = "✅" if module_info['available'] else "❌"
        print(f"  {status_icon} {module_name:<20} - {module_info['description']}")
        print(f"     Status: {module_info['status']}")
    
    print("="*80 + "\n")


# Convenience functions for running pipelines
def run_lane_detection(video_name=None):
    """
    Run Phase 1: Lane Detection pipeline.
    
    Parameters:
    -----------
    video_name : str, optional
        Name of video to process. If None, processes all configured videos.
        
    Returns:
    --------
    dict : Boundary detection results
    """
    if lane_detection is None:
        raise ImportError("lane_detection module not available")
    
    from .lane_detection import LaneDetector
    
    if video_name:
        # Process single video
        detector = LaneDetector(video_name)
        return detector.detect_all()
    else:
        # Process all configured videos
        from .lane_detection import config
        results = {}
        for vid in config.VIDEO_NAMES:
            detector = LaneDetector(vid)
            results[vid] = detector.detect_all_boundaries()
        return results


def run_ball_detection(video_name=None):
    """
    Run Phase 2: Ball Detection pipeline.
    
    Parameters:
    -----------
    video_name : str, optional
        Name of video to process. If None, processes all configured videos.
        
    Returns:
    --------
    dict : Ball trajectory results
    """
    if ball_detection is None:
        raise ImportError("ball_detection module not available")
    
    from .ball_detection import main
    
    # Run ball detection pipeline
    return main.main()


def run_pin_detection(video_name):
    """
    Run Phase 4: Pin Detection pipeline.
    
    Parameters:
    -----------
    video_name : str
        Name of video to process
        
    Returns:
    --------
    dict : Pin detection results
    """
    if pin_detection is None:
        raise ImportError("pin_detection module not available")
    
    from .pin_detection import detect_pins_in_video
    
    return detect_pins_in_video(video_name)


def run_complete_pipeline(video_name):
    """
    Run complete analysis pipeline: Phases 1, 2, and 4.
    
    Parameters:
    -----------
    video_name : str
        Name of video to process
        
    Returns:
    --------
    dict : Complete analysis results from all phases
    """
    print("\n" + "="*80)
    print(f"COMPLETE PIPELINE: {video_name}")
    print("="*80)
    
    results = {}
    
    # Phase 1: Lane Detection
    print("\n🎯 Running Phase 1: Lane Detection...")
    try:
        results['lane_detection'] = run_lane_detection(video_name)
        print("✅ Phase 1 complete!")
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        results['lane_detection'] = None
    
    # Phase 2: Ball Detection
    print("\n🎯 Running Phase 2: Ball Detection...")
    try:
        results['ball_detection'] = run_ball_detection(video_name)
        print("✅ Phase 2 complete!")
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        results['ball_detection'] = None
    
    # Phase 4: Pin Detection
    print("\n🎯 Running Phase 4: Pin Detection...")
    try:
        results['pin_detection'] = run_pin_detection(video_name)
        print("✅ Phase 4 complete!")
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        results['pin_detection'] = None
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    # Print summary
    print("\n📊 Results Summary:")
    print("-" * 80)
    
    if results['lane_detection']:
        print("✅ Lane boundaries detected")
    
    if results['ball_detection']:
        print("✅ Ball trajectory tracked")
    
    if results['pin_detection']:
        pin_result = results['pin_detection']['result']
        toppled = results['pin_detection']['toppled_pins']
        print(f"✅ Pins analyzed: {pin_result} ({toppled}/10 toppled)")
    
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Print project info when module is run directly
    print_project_info()
    
    # Usage examples
    print("📝 Usage Examples:")
    print("-" * 80)
    print("""
    # Import project
    import src
    
    # Check available modules
    src.print_project_info()
    
    # Run individual phases
    lane_results = src.run_lane_detection('cropped_test3')
    ball_results = src.run_ball_detection('cropped_test3')
    pin_results = src.run_pin_detection('cropped_test3')
    
    # Run complete pipeline
    results = src.run_complete_pipeline('cropped_test3')
    """)
    print("="*80 + "\n")