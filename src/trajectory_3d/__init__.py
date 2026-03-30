"""
Trajectory 3D Reconstruction Module - Phase 3

Reconstructs the ball trajectory in real-world lane coordinates using
homography-based perspective transformation. Reads boundary data from
Phase 1 (lane detection) and ball trajectory from Phase 2 (ball detection),
then maps pixel positions to lane-space coordinates (0-106 x 0-1829 inches).

Current Implementation:
- Boundary-to-corner conversion (from boundary_data.json intersections)
- Per-frame homography computation
- Ball position transformation via perspective mapping
- Savitzky-Golay trajectory smoothing
- Overhead trajectory visualization video

Version: 1.0.0
Authors: Pedro Roriz
Created: March 30, 2026
"""

from . import config
from .reconstruction import process_reconstruction

__all__ = ['config', 'process_reconstruction']
__version__ = '1.0.0'
