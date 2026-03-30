"""
Spin Analysis Module - Phase 5

This module provides spin/rotation analysis for bowling ball videos.
Uses ball trajectory data from Phase 2 (ball detection) to compute
frame-by-frame rotation via optical flow and the Kabsch algorithm.

Pipeline:
1. Detection: Optical flow within ball ROI -> 3D projection -> Kabsch rotation
2. Post-processing: Outlier removal, Gaussian smoothing, cubic interpolation
3. Visualization: 3D sphere video showing cumulative rotation

Version: 1.0.0
Created: March 30, 2026
"""

from .detection import process_spin

__all__ = ['process_spin']
__version__ = '1.0.0'
