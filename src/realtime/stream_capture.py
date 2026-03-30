"""
Camera stream capture with frame buffering.

Supports:
- USB webcam / Continuity Camera (iPhone via USB)
- IP camera streams (RTSP/HTTP URLs)
"""

import cv2
import threading
import time
from collections import deque


class StreamCapture:
    """
    Threaded camera capture that always provides the latest frame.
    Prevents frame accumulation lag by reading continuously in background.
    """

    def __init__(self, source=0, width=1280, height=720, fps=30):
        """
        Args:
            source: Camera index (int) or stream URL (str)
            width: Requested frame width
            height: Requested frame height
            fps: Requested FPS
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {source}\n"
                f"Tips:\n"
                f"  - For Continuity Camera: connect iPhone via USB, try index 0 or 1\n"
                f"  - For webcam: try index 0\n"
                f"  - For IP camera: pass the RTSP/HTTP URL as source"
            )

        # Request resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Read actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Threading
        self._lock = threading.Lock()
        self._frame = None
        self._frame_idx = 0
        self._running = False
        self._thread = None

    def start(self):
        """Start background frame reading thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        # Wait for first frame
        timeout = 3.0
        start = time.time()
        while self._frame is None and time.time() - start < timeout:
            time.sleep(0.01)

        if self._frame is None:
            self.stop()
            raise RuntimeError("Timeout waiting for first frame from camera")

        return self

    def _read_loop(self):
        """Continuously read frames in background."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            with self._lock:
                self._frame = frame
                self._frame_idx += 1

    def read(self):
        """
        Get the latest frame.

        Returns:
            tuple: (frame_index, frame) or (None, None) if no frame available
        """
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame_idx, self._frame.copy()

    def stop(self):
        """Stop capture and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def __repr__(self):
        return (
            f"StreamCapture(source={self.source}, "
            f"{self.width}x{self.height} @ {self.fps:.0f}fps)"
        )
