"""
utils/camera_utils.py

Reusable webcam helper functions built on OpenCV, shared across the
registration and recognition workflows.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def open_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    """
    Open a webcam device by index.

    Args:
        camera_index: Index of the webcam device (0 = default).

    Returns:
        Optional[cv2.VideoCapture]: The opened capture object, or None
        if the camera could not be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open webcam at index %d.", camera_index)
        return None
    return cap


def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
    """
    Safely release a webcam capture object.

    Args:
        cap: The capture object to release (no-op if None).
    """
    if cap is not None:
        cap.release()


def capture_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Read a single frame from an open webcam capture.

    Args:
        cap: An open `cv2.VideoCapture` object.

    Returns:
        Optional[np.ndarray]: The captured BGR frame, or None if the
        read failed.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.warning("Failed to read frame from webcam.")
        return None
    return frame


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a frame/image to the given (width, height).

    Args:
        frame: The BGR image to resize.
        size: Target (width, height) in pixels.

    Returns:
        np.ndarray: The resized image.
    """
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def load_haar_cascade() -> cv2.CascadeClassifier:
    """
    Load OpenCV's built-in Haar cascade frontal face detector.

    Returns:
        cv2.CascadeClassifier: Initialized face detector.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces_haar(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    min_size: Tuple[int, int] = (80, 80),
):
    """
    Detect faces in a frame using a Haar cascade classifier.

    Args:
        frame: BGR image to search for faces.
        detector: A loaded `cv2.CascadeClassifier`.
        min_size: Minimum face size (width, height) in pixels.

    Returns:
        Sequence of (x, y, w, h) bounding boxes for detected faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
    )