"""
utils/encoding_utils.py

Reusable helpers around the `face_recognition` library for generating
and comparing face encodings.
"""

import logging
from typing import Any, List, Optional, Tuple

import face_recognition
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image file for face encoding, handling corrupted/unsupported
    files gracefully.

    Args:
        image_path: Path to the image file.

    Returns:
        Optional[np.ndarray]: The loaded RGB image array, or None on
        failure.
    """
    try:
        return face_recognition.load_image_file(image_path)
    except Exception as err:  # noqa: BLE001
        logger.warning("Failed to load image '%s': %s", image_path, err)
        return None


def get_single_face_encoding(image: np.ndarray) -> Optional[Any]:
    """
    Generate a face encoding for an image, only if exactly one face is
    detected.

    Args:
        image: RGB image array.

    Returns:
        Optional[Any]: The face encoding, or None if zero or multiple
        faces were detected, or on error.
    """
    try:
        face_locations = face_recognition.face_locations(image)
    except Exception as err:  # noqa: BLE001
        logger.warning("Face detection failed: %s", err)
        return None

    if len(face_locations) != 1:
        return None

    try:
        encodings = face_recognition.face_encodings(
            image, known_face_locations=face_locations
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("Face encoding failed: %s", err)
        return None

    return encodings[0] if encodings else None


def detect_all_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect all faces in an image and return their bounding boxes.

    Args:
        image: RGB image array.

    Returns:
        List[Tuple[int, int, int, int]]: List of (top, right, bottom,
        left) bounding boxes. Empty list on error.
    """
    try:
        return face_recognition.face_locations(image)
    except Exception as err:  # noqa: BLE001
        logger.warning("Face detection failed: %s", err)
        return []


def encode_all_faces(
    image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]
) -> List[Any]:
    """
    Generate encodings for all given face locations in an image.

    Args:
        image: RGB image array.
        face_locations: Bounding boxes from `detect_all_faces`.

    Returns:
        List[Any]: List of face encodings (empty list on error).
    """
    try:
        return face_recognition.face_encodings(
            image, known_face_locations=face_locations
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("Face encoding failed: %s", err)
        return []


def find_best_match(
    target_encoding: Any,
    known_encodings: List[Any],
    threshold: float,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Compare a target encoding against a list of known encodings and
    return the index of the closest match within the threshold.

    Args:
        target_encoding: The encoding to identify.
        known_encodings: List of known face encodings to compare against.
        threshold: Maximum distance to be considered a match (lower is
            stricter).

    Returns:
        Tuple[Optional[int], Optional[float]]: (best_index, distance),
        or (None, None) if no known encodings exist or no match is
        within the threshold.
    """
    if not known_encodings:
        return None, None

    try:
        distances = face_recognition.face_distance(known_encodings, target_encoding)
    except Exception as err:  # noqa: BLE001
        logger.error("Error computing face distances: %s", err)
        return None, None

    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])

    if best_distance <= threshold:
        return best_index, best_distance

    return None, best_distance