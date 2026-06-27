"""
train_model.py

Face encoding training module for FaceTrack AI.

Reads all student image folders from `dataset/<student_id>/`, generates
face encodings using the `face_recognition` library, and stores the
trained data as a pickle file at `models/face_encodings.pkl` with the
structure:

    {
        "student_id": {
            "name": "Student Name",
            "encodings": [encoding1, encoding2, ...]
        }
    }

Provides a Streamlit page with progress reporting and a "Retrain Model"
action that overwrites any existing trained data.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import face_recognition
import streamlit as st

import config
import database

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image file extensions considered valid for training.
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------
def get_student_folders() -> List[Path]:
    """
    Retrieve all student dataset folders under `config.DATASET_DIR`.

    Returns:
        List[Path]: A sorted list of subdirectories, one per student,
        each expected to be named after a student ID. Returns an empty
        list if the dataset directory does not exist or has no
        subfolders.
    """
    if not config.DATASET_DIR.exists():
        logger.warning("Dataset directory does not exist: %s", config.DATASET_DIR)
        return []

    folders = [item for item in config.DATASET_DIR.iterdir() if item.is_dir()]
    return sorted(folders, key=lambda path: path.name)


def get_image_files(student_dir: Path) -> List[Path]:
    """
    List all valid image files within a student's dataset folder.

    Args:
        student_dir: Path to the student's dataset folder.

    Returns:
        List[Path]: Sorted list of image file paths with a recognized
        extension.
    """
    images = [
        item
        for item in student_dir.iterdir()
        if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.name)


# ---------------------------------------------------------------------------
# Encoding generation
# ---------------------------------------------------------------------------
def process_student_images(
    student_id: str, image_paths: List[Path]
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Generate face encodings for all images belonging to one student.

    Images are skipped (and counted separately) if:
        - The file cannot be read/decoded (corrupted or unsupported).
        - No face is detected in the image.
        - More than one face is detected in the image (ambiguous data).

    Args:
        student_id: ID of the student whose images are being processed
            (used only for logging).
        image_paths: List of image file paths to process.

    Returns:
        Tuple[List[Any], Dict[str, int]]: A list of successfully
        generated face encodings, and a stats dictionary with keys
        "processed", "successful", "skipped_no_face",
        "skipped_multiple_faces", and "skipped_error".
    """
    encodings: List[Any] = []
    stats = {
        "processed": 0,
        "successful": 0,
        "skipped_no_face": 0,
        "skipped_multiple_faces": 0,
        "skipped_error": 0,
    }

    for image_path in image_paths:
        stats["processed"] += 1

        try:
            image = face_recognition.load_image_file(str(image_path))
        except Exception as err:  # noqa: BLE001 - corrupted/unsupported file
            logger.warning(
                "Skipping corrupted/unreadable image '%s' for student '%s': %s",
                image_path.name,
                student_id,
                err,
            )
            stats["skipped_error"] += 1
            continue

        try:
            face_locations = face_recognition.face_locations(image)
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Face detection failed on '%s' for student '%s': %s",
                image_path.name,
                student_id,
                err,
            )
            stats["skipped_error"] += 1
            continue

        if len(face_locations) == 0:
            logger.info(
                "No face detected in '%s' for student '%s'; skipping.",
                image_path.name,
                student_id,
            )
            stats["skipped_no_face"] += 1
            continue

        if len(face_locations) > 1:
            logger.info(
                "Multiple faces (%d) detected in '%s' for student '%s'; skipping.",
                len(face_locations),
                image_path.name,
                student_id,
            )
            stats["skipped_multiple_faces"] += 1
            continue

        try:
            face_encodings = face_recognition.face_encodings(
                image, known_face_locations=face_locations
            )
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Encoding generation failed on '%s' for student '%s': %s",
                image_path.name,
                student_id,
                err,
            )
            stats["skipped_error"] += 1
            continue

        if not face_encodings:
            stats["skipped_error"] += 1
            continue

        encodings.append(face_encodings[0])
        stats["successful"] += 1

    return encodings, stats


# ---------------------------------------------------------------------------
# Pickle persistence
# ---------------------------------------------------------------------------
def save_encodings(encodings_data: Dict[str, Dict[str, Any]]) -> bool:
    """
    Save the trained encodings dictionary to the pickle file, overwriting
    any existing file at that path.

    Args:
        encodings_data: Dictionary mapping student_id to a dict with
            "name" and "encodings" keys.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.FACE_ENCODINGS_PATH, "wb") as file_handle:
            pickle.dump(encodings_data, file_handle)
        logger.info("Saved face encodings to '%s'.", config.FACE_ENCODINGS_PATH)
        return True
    except (OSError, pickle.PickleError) as err:
        logger.error("Failed to save face encodings: %s", err)
        return False


def load_encodings() -> Dict[str, Dict[str, Any]]:
    """
    Load previously trained encodings from the pickle file, if present.

    Returns:
        Dict[str, Dict[str, Any]]: The stored encodings dictionary, or
        an empty dictionary if the file does not exist or cannot be
        read.
    """
    if not config.FACE_ENCODINGS_PATH.exists():
        return {}

    try:
        with open(config.FACE_ENCODINGS_PATH, "rb") as file_handle:
            return pickle.load(file_handle)
    except (OSError, pickle.PickleError) as err:
        logger.error("Failed to load existing face encodings: %s", err)
        return {}


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------
def train_model(progress_callback=None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Train (or retrain) the face recognition model from all student
    dataset folders, overwriting any existing pickle file.

    Args:
        progress_callback: Optional callable invoked as
            `progress_callback(current_index, total, student_id)` after
            each student is processed, for UI progress reporting.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]: The final
        encodings dictionary that was saved, and an aggregate stats
        dictionary summarizing totals across all students (keys:
        "total_students", "trained_students", "skipped_students",
        "total_images_processed", "total_images_successful",
        "total_skipped_no_face", "total_skipped_multiple_faces",
        "total_skipped_error").
    """
    aggregate_stats = {
        "total_students": 0,
        "trained_students": 0,
        "skipped_students": 0,
        "total_images_processed": 0,
        "total_images_successful": 0,
        "total_skipped_no_face": 0,
        "total_skipped_multiple_faces": 0,
        "total_skipped_error": 0,
    }

    student_folders = get_student_folders()
    aggregate_stats["total_students"] = len(student_folders)

    if not student_folders:
        logger.warning("No student dataset folders found; nothing to train.")
        return {}, aggregate_stats

    encodings_data: Dict[str, Dict[str, Any]] = {}

    for index, student_dir in enumerate(student_folders, start=1):
        student_id = student_dir.name
        image_paths = get_image_files(student_dir)

        if not image_paths:
            logger.warning(
                "Student '%s' has no images; skipping.", student_id
            )
            aggregate_stats["skipped_students"] += 1
            if progress_callback:
                progress_callback(index, len(student_folders), student_id)
            continue

        student_record = database.get_student_by_id(student_id)
        student_name = student_record["name"] if student_record else student_id

        if student_record is None:
            logger.warning(
                "No database record found for student '%s'; using ID as name.",
                student_id,
            )

        encodings, stats = process_student_images(student_id, image_paths)

        aggregate_stats["total_images_processed"] += stats["processed"]
        aggregate_stats["total_images_successful"] += stats["successful"]
        aggregate_stats["total_skipped_no_face"] += stats["skipped_no_face"]
        aggregate_stats["total_skipped_multiple_faces"] += stats["skipped_multiple_faces"]
        aggregate_stats["total_skipped_error"] += stats["skipped_error"]

        if encodings:
            encodings_data[student_id] = {
                "name": student_name,
                "encodings": encodings,
            }
            aggregate_stats["trained_students"] += 1
        else:
            logger.warning(
                "No usable face encodings generated for student '%s'.", student_id
            )
            aggregate_stats["skipped_students"] += 1

        if progress_callback:
            progress_callback(index, len(student_folders), student_id)

    save_encodings(encodings_data)
    return encodings_data, aggregate_stats


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------
def show_train_page() -> None:
    """
    Render the 'Train Model' Streamlit page.

    Displays current model status and provides a "Retrain Model" button
    that re-processes all dataset images and overwrites
    `models/face_encodings.pkl`.
    """
    st.header("🧠 Train Model")
    st.write(
        "Generate or refresh face encodings from all registered students' "
        "images. Run this every time you register a new student."
    )

    # --- Current model status -------------------------------------------------
    existing_data = load_encodings()
    if existing_data:
        st.info(
            f"Current model contains encodings for **{len(existing_data)}** "
            f"student(s)."
        )
    else:
        st.warning("No trained model found yet. Train the model to get started.")

    # --- Pre-training validation -----------------------------------------------
    student_folders = get_student_folders()
    if not student_folders:
        st.error(
            "No student dataset folders found under "
            f"'{config.DATASET_DIR}'. Please register at least one "
            "student before training."
        )
        return

    st.write(f"Found **{len(student_folders)}** student folder(s) ready for training.")

    if not st.button("🔁 Retrain Model", use_container_width=True):
        return

    # --- Run training with progress reporting -----------------------------------
    progress_bar = st.progress(0)
    status_text = st.empty()

    def _update_progress(current: int, total: int, student_id: str) -> None:
        """Update the Streamlit progress bar and status text."""
        progress_bar.progress(current / total)
        status_text.info(f"Processing student '{student_id}' ({current}/{total})...")

    try:
        encodings_data, stats = train_model(progress_callback=_update_progress)
    except Exception as err:  # noqa: BLE001
        st.error(f"An unexpected error occurred during training: {err}")
        return

    status_text.empty()

    if not encodings_data:
        st.error(
            "Training completed, but no usable face encodings were "
            "generated. Please check that dataset images contain clear, "
            "single faces."
        )
        return

    # --- Summary report -------------------------------------------------------
    st.success(
        f"✅ Model trained successfully! Encodings saved for "
        f"**{stats['trained_students']}** of **{stats['total_students']}** "
        f"student(s)."
    )

    with st.expander("📊 Training Details"):
        st.write(f"- Students processed: **{stats['total_students']}**")
        st.write(f"- Students successfully trained: **{stats['trained_students']}**")
        st.write(f"- Students skipped (no usable images): **{stats['skipped_students']}**")
        st.write(f"- Total images processed: **{stats['total_images_processed']}**")
        st.write(f"- Images successfully encoded: **{stats['total_images_successful']}**")
        st.write(f"- Images skipped (no face detected): **{stats['total_skipped_no_face']}**")
        st.write(
            f"- Images skipped (multiple faces detected): "
            f"**{stats['total_skipped_multiple_faces']}**"
        )
        st.write(f"- Images skipped (errors/corrupted): **{stats['total_skipped_error']}**")