"""
config.py

Central configuration module for FaceTrack AI.

Loads environment variables from a `.env` file using python-dotenv,
defines MySQL connection settings, filesystem paths, and tunable
recognition/capture constants. Also ensures that all required
project directories exist at import time.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
# Loads variables defined in a `.env` file (in the project root) into the
# process environment. If `.env` is missing, os.getenv() fallbacks below
# are used instead, so the app can still run with sensible defaults.
load_dotenv()


# ---------------------------------------------------------------------------
# Base project directory
# ---------------------------------------------------------------------------
# Resolves to the directory containing this config.py file, i.e. the
# project root (FaceTrackAI/). Used as the anchor for all other paths.
BASE_DIR: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# MySQL database configuration
# ---------------------------------------------------------------------------
# All credentials are pulled from environment variables so that no
# sensitive values are hard-coded into the source code.
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
DB_USER: str = os.getenv("DB_USER", "root")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_NAME: str = os.getenv("DB_NAME", "facetrack_ai")


# ---------------------------------------------------------------------------
# Filesystem directory constants
# ---------------------------------------------------------------------------
# All paths use pathlib.Path for cross-platform compatibility and are
# defined relative to BASE_DIR so the project is portable regardless of
# where it is cloned/run from.
DATASET_DIR: Path = BASE_DIR / "dataset"
MODELS_DIR: Path = BASE_DIR / "models"
ATTENDANCE_DIR: Path = BASE_DIR / "attendance_records"
ASSETS_DIR: Path = BASE_DIR / "assets"
SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"

# Full path to the pickle file that stores trained face encodings.
FACE_ENCODINGS_PATH: Path = MODELS_DIR / "face_encodings.pkl"


# ---------------------------------------------------------------------------
# Configurable application constants
# ---------------------------------------------------------------------------
# FACE_MATCH_THRESHOLD: lower values make face matching stricter
# (fewer false positives, more false negatives). The face_recognition
# library typically uses ~0.6 as a reasonable default.
FACE_MATCH_THRESHOLD: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))

# IMAGE_CAPTURE_COUNT: number of face images captured per student
# during registration (recommended range: 20-30).
IMAGE_CAPTURE_COUNT: int = int(os.getenv("IMAGE_CAPTURE_COUNT", "25"))

# CAMERA_INDEX: index of the webcam device to use with OpenCV
# (0 is typically the default/built-in webcam).
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------
def _ensure_directories_exist() -> None:
    """
    Create all required project directories if they do not already exist.

    This is called automatically when the module is imported, so any
    other module that imports `config` can safely assume these
    directories are present before performing file I/O.
    """
    required_dirs = (
        DATASET_DIR,
        MODELS_DIR,
        ATTENDANCE_DIR,
        ASSETS_DIR,
        SCREENSHOTS_DIR,
    )

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


# Run directory bootstrap immediately on import.
_ensure_directories_exist()