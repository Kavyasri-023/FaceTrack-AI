"""
attendance.py

Attendance business logic module for FaceTrack AI.

This module contains pure business logic for marking attendance,
computing dashboard statistics, retrieving attendance history, and
exporting attendance records to CSV. It builds on top of the database
layer (`database.py`) and does not contain any UI code.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config
import database

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default status assigned to a successful attendance mark.
DEFAULT_ATTENDANCE_STATUS = "Present"


# ---------------------------------------------------------------------------
# Attendance marking
# ---------------------------------------------------------------------------
def mark_attendance_for_student(
    student_id: str, status: str = DEFAULT_ATTENDANCE_STATUS
) -> Dict[str, Any]:
    """
    Mark attendance for a recognized student, enforcing the
    "once per day" rule.

    Date and time are captured automatically at the moment of marking.

    Args:
        student_id: ID of the recognized student.
        status: Attendance status label (default: "Present").

    Returns:
        Dict[str, Any]: A result dictionary with keys:
            - "success" (bool): True if a new attendance record was
              inserted.
            - "already_marked" (bool): True if attendance for today
              already existed (no new record was inserted).
            - "message" (str): Human-readable outcome message.
    """
    if not student_id or not student_id.strip():
        return {
            "success": False,
            "already_marked": False,
            "message": "Invalid student ID provided.",
        }

    student_id = student_id.strip()

    try:
        # Duplicate check is also enforced inside database.mark_attendance,
        # but checking here first lets us return a clear, specific message
        # without relying on a failed insert/exception.
        if database.has_attendance_today(student_id):
            logger.info("Duplicate attendance attempt ignored for '%s'.", student_id)
            return {
                "success": False,
                "already_marked": True,
                "message": "Attendance already marked for today.",
            }

        marked = database.mark_attendance(student_id, status=status)

        if marked:
            logger.info("Attendance marked successfully for '%s'.", student_id)
            return {
                "success": True,
                "already_marked": False,
                "message": "Attendance Marked Successfully!",
            }

        return {
            "success": False,
            "already_marked": False,
            "message": "Failed to mark attendance due to a database error.",
        }

    except Exception as err:  # noqa: BLE001
        logger.error("Unexpected error marking attendance for '%s': %s", student_id, err)
        return {
            "success": False,
            "already_marked": False,
            "message": f"An unexpected error occurred: {err}",
        }


# ---------------------------------------------------------------------------
# Dashboard statistics
# ---------------------------------------------------------------------------
def get_dashboard_summary() -> Dict[str, Any]:
    """
    Retrieve summary statistics for the attendance dashboard.

    Returns:
        Dict[str, Any]: A dictionary with keys:
            - "total_students" (int)
            - "today_attendance_count" (int)
            - "attendance_percentage" (float)
        All values default to 0/0.0 if statistics cannot be computed.
    """
    try:
        return database.get_dashboard_stats()
    except Exception as err:  # noqa: BLE001
        logger.error("Failed to retrieve dashboard statistics: %s", err)
        return {
            "total_students": 0,
            "today_attendance_count": 0,
            "attendance_percentage": 0.0,
        }


# ---------------------------------------------------------------------------
# Attendance history retrieval
# ---------------------------------------------------------------------------
def fetch_attendance_records(
    filter_date: Optional[date] = None,
    student_id: Optional[str] = None,
    name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve attendance records with optional filtering.

    Args:
        filter_date: If provided, restrict results to this exact date.
        student_id: If provided, restrict results to this exact student
            ID.
        name: If provided, performs a partial match against the
            student's name.

    Returns:
        List[Dict[str, Any]]: List of attendance records, ordered most
        recent first. Returns an empty list on error.
    """
    try:
        return database.get_attendance_records(
            filter_date=filter_date, student_id=student_id, name=name
        )
    except Exception as err:  # noqa: BLE001
        logger.error("Failed to fetch attendance records: %s", err)
        return []


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_attendance_to_csv(
    filter_date: Optional[date] = None,
    student_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Optional[Path]:
    """
    Export attendance records (optionally filtered) to a CSV file inside
    `config.ATTENDANCE_DIR`.

    The output filename includes a timestamp to avoid overwriting
    previous exports, e.g. `attendance_export_20260620_153045.csv`.

    Args:
        filter_date: If provided, restrict the export to this exact
            date.
        student_id: If provided, restrict the export to this exact
            student ID.
        name: If provided, performs a partial match against the
            student's name.

    Returns:
        Optional[Path]: Path to the generated CSV file, or None if
        there were no records to export or an error occurred.
    """
    records = fetch_attendance_records(
        filter_date=filter_date, student_id=student_id, name=name
    )

    if not records:
        logger.info("No attendance records found to export.")
        return None

    try:
        config.ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

        dataframe = pd.DataFrame(records)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = config.ATTENDANCE_DIR / f"attendance_export_{timestamp}.csv"

        dataframe.to_csv(file_path, index=False)
        logger.info("Exported %d records to '%s'.", len(records), file_path)
        return file_path

    except (OSError, ValueError) as err:
        logger.error("Failed to export attendance records to CSV: %s", err)
        return None