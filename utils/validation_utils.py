"""
utils/validation_utils.py

Reusable input validation helpers for FaceTrack AI.
"""

import re
from typing import Optional

STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,20}$")
NAME_PATTERN = re.compile(r"^[A-Za-z\s.'-]{2,100}$")
EMAIL_PATTERN = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")

MAX_DEPARTMENT_LENGTH = 100


def validate_student_id(student_id: str) -> Optional[str]:
    """
    Validate a student ID string.

    Returns:
        Optional[str]: Error message, or None if valid.
    """
    if not student_id or not student_id.strip():
        return "Student ID is required."
    if not STUDENT_ID_PATTERN.match(student_id.strip()):
        return (
            "Student ID must be 2-20 characters long and contain only "
            "letters, numbers, hyphens, or underscores."
        )
    return None


def validate_name(name: str) -> Optional[str]:
    """
    Validate a full name string.

    Returns:
        Optional[str]: Error message, or None if valid.
    """
    if not name or not name.strip():
        return "Full Name is required."
    if not NAME_PATTERN.match(name.strip()):
        return "Full Name must contain only letters, spaces, and basic punctuation."
    return None


def validate_department(department: str) -> Optional[str]:
    """
    Validate a department string.

    Returns:
        Optional[str]: Error message, or None if valid.
    """
    if not department or not department.strip():
        return "Department is required."
    if len(department.strip()) > MAX_DEPARTMENT_LENGTH:
        return "Department name is too long."
    return None


def validate_email(email: str) -> Optional[str]:
    """
    Validate an email address string.

    Returns:
        Optional[str]: Error message, or None if valid.
    """
    if not email or not email.strip():
        return "Email Address is required."
    if not EMAIL_PATTERN.match(email.strip()):
        return "Please enter a valid email address."
    return None


def validate_registration_form(
    student_id: str, name: str, department: str, email: str
) -> Optional[str]:
    """
    Run all registration field validators in sequence.

    Returns:
        Optional[str]: The first error message encountered, or None if
        all fields are valid.
    """
    for validator, value in (
        (validate_student_id, student_id),
        (validate_name, name),
        (validate_department, department),
        (validate_email, email),
    ):
        error = validator(value)
        if error:
            return error
    return None