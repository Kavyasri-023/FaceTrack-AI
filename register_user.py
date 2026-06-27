"""
register_user.py

Streamlit page for registering new students in FaceTrack AI.

Handles:
    - Collecting and validating student details (ID, name, department,
      email) via a Streamlit form.
    - Preventing duplicate student IDs and duplicate emails.
    - Capturing face images from the webcam using OpenCV, saving them
      into `dataset/<student_id>/`.
    - Persisting the student record (including the dataset folder path)
      into MySQL via `database.py`.
"""

import re
from typing import Optional, Tuple

import cv2
import streamlit as st

import config
import database

# ---------------------------------------------------------------------------
# Validation patterns
# ---------------------------------------------------------------------------
STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,20}$")
NAME_PATTERN = re.compile(r"^[A-Za-z\s.'-]{2,100}$")
EMAIL_PATTERN = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")

# Size to which captured face crops are resized before saving, balancing
# image quality for encoding against disk/memory footprint.
FACE_IMAGE_SIZE: Tuple[int, int] = (200, 200)

# Minimum detected face size (in pixels) to filter out tiny/false
# detections during capture.
MIN_FACE_SIZE: Tuple[int, int] = (80, 80)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_inputs(
    student_id: str, name: str, department: str, email: str
) -> Optional[str]:
    """
    Validate all registration form fields.

    Args:
        student_id: Raw student ID input.
        name: Raw full name input.
        department: Raw department input.
        email: Raw email address input.

    Returns:
        Optional[str]: A human-readable error message if validation
        fails, or None if all fields are valid.
    """
    if not student_id or not student_id.strip():
        return "Student ID is required."
    if not STUDENT_ID_PATTERN.match(student_id.strip()):
        return (
            "Student ID must be 2-20 characters long and contain only "
            "letters, numbers, hyphens, or underscores."
        )

    if not name or not name.strip():
        return "Full Name is required."
    if not NAME_PATTERN.match(name.strip()):
        return "Full Name must contain only letters, spaces, and basic punctuation."

    if not department or not department.strip():
        return "Department is required."
    if len(department.strip()) > 100:
        return "Department name is too long."

    if not email or not email.strip():
        return "Email Address is required."
    if not EMAIL_PATTERN.match(email.strip()):
        return "Please enter a valid email address."

    return None


# ---------------------------------------------------------------------------
# Face detection / webcam capture
# ---------------------------------------------------------------------------
def get_face_detector() -> cv2.CascadeClassifier:
    """
    Load OpenCV's built-in Haar cascade frontal face detector.

    Returns:
        cv2.CascadeClassifier: Initialized face detector.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def capture_face_images(
    student_id: str,
    preview_placeholder: "st.delta_generator.DeltaGenerator",
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> int:
    """
    Capture face images from the webcam and save them to
    `dataset/<student_id>/`.

    Images are only saved when a face is detected in the current frame.
    A live preview (with a bounding box around detected faces) is shown
    while capturing.

    Args:
        student_id: ID of the student being registered; used to name
            the dataset subfolder.
        preview_placeholder: Streamlit placeholder used to display the
            live webcam feed.
        progress_bar: Streamlit placeholder used to display capture
            progress.
        status_text: Streamlit placeholder used to display status
            messages during capture.

    Returns:
        int: The number of images successfully captured and saved.
    """
    student_dir = config.DATASET_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    captured = 0
    target = config.IMAGE_CAPTURE_COUNT

    try:
        detector = get_face_detector()
    except Exception as err:  # noqa: BLE001
        status_text.error(f"Failed to load face detector: {err}")
        return 0

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        status_text.error(
            "Could not access the webcam. Please check that it is "
            "connected and that CAMERA_INDEX is correct in your .env file."
        )
        return 0

    try:
        while captured < target:
            ret, frame = cap.read()
            if not ret or frame is None:
                status_text.warning("Failed to read frame from webcam. Retrying...")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE,
            )

            # Draw bounding boxes on a copy of the frame for the live preview.
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            preview_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                caption="Live Webcam Preview",
            )

            if len(faces) > 0:
                # Only capture the first (largest/primary) detected face
                # per frame to avoid accidentally capturing bystanders.
                x, y, w, h = faces[0]
                face_crop = frame[y : y + h, x : x + w]

                try:
                    resized_face = cv2.resize(
                        face_crop, FACE_IMAGE_SIZE, interpolation=cv2.INTER_AREA
                    )
                except cv2.error as resize_err:
                    status_text.warning(f"Skipped a frame (resize error): {resize_err}")
                    continue

                image_path = student_dir / f"img_{captured + 1:03d}.jpg"
                write_success = cv2.imwrite(str(image_path), resized_face)

                if write_success:
                    captured += 1
                    progress_bar.progress(captured / target)
                    status_text.info(f"Captured {captured}/{target} images...")
            else:
                status_text.warning("No face detected. Please face the camera.")

    except Exception as err:  # noqa: BLE001
        status_text.error(f"An error occurred during capture: {err}")
    finally:
        cap.release()

    return captured


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------
def show_register_page() -> None:
    """
    Render the 'Register Student' Streamlit page.

    Collects student details via a form, validates them, checks for
    duplicates, captures face images via webcam, and persists the
    student record to the database.
    """
    st.header("📝 Register Student")
    st.write(
        "Fill in the student's details below, then start registration to "
        "capture face images via webcam."
    )

    with st.form("registration_form", clear_on_submit=False):
        student_id = st.text_input("Student ID", placeholder="e.g. STU001")
        name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
        department = st.text_input("Department", placeholder="e.g. Computer Science")
        email = st.text_input("Email Address", placeholder="e.g. jane.doe@example.com")
        submitted = st.form_submit_button("Start Registration", use_container_width=True)

    if not submitted:
        return

    # --- Validate form inputs -------------------------------------------------
    error_message = validate_inputs(student_id, name, department, email)
    if error_message:
        st.error(error_message)
        return

    student_id = student_id.strip()
    name = name.strip()
    department = department.strip()
    email = email.strip()

    # --- Duplicate checks -------------------------------------------------------
    try:
        if database.student_exists(student_id):
            st.error(f"Student ID '{student_id}' is already registered.")
            
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not verify Student ID uniqueness: {err}")
        return

    try:
        if database.email_exists(email):
            st.error(f"Email '{email}' is already registered with another student.")
            return
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not verify email uniqueness: {err}")
        return

    # --- Webcam capture -----------------------------------------------------
    st.info(
        f"Starting webcam capture. Please look directly at the camera. "
        f"Capturing {config.IMAGE_CAPTURE_COUNT} images..."
    )

    preview_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    captured_count = capture_face_images(
        student_id, preview_placeholder, progress_bar, status_text
    )

    preview_placeholder.empty()

    if captured_count == 0:
        status_text.error(
            "No face images were captured. Registration aborted. "
            "Please check your webcam and try again."
        )
        return

    status_text.success(f"Captured {captured_count} face images successfully.")

    # --- Persist to database -------------------------------------------------
    student_dir = config.DATASET_DIR / student_id

    try:
        success = database.add_student(
            student_id=student_id,
            name=name,
            department=department,
            email=email,
            image_path=str(student_dir),
        )
    except Exception as err:  # noqa: BLE001
        st.error(f"An unexpected error occurred while saving the student: {err}")
        return

    if success:
        st.success(
            f"✅ Student '{name}' (ID: {student_id}) registered successfully "
            f"with {captured_count} images!"
        )
        st.info(
            "Remember to go to the 'Train Model' page to update face "
            "encodings before marking attendance."
        )
        st.balloons()
    else:
        st.error(
            "Failed to save student details to the database. "
            "The captured images have been kept on disk; please retry."
        )