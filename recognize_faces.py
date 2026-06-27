"""
recognize_faces.py

Real-time face recognition and automatic attendance marking module for
FaceTrack AI.

Loads trained face encodings from `models/face_encodings.pkl`, opens
the webcam via OpenCV, detects and recognizes faces frame-by-frame,
draws bounding boxes with student details, and automatically marks
attendance (once per day per student) using the helper functions in
`attendance.py`.

The live video feed is shown in a native OpenCV window (press 'q' to
stop), since this avoids the rerun/threading limitations of embedding
a continuous webcam loop directly inside a Streamlit script. The
Streamlit page acts as the launcher and displays a summary once the
session ends.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
import streamlit as st

import attendance
import config
import train_model

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Colors used for bounding boxes (BGR format, as used by OpenCV).
COLOR_RECOGNIZED = (0, 200, 0)      # Green
COLOR_UNKNOWN = (0, 0, 200)         # Red

# Scale factor applied before face detection to speed up processing on
# lower-end hardware; detected coordinates are scaled back up for
# drawing on the full-resolution frame.
FRAME_DETECTION_SCALE = 0.5

# How long (in seconds) an on-screen notification banner stays visible
# after an attendance event occurs.
NOTIFICATION_DISPLAY_SECONDS = 3.0

WINDOW_TITLE = "FaceTrack AI - Recognition (Press 'q' to Stop)"


# ---------------------------------------------------------------------------
# Loading known encodings
# ---------------------------------------------------------------------------
def load_known_encodings() -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load trained face encodings from `models/face_encodings.pkl` and
    flatten them into parallel lists suitable for fast comparison.

    Returns:
        Tuple[List[np.ndarray], List[str], List[str]]: A tuple of
        (encodings, student_ids, names), where each index across the
        three lists corresponds to the same face encoding. Returns
        three empty lists if the pickle file is missing or empty.
    """
    encodings_data = train_model.load_encodings()

    known_encodings: List[np.ndarray] = []
    known_ids: List[str] = []
    known_names: List[str] = []

    for student_id, record in encodings_data.items():
        name = record.get("name", student_id)
        for encoding in record.get("encodings", []):
            known_encodings.append(encoding)
            known_ids.append(student_id)
            known_names.append(name)

    return known_encodings, known_ids, known_names


# ---------------------------------------------------------------------------
# Recognition logic
# ---------------------------------------------------------------------------
def recognize_face(
    face_encoding: np.ndarray,
    known_encodings: List[np.ndarray],
    known_ids: List[str],
    known_names: List[str],
) -> Tuple[Optional[str], str, Optional[float]]:
    """
    Compare a single face encoding against all known encodings and
    return the best match, if any, within the configured threshold.

    Args:
        face_encoding: The 128-d encoding of the face to identify.
        known_encodings: List of all known face encodings.
        known_ids: Student IDs parallel to `known_encodings`.
        known_names: Student names parallel to `known_encodings`.

    Returns:
        Tuple[Optional[str], str, Optional[float]]: (student_id, name,
        distance). If no match is found within
        `config.FACE_MATCH_THRESHOLD`, returns (None, "Unknown User",
        None).
    """
    if not known_encodings:
        return None, "Unknown User", None

    try:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
    except Exception as err:  # noqa: BLE001
        logger.error("Error computing face distances: %s", err)
        return None, "Unknown User", None

    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])

    if best_distance <= config.FACE_MATCH_THRESHOLD:
        return known_ids[best_index], known_names[best_index], best_distance

    return None, "Unknown User", best_distance


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_face_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    student_id: Optional[str],
    name: str,
) -> None:
    """
    Draw a bounding box and identity label(s) on the frame, in place.

    Recognized students get a green box with two text lines
    ("Name: <name>" and "ID: <student_id>"); unrecognized faces get a
    red box labeled "Unknown User".

    Args:
        frame: The BGR image (modified in place) to draw on.
        box: Bounding box as (top, right, bottom, left), as returned by
            `face_recognition.face_locations`.
        student_id: Recognized student's ID, or None if unrecognized.
        name: Recognized student's name, or "Unknown User".
    """
    top, right, bottom, left = box
    recognized = student_id is not None
    color = COLOR_RECOGNIZED if recognized else COLOR_UNKNOWN

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    lines = [f"Name: {name}", f"ID: {student_id}"] if recognized else ["Unknown User"]

    # Draw a filled label background below the box for readability.
    line_height = 22
    label_height = line_height * len(lines) + 10
    cv2.rectangle(
        frame,
        (left, bottom),
        (right, bottom + label_height),
        color,
        cv2.FILLED,
    )

    for i, line in enumerate(lines):
        text_y = bottom + (i + 1) * line_height - 5
        cv2.putText(
            frame,
            line,
            (left + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_notifications(frame: np.ndarray, notifications: List[Dict[str, Any]]) -> None:
    """
    Draw active attendance notification banners at the top of the frame.

    Args:
        frame: The BGR image (modified in place) to draw on.
        notifications: List of dicts with keys "message" and
            "expires_at" (a `time.time()`-style timestamp). Expired
            notifications are skipped but not removed here.
    """
    now = time.time()
    y_offset = 30

    for notification in notifications:
        if notification["expires_at"] < now:
            continue

        cv2.putText(
            frame,
            notification["message"],
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------
def process_frame(
    frame: np.ndarray,
    known_encodings: List[np.ndarray],
    known_ids: List[str],
    known_names: List[str],
    attendance_cache: Dict[str, str],
    notifications: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect, recognize, annotate, and (if applicable) mark attendance for
    all faces in a single frame.

    Args:
        frame: The raw BGR frame captured from the webcam.
        known_encodings: List of all known face encodings.
        known_ids: Student IDs parallel to `known_encodings`.
        known_names: Student names parallel to `known_encodings`.
        attendance_cache: Dict mapping student_id -> outcome message,
            used to avoid repeatedly hitting the database for the same
            student within a single recognition session. Mutated
            in place.
        notifications: List of active on-screen notification banners.
            Mutated in place (new notifications appended).

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: The annotated frame
        (same array, modified in place) and a list of new attendance
        events generated during this frame (each a dict with
        "student_id", "name", and "message").
    """
    new_events: List[Dict[str, Any]] = []

    try:
        small_frame = cv2.resize(
            frame, (0, 0), fx=FRAME_DETECTION_SCALE, fy=FRAME_DETECTION_SCALE
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, known_face_locations=face_locations
        )
    except Exception as err:  # noqa: BLE001
        logger.error("Error during face detection/encoding: %s", err)
        return frame, new_events

    scale_back = 1.0 / FRAME_DETECTION_SCALE

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale bounding box coordinates back up to the full-size frame.
        box = (
            int(top * scale_back),
            int(right * scale_back),
            int(bottom * scale_back),
            int(left * scale_back),
        )

        student_id, name, _distance = recognize_face(
            face_encoding, known_encodings, known_ids, known_names
        )

        draw_face_box(frame, box, student_id, name)

        if student_id is None:
            continue

        # Only attempt to mark attendance once per student per session
        # to avoid hammering the database on every frame.
        if student_id in attendance_cache:
            continue

        result = attendance.mark_attendance_for_student(student_id)
        attendance_cache[student_id] = result["message"]

        if result["success"]:
            notifications.append(
                {
                    "message": f"✅ Attendance Marked Successfully! ({name})",
                    "expires_at": time.time() + NOTIFICATION_DISPLAY_SECONDS,
                }
            )
        elif result["already_marked"]:
            notifications.append(
                {
                    "message": f"ℹ️ {name} already marked today.",
                    "expires_at": time.time() + NOTIFICATION_DISPLAY_SECONDS,
                }
            )

        new_events.append(
            {
                "student_id": student_id,
                "name": name,
                "message": result["message"],
            }
        )

    return frame, new_events


# ---------------------------------------------------------------------------
# Recognition session
# ---------------------------------------------------------------------------
def run_recognition_session() -> Dict[str, Any]:
    """
    Open the webcam and run the real-time recognition + attendance loop
    until the user presses 'q' or the window is closed.

    Returns:
        Dict[str, Any]: A summary dictionary with keys:
            - "success" (bool): False if the session could not start
              (e.g. no trained model or webcam unavailable).
            - "message" (str): Human-readable status/error message.
            - "events" (List[Dict[str, Any]]): All attendance events
              generated during the session.
    """
    known_encodings, known_ids, known_names = load_known_encodings()

    if not known_encodings:
        return {
            "success": False,
            "message": (
                "No trained face encodings found. Please register "
                "students and train the model first."
            ),
            "events": [],
        }

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        return {
            "success": False,
            "message": (
                "Could not access the webcam. Please check that it is "
                "connected and that CAMERA_INDEX is correct in your "
                ".env file."
            ),
            "events": [],
        }

    attendance_cache: Dict[str, str] = {}
    notifications: List[Dict[str, Any]] = []
    session_events: List[Dict[str, Any]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from webcam.")
                continue

            annotated_frame, new_events = process_frame(
                frame,
                known_encodings,
                known_ids,
                known_names,
                attendance_cache,
                notifications,
            )
            session_events.extend(new_events)

            draw_notifications(annotated_frame, notifications)

            cv2.imshow(WINDOW_TITLE, annotated_frame)

            # Exit the loop when 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Exit the loop if the user closes the window directly.
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as err:  # noqa: BLE001
        logger.error("Unexpected error during recognition session: %s", err)
        return {
            "success": False,
            "message": f"An unexpected error occurred: {err}",
            "events": session_events,
        }
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return {
        "success": True,
        "message": "Recognition session ended.",
        "events": session_events,
    }


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------
def show_recognize_page() -> None:
    """
    Render the 'Mark Attendance' Streamlit page.

    Provides a button to launch a live recognition session in a native
    OpenCV window. Once the session ends (user presses 'q' or closes
    the window), a summary of attendance events is displayed.
    """
    st.header("📷 Mark Attendance")
    st.write(
        "Click below to start the webcam. A live video window will "
        "open with face recognition and automatic attendance marking. "
        "Press **'q'** in that window to stop."
    )

    existing_data = train_model.load_encodings()
    if not existing_data:
        st.error(
            "No trained model found. Please register students and "
            "train the model before marking attendance."
        )
        return

    st.info(f"Model currently recognizes **{len(existing_data)}** student(s).")

    if not st.button("▶️ Start Camera & Recognize", use_container_width=True):
        return

    with st.spinner("Recognition session running... check the webcam window."):
        result = run_recognition_session()

    if not result["success"]:
        st.error(result["message"])
        return

    st.success("Recognition session ended.")

    events = result["events"]
    if not events:
        st.info("No attendance events were recorded during this session.")
        return

    st.subheader("Session Summary")
    for event in events:
        st.write(f"**{event['name']}** (ID: {event['student_id']}) — {event['message']}")