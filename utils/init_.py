"""
utils package

Shared, reusable helper modules for FaceTrack AI:
    - camera_utils: webcam open/close/frame capture helpers
    - validation_utils: input validation (email, ID, name format)
    - encoding_utils: face_recognition encoding/comparison helpers
    - ui_helpers: reusable Streamlit UI components

Note: the core page modules (register_user.py, train_model.py,
recognize_faces.py, app.py) currently implement this logic inline for
simplicity. These utils are provided as optional, drop-in replacements
if you want to refactor toward a thinner page layer.
"""