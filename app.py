"""
app.py

Main Streamlit entry point for FaceTrack AI – Smart Attendance System
with Face Recognition.

Wires together all project modules (config, database, register_user,
train_model, recognize_faces, attendance) behind a sidebar-navigated,
dark-blue-themed interface with the following pages:

    - Home
    - Register Student
    - Train Model
    - Mark Attendance
    - Attendance Report
    - About Project
"""

import logging
from datetime import date

import pandas as pd
import streamlit as st

import attendance
import config
import database
import recognize_faces
import register_user
import train_model

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FaceTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation options, each mapped to an icon for display.
NAV_PAGES = {
    "Home": "🏠",
    "Register Student": "📝",
    "Train Model": "🧠",
    "Mark Attendance": "📷",
    "Attendance Report": "📊",
    "About Project": "ℹ️",
}


# ---------------------------------------------------------------------------
# Theming
# ---------------------------------------------------------------------------
def apply_custom_theme() -> None:
    """
    Inject custom CSS to apply a modern dark-blue theme across the app,
    including styled dashboard cards, sidebar, buttons, and status
    indicators.
    """
    st.markdown(
        """
        <style>
            /* --- Overall app background --- */
            .stApp {
                background-color: #0b132b;
                color: #e8eaf0;
            }

            /* --- Sidebar --- */
            section[data-testid="stSidebar"] {
                background-color: #0f1b3d;
                border-right: 1px solid #1f2f5c;
            }
            section[data-testid="stSidebar"] * {
                color: #e8eaf0 !important;
            }

            /* --- Headings --- */
            h1, h2, h3 {
                color: #5bc0ff;
            }

            /* --- Buttons --- */
            .stButton > button {
                background-color: #1b3a6b;
                color: #ffffff;
                border: 1px solid #2e5aa8;
                border-radius: 8px;
                padding: 0.5em 1.2em;
                font-weight: 600;
                transition: background-color 0.2s ease-in-out;
            }
            .stButton > button:hover {
                background-color: #2e5aa8;
                border-color: #5bc0ff;
                color: #ffffff;
            }

            /* --- Dashboard cards --- */
            .ft-card {
                background-color: #122451;
                border: 1px solid #1f2f5c;
                border-radius: 12px;
                padding: 1.2em;
                text-align: center;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
            }
            .ft-card h2 {
                margin: 0;
                font-size: 2em;
                color: #5bc0ff;
            }
            .ft-card p {
                margin: 0.3em 0 0 0;
                color: #aab4cc;
                font-size: 0.95em;
            }

            /* --- Inputs --- */
            .stTextInput input, .stSelectbox div[data-baseweb="select"] {
                background-color: #122451;
                color: #e8eaf0;
                border-radius: 6px;
            }

            /* --- Dataframes/tables --- */
            div[data-testid="stDataFrame"] {
                border-radius: 8px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(value: str, label: str) -> str:
    """
    Build the HTML markup for a single dashboard stat card.

    Args:
        value: The headline value to display (e.g. "42").
        label: The caption describing the value (e.g. "Total Students").

    Returns:
        str: HTML string for the rendered card.
    """
    return f"""
        <div class="ft-card">
            <h2>{value}</h2>
            <p>{label}</p>
        </div>
    """


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------
def initialize_database() -> bool:
    """
    Initialize required database tables, surfacing any failure to the
    user via the UI instead of crashing the app.

    Returns:
        bool: True if initialization succeeded, False otherwise.
    """
    try:
        database.init_db()
        return True
    except Exception as err:  # noqa: BLE001
        st.error(
            "⚠️ Could not connect to the database. Please verify your "
            f"MySQL settings in `.env` and that the server is running.\n\n"
            f"Details: {err}"
        )
        logger.error("Database initialization failed: %s", err)
        return False


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------
def show_home_page() -> None:
    """Render the Home page with a project intro and live dashboard stats."""
    st.title("🎯 FaceTrack AI")
    st.subheader("Smart Attendance System with Face Recognition")
    st.write(
        "Automate attendance tracking using real-time facial "
        "recognition — no manual roll calls, no proxy attendance, "
        "fully digital records."
    )

    st.markdown("### 📊 Live Overview")

    try:
        stats = attendance.get_dashboard_summary()
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load dashboard statistics: {err}")
        stats = {
            "total_students": 0,
            "today_attendance_count": 0,
            "attendance_percentage": 0.0,
        }

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            render_card(str(stats["total_students"]), "Total Registered Students"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_card(str(stats["today_attendance_count"]), "Today's Attendance"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_card(f"{stats['attendance_percentage']}%", "Attendance Percentage"),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.write(
        "1. **Register Student** — capture face images and save student details.\n"
        "2. **Train Model** — generate face encodings from registered students.\n"
        "3. **Mark Attendance** — open the webcam and recognize faces in real time.\n"
        "4. **Attendance Report** — review, search, filter, and export records."
    )


# ---------------------------------------------------------------------------
# Page: Attendance Report
# ---------------------------------------------------------------------------
def show_report_page() -> None:
    """
    Render the Attendance Report page with dashboard cards, search/filter
    controls, attendance history, and CSV export.
    """
    st.title("📊 Attendance Report")

    try:
        stats = attendance.get_dashboard_summary()
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load dashboard statistics: {err}")
        stats = {
            "total_students": 0,
            "today_attendance_count": 0,
            "attendance_percentage": 0.0,
        }

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            render_card(str(stats["total_students"]), "Total Registered Students"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_card(str(stats["today_attendance_count"]), "Today's Attendance"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_card(f"{stats['attendance_percentage']}%", "Attendance Percentage"),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🔍 Search & Filter")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        search_id = st.text_input("Search by Student ID", placeholder="e.g. STU001")
    with filter_col2:
        search_name = st.text_input("Search by Name", placeholder="e.g. Jane Doe")
    with filter_col3:
        use_date_filter = st.checkbox("Filter by Date")
        selected_date = st.date_input("Date", value=date.today()) if use_date_filter else None

    try:
        records = attendance.fetch_attendance_records(
            filter_date=selected_date,
            student_id=search_id.strip() or None,
            name=search_name.strip() or None,
        )
    except Exception as err:  # noqa: BLE001
        st.error(f"Failed to fetch attendance records: {err}")
        records = []

    st.markdown("### 🗒️ Attendance History")

    if not records:
        st.info("No attendance records found for the selected criteria.")
        return

    dataframe = pd.DataFrame(records)
    st.dataframe(dataframe, use_container_width=True, hide_index=True)

    st.markdown("### ⬇️ Export")
    if st.button("Export to CSV", use_container_width=False):
        try:
            export_path = attendance.export_attendance_to_csv(
                filter_date=selected_date,
                student_id=search_id.strip() or None,
                name=search_name.strip() or None,
            )
        except Exception as err:  # noqa: BLE001
            st.error(f"Failed to export attendance: {err}")
            export_path = None

        if export_path is not None:
            st.success(f"✅ Exported to `{export_path}`")
            try:
                with open(export_path, "rb") as file_handle:
                    st.download_button(
                        label="Download CSV",
                        data=file_handle.read(),
                        file_name=export_path.name,
                        mime="text/csv",
                    )
            except OSError as err:
                st.warning(f"Export saved, but could not prepare download: {err}")
        else:
            st.warning("No records were available to export.")


# ---------------------------------------------------------------------------
# Page: About Project
# ---------------------------------------------------------------------------
def show_about_page() -> None:
    """Render the About Project page with tech stack and project details."""
    st.title("ℹ️ About FaceTrack AI")
    st.write(
        "**FaceTrack AI** is an intelligent attendance management system "
        "that uses facial recognition to automatically identify students "
        "and mark attendance in real time through a webcam — eliminating "
        "manual roll calls and preventing proxy attendance."
    )

    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        """
        - **Frontend:** Streamlit
        - **Backend:** Python
        - **Face Detection:** OpenCV
        - **Face Recognition:** `face_recognition` (built on dlib)
        - **Database:** MySQL
        - **Data Processing:** Pandas
        - **Model Storage:** Pickle (`.pkl`)
        - **Configuration:** python-dotenv
        """
    )

    st.markdown("### 📂 Core Modules")
    st.markdown(
        """
        - `register_user.py` — Student registration & face capture
        - `train_model.py` — Face encoding generation & training
        - `recognize_faces.py` — Real-time recognition & attendance marking
        - `attendance.py` — Attendance business logic & reporting
        - `database.py` — MySQL data access layer
        - `config.py` — Centralized configuration
        """
    )

    st.markdown("### 🏆 Built For")
    st.write("Hackathon demonstration of a practical, end-to-end EdTech AI solution.")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    """
    Render the sidebar navigation menu.

    Returns:
        str: The label of the page selected by the user.
    """
    with st.sidebar:
        st.markdown("## 🎯 FaceTrack AI")
        st.caption("Smart Attendance System")
        st.markdown("---")

        selected_page = st.radio(
            "Navigation",
            options=list(NAV_PAGES.keys()),
            format_func=lambda page: f"{NAV_PAGES[page]}  {page}",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("© 2026 FaceTrack AI — Hackathon Project")

    return selected_page


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Application entry point: applies theming, initializes the database,
    renders the sidebar, and routes to the selected page.
    """
    apply_custom_theme()

    db_ready = initialize_database()

    selected_page = render_sidebar()

    try:
        if selected_page == "Home":
            show_home_page()
        elif selected_page == "Register Student":
            if db_ready:
                register_user.show_register_page()
            else:
                st.warning("Database is unavailable. Registration is disabled.")
        elif selected_page == "Train Model":
            train_model.show_train_page()
        elif selected_page == "Mark Attendance":
            if db_ready:
                recognize_faces.show_recognize_page()
            else:
                st.warning("Database is unavailable. Attendance marking is disabled.")
        elif selected_page == "Attendance Report":
            if db_ready:
                show_report_page()
            else:
                st.warning("Database is unavailable. Reports are disabled.")
        elif selected_page == "About Project":
            show_about_page()
        else:
            st.error("Unknown page selected.")
    except Exception as err:  # noqa: BLE001
        logger.error("Unhandled error while rendering page '%s': %s", selected_page, err)
        st.error(
            "⚠️ An unexpected error occurred while loading this page. "
            f"Details: {err}"
        )


if __name__ == "__main__":
    main()