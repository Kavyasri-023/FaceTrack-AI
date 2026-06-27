"""
utils/ui_helpers.py

Reusable Streamlit UI components for FaceTrack AI: dashboard stat
cards and status badges, styled for the app's dark-blue theme.
"""

import streamlit as st

# Status badge color mapping (background, text) for the dark-blue theme.
_STATUS_COLORS = {
    "success": ("#13361f", "#5ad17c"),
    "error": ("#3a1320", "#ff6b81"),
    "warning": ("#3a2f10", "#f0c674"),
    "info": ("#10283a", "#5bc0ff"),
}


def render_stat_card(value: str, label: str) -> str:
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


def render_status_badge(text: str, status: str = "info") -> str:
    """
    Build the HTML markup for a colored status badge/indicator.

    Args:
        text: The label text to display inside the badge.
        status: One of "success", "error", "warning", "info".

    Returns:
        str: HTML string for the rendered badge.
    """
    background, foreground = _STATUS_COLORS.get(status, _STATUS_COLORS["info"])
    return f"""
        <span style="
            background-color: {background};
            color: {foreground};
            padding: 0.25em 0.75em;
            border-radius: 999px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
        ">
            {text}
        </span>
    """


def show_stat_cards(stats: dict) -> None:
    """
    Render a row of three dashboard stat cards (total students, today's
    attendance, attendance percentage) using `st.columns`.

    Args:
        stats: Dict with keys "total_students", "today_attendance_count",
            and "attendance_percentage".
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            render_stat_card(str(stats.get("total_students", 0)), "Total Registered Students"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_stat_card(str(stats.get("today_attendance_count", 0)), "Today's Attendance"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_stat_card(f"{stats.get('attendance_percentage', 0.0)}%", "Attendance Percentage"),
            unsafe_allow_html=True,
        )