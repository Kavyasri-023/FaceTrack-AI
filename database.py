"""
database.py

Database access layer for FaceTrack AI.

Provides a pooled MySQL connection manager along with database/table
initialization and CRUD operations for the `students` and
`attendance` tables. All queries are parameterized to prevent SQL
injection, and all database access goes through a single context
manager to guarantee proper commit/rollback/close behavior.
"""

import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import Error as MySQLError
from mysql.connector import pooling

import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
_POOL_NAME = "facetrack_pool"
_POOL_SIZE = 5

# Lazily-initialized module-level connection pool. Created on first use
# rather than at import time, so importing this module never fails just
# because the database happens to be unreachable at that moment.
_connection_pool: Optional[pooling.MySQLConnectionPool] = None

# Tracks whether we've already verified/created the target database in
# this process, so we don't re-run CREATE DATABASE on every call.
_database_verified = False


def _ensure_database_exists() -> None:
    """
    Verify that the target database (config.DB_NAME) exists, creating it
    if necessary.

    This connects to the MySQL server WITHOUT specifying a database
    name (since the connection pool itself requires the database to
    already exist), runs `CREATE DATABASE IF NOT EXISTS`, then closes
    the connection. This allows the app to work on a fresh MySQL
    server where only the user/credentials have been configured.

    Raises:
        mysql.connector.Error: If the server is unreachable or the
            database cannot be created (e.g. insufficient privileges).
    """
    global _database_verified

    if _database_verified:
        return

    conn = None
    try:
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        cursor.close()
        _database_verified = True
        logger.info("Verified database '%s' exists.", config.DB_NAME)
    except MySQLError as err:
        logger.error(
            "Failed to verify/create database '%s': %s", config.DB_NAME, err
        )
        raise
    finally:
        if conn is not None and conn.is_connected():
            conn.close()


def _get_pool() -> pooling.MySQLConnectionPool:
    """
    Return the shared MySQL connection pool, creating it on first call.

    Ensures the target database exists before building the pool, since
    the pool is configured to connect directly into config.DB_NAME.

    Raises:
        mysql.connector.Error: If the pool cannot be created (e.g. bad
            credentials or unreachable host).
    """
    global _connection_pool

    if _connection_pool is None:
        _ensure_database_exists()
        try:
            _connection_pool = pooling.MySQLConnectionPool(
                pool_name=_POOL_NAME,
                pool_size=_POOL_SIZE,
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                database=config.DB_NAME,
            )
            logger.info("MySQL connection pool created successfully.")
        except MySQLError as err:
            logger.error("Failed to create MySQL connection pool: %s", err)
            raise

    return _connection_pool


@contextmanager
def get_db_cursor(dictionary: bool = True, commit: bool = False):
    """
    Context manager that yields a (connection, cursor) pair from the pool.

    Args:
        dictionary: If True, the cursor returns rows as dictionaries
            (column name -> value) instead of plain tuples.
        commit: If True, automatically commits the transaction when the
            `with` block exits without raising an exception.

    Yields:
        Tuple[MySQLConnection, MySQLCursor]: An active connection and
        cursor pulled from the pool.

    Raises:
        mysql.connector.Error: Re-raised after rolling back the
            transaction and logging the error.
    """
    pool = _get_pool()
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=dictionary)

    try:
        yield conn, cursor
        if commit:
            conn.commit()
    except MySQLError as err:
        conn.rollback()
        logger.error("Database error: %s", err)
        raise
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Table initialization
# ---------------------------------------------------------------------------
def init_db() -> None:
    """
    Create the `students` and `attendance` tables if they do not exist.

    The target database itself is created automatically (if missing)
    the first time the connection pool is built. Safe to call on every
    app startup, since `CREATE TABLE IF NOT EXISTS` is idempotent.

    Raises:
        mysql.connector.Error: If table creation fails.
    """
    create_students_table = """
        CREATE TABLE IF NOT EXISTS students (
            student_id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            department VARCHAR(100) NOT NULL,
            email VARCHAR(100) NOT NULL UNIQUE,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

    create_attendance_table = """
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status VARCHAR(20) DEFAULT 'Present',
            CONSTRAINT fk_student
                FOREIGN KEY (student_id) REFERENCES students(student_id)
                ON DELETE CASCADE,
            UNIQUE KEY unique_attendance_per_day (student_id, date)
        )
    """

    try:
        with get_db_cursor(commit=True) as (conn, cursor):
            cursor.execute(create_students_table)
            cursor.execute(create_attendance_table)
        logger.info("Database tables verified/created successfully.")
    except MySQLError as err:
        logger.error("Failed to initialize database tables: %s", err)
        raise


# ---------------------------------------------------------------------------
# Student CRUD operations
# ---------------------------------------------------------------------------
def add_student(
    student_id: str,
    name: str,
    department: str,
    email: str,
    image_path: str,
) -> bool:
    """
    Insert a new student record into the `students` table.

    Args:
        student_id: Unique identifier for the student.
        name: Full name of the student.
        department: Department/branch the student belongs to.
        email: Student's email address (must be unique).
        image_path: Filesystem path to the student's image dataset
            folder.

    Returns:
        bool: True if the insert succeeded, False otherwise (e.g.
        duplicate student_id/email or connection failure).
    """
    query = """
        INSERT INTO students (student_id, name, department, email, image_path)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        with get_db_cursor(commit=True) as (conn, cursor):
            cursor.execute(query, (student_id, name, department, email, image_path))
        logger.info("Student '%s' added successfully.", student_id)
        return True
    except MySQLError as err:
        logger.error("Failed to add student '%s': %s", student_id, err)
        return False


def get_student_by_id(student_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single student's details by their student ID.

    Args:
        student_id: The student ID to look up.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of the student's columns,
        or None if not found or on error.
    """
    query = "SELECT * FROM students WHERE student_id = %s"
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute(query, (student_id,))
            result = cursor.fetchone()
        return result
    except MySQLError as err:
        logger.error("Failed to fetch student '%s': %s", student_id, err)
        return None


def student_exists(student_id: str) -> bool:
    """
    Check whether a student with the given ID is already registered.

    Args:
        student_id: The student ID to check.

    Returns:
        bool: True if the student exists, False otherwise (including
        on error, to be conservative about duplicate prevention).
    """
    query = "SELECT 1 FROM students WHERE student_id = %s LIMIT 1"
    try:
        with get_db_cursor(dictionary=False) as (conn, cursor):
            cursor.execute(query, (student_id,))
            result = cursor.fetchone()
        return result is not None
    except MySQLError as err:
        logger.error("Failed to check existence of student '%s': %s", student_id, err)
        return False


def email_exists(email: str) -> bool:
    """
    Check whether a student with the given email is already registered.

    Args:
        email: The email address to check.

    Returns:
        bool: True if the email is already registered, False otherwise
        (including on database error).
    """
    query = "SELECT 1 FROM students WHERE email = %s LIMIT 1"
    try:
        with get_db_cursor(dictionary=False) as (conn, cursor):
            cursor.execute(query, (email,))
            result = cursor.fetchone()
        return result is not None
    except MySQLError as err:
        logger.error("Failed to check existence of email '%s': %s", email, err)
        return False


# ---------------------------------------------------------------------------
# Attendance CRUD operations
# ---------------------------------------------------------------------------
def has_attendance_today(student_id: str) -> bool:
    """
    Check whether attendance has already been marked today for a student.

    Args:
        student_id: The student ID to check.

    Returns:
        bool: True if an attendance record exists for today, False
        otherwise (including on error).
    """
    query = "SELECT 1 FROM attendance WHERE student_id = %s AND date = %s LIMIT 1"
    try:
        with get_db_cursor(dictionary=False) as (conn, cursor):
            cursor.execute(query, (student_id, date.today()))
            result = cursor.fetchone()
        return result is not None
    except MySQLError as err:
        logger.error(
            "Failed to check today's attendance for '%s': %s", student_id, err
        )
        return False


def mark_attendance(student_id: str, status: str = "Present") -> bool:
    """
    Mark attendance for a student for the current date and time.

    Enforces the "once per day" rule at the application level (in
    addition to the database UNIQUE constraint on student_id+date).

    Args:
        student_id: The student ID to mark attendance for.
        status: Attendance status label (default: "Present").

    Returns:
        bool: True if attendance was newly marked, False if attendance
        was already marked today or if the insert failed.
    """
    if has_attendance_today(student_id):
        logger.info("Attendance already marked today for '%s'.", student_id)
        return False

    query = """
        INSERT INTO attendance (student_id, date, time, status)
        VALUES (%s, %s, %s, %s)
    """
    now = datetime.now()

    try:
        with get_db_cursor(commit=True) as (conn, cursor):
            cursor.execute(
                query,
                (student_id, now.date(), now.time().strftime("%H:%M:%S"), status),
            )
        logger.info("Attendance marked for '%s' at %s.", student_id, now)
        return True
    except MySQLError as err:
        logger.error("Failed to mark attendance for '%s': %s", student_id, err)
        return False


def get_attendance_records(
    filter_date: Optional[date] = None,
    student_id: Optional[str] = None,
    name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve attendance records, optionally filtered by date, student
    ID, and/or student name.

    Args:
        filter_date: If provided, only return records for this exact
            date.
        student_id: If provided, only return records for this exact
            student ID.
        name: If provided, performs a partial (LIKE) match against the
            student's name.

    Returns:
        List[Dict[str, Any]]: A list of attendance records (each
        including joined student name and department), ordered by most
        recent first. Returns an empty list on error.
    """
    query = """
        SELECT
            a.attendance_id,
            a.student_id,
            s.name,
            s.department,
            a.date,
            a.time,
            a.status
        FROM attendance a
        INNER JOIN students s ON a.student_id = s.student_id
        WHERE 1 = 1
    """
    params: List[Any] = []

    if filter_date is not None:
        query += " AND a.date = %s"
        params.append(filter_date)

    if student_id:
        query += " AND a.student_id = %s"
        params.append(student_id)

    if name:
        query += " AND s.name LIKE %s"
        params.append(f"%{name}%")

    query += " ORDER BY a.date DESC, a.time DESC"

    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute(query, tuple(params))
            results = cursor.fetchall()
        return results
    except MySQLError as err:
        logger.error("Failed to fetch attendance records: %s", err)
        return []


# ---------------------------------------------------------------------------
# Dashboard statistics
# ---------------------------------------------------------------------------
def get_dashboard_stats() -> Dict[str, Any]:
    """
    Compute summary statistics for the attendance dashboard.

    Returns:
        Dict[str, Any]: A dictionary with the keys:
            - "total_students" (int): total registered students.
            - "today_attendance_count" (int): students marked present
              today.
            - "attendance_percentage" (float): percentage of registered
              students who have attended today (0.0 if no students are
              registered, or on error).
    """
    stats: Dict[str, Any] = {
        "total_students": 0,
        "today_attendance_count": 0,
        "attendance_percentage": 0.0,
    }

    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT COUNT(*) AS count FROM students")
            total_students = cursor.fetchone()["count"]

            cursor.execute(
                "SELECT COUNT(*) AS count FROM attendance WHERE date = %s",
                (date.today(),),
            )
            today_count = cursor.fetchone()["count"]

        stats["total_students"] = total_students
        stats["today_attendance_count"] = today_count

        if total_students > 0:
            stats["attendance_percentage"] = round(
                (today_count / total_students) * 100, 2
            )

        return stats
    except MySQLError as err:
        logger.error("Failed to compute dashboard statistics: %s", err)
        return stats