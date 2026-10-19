

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "jobs.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the jobs table and safely add missing tracking columns."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                location TEXT DEFAULT 'Not specified',
                description TEXT DEFAULT '',
                url TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                posted_at TEXT,
                discovered_at TEXT NOT NULL,
                category TEXT DEFAULT 'Other',
                score INTEGER DEFAULT 0,
                status TEXT DEFAULT 'New'
            )
        """)

        # This migration must be outside the SQL string.
        existing_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(jobs)"
            ).fetchall()
        }

        new_columns = {
            "application_date": "TEXT",
            "follow_up_date": "TEXT",
            "notes": "TEXT DEFAULT ''",
        }

        for column, definition in new_columns.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE jobs ADD COLUMN {column} {definition}"
                )

        conn.commit()


def add_job(job):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO jobs (
                title, company, location, description, url,
                source, posted_at, discovered_at, category, score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job["title"],
            job["company"],
            job.get("location", "Not specified"),
            job.get("description", ""),
            job["url"],
            job.get("source", "Unknown"),
            job.get("posted_at"),
            job["discovered_at"],
            job.get("category", "Other"),
            job.get("score", 0),
        ))
        conn.commit()
        return cursor.rowcount == 1


def get_all_jobs():
    with get_connection() as conn:
        return conn.execute("""
            SELECT * FROM jobs
            ORDER BY score DESC, discovered_at DESC
        """).fetchall()


def update_job_status(job_id, status):
    allowed = {
        "New", "To Apply", "Applied",
        "Interview", "Rejected", "Offer", "Skipped",
    }

    if status not in allowed:
        raise ValueError(f"Invalid status: {status}")

    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = ? WHERE id = ?",
            (status, job_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def remove_demo_jobs():
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM jobs WHERE source = ?",
            ("DEMO",),
        )
        conn.commit()
        return cursor.rowcount


def update_job_match(job_url, category, score):
    """Refresh a saved job's category and relevance score."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE jobs
            SET category = ?, score = ?
            WHERE url = ?
            """,
            (category, score, job_url),
        )
        conn.commit()
        return cursor.rowcount > 0


def update_job_tracking(
    job_id,
    status,
    application_date,
    follow_up_date,
    notes,
):
    """Save application progress without changing job details."""
    allowed_statuses = {
        "New", "To Apply", "Applied",
        "Interview", "Rejected", "Offer", "Skipped",
    }

    if status not in allowed_statuses:
        raise ValueError(f"Invalid status: {status}")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                application_date = ?,
                follow_up_date = ?,
                notes = ?
            WHERE id = ?
            """,
            (
                status,
                application_date or None,
                follow_up_date or None,
                notes or "",
                job_id,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_due_follow_ups(today):
    """Return active applications due for follow-up."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT *
            FROM jobs
            WHERE follow_up_date IS NOT NULL
              AND follow_up_date != ''
              AND follow_up_date <= ?
              AND status IN ('To Apply', 'Applied', 'Interview')
            ORDER BY follow_up_date ASC
            """,
            (today,),
        ).fetchall()