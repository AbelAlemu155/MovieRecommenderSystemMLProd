"""
Database utility functions for monitoring components
Provides consistent database configuration across all monitoring modules
"""
import sqlite3


def get_db_connection(db_path: str, timeout: float = 60.0) -> sqlite3.Connection:
    """
    Create a database connection with proper configuration for concurrency.

    Args:
        db_path: Path to SQLite database file
        timeout: Connection timeout in seconds (default 60s for Windows Docker issues)

    Returns:
        Configured SQLite connection
    """
    conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)

    # Configure for concurrent access
    try:
        # Enable WAL mode for concurrent reads/writes
        conn.execute('PRAGMA journal_mode=WAL')
        # Use NORMAL synchronous mode for better performance
        conn.execute('PRAGMA synchronous=NORMAL')
        # Set busy timeout
        conn.execute(f'PRAGMA busy_timeout={int(timeout * 1000)}')
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not configure database connection: {e}")

    return conn
