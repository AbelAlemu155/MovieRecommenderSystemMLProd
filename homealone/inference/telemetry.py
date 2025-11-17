import sqlite3
import time


class Telemetry:
    
    def __init__(self, conn):
        self.db_connection = conn
        self._configure_connection()
        self._setup()

    def _configure_connection(self):
        """Configure SQLite connection for better concurrency"""
        try:
            # Enable WAL mode for concurrent reads/writes
            self.db_connection.execute('PRAGMA journal_mode=WAL')
            # Use NORMAL synchronous mode for better performance
            self.db_connection.execute('PRAGMA synchronous=NORMAL')
            # Increase busy timeout to 60 seconds
            self.db_connection.execute('PRAGMA busy_timeout=60000')
            print("Database configured with WAL mode for concurrency")
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not configure database: {e}")

    def _setup(self):
        """Create tables if they don't exist"""
        cursor = self.db_connection.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    recommended_items TEXT NOT NULL,
                    model_version TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status INTEGER DEFAULT 200,
                    response_time_ms REAL,
                    server TEXT
                )
            """)
            self.db_connection.commit()
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not create recommendations table: {e}")
            # Table might already exist, continue

        # Migration: Add missing columns if they don't exist (for backwards compatibility)
        try:
            cursor.execute("ALTER TABLE recommendations ADD COLUMN model_version TEXT")
            self.db_connection.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE recommendations ADD COLUMN status INTEGER DEFAULT 200")
            self.db_connection.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE recommendations ADD COLUMN response_time_ms REAL")
            self.db_connection.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE recommendations ADD COLUMN server TEXT")
            self.db_connection.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create index for efficient time-based queries used by monitoring
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp
                ON recommendations(timestamp)
            """)
            self.db_connection.commit()
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not create index: {e}")
            # Index might already exist, continue
        
    def log(self, event_type, details):
        if event_type not in ['recommendation']:
            raise ValueError("Invalid event type for telemetry logging.")

        if event_type == 'recommendation':
            user_id = details['user_id']
            recommended_items = details['recommended_items']
            timestamp = details.get('timestamp')
            model_version = details.get('model_version')
            status = details.get('status', 200)
            response_time_ms = details.get('response_time_ms')
            server = details.get('server')
            self._recommendation_log(user_id, recommended_items, timestamp, model_version,
                                    status, response_time_ms, server)

    def _recommendation_log(self, user_id, recommended_items, timestamp, model_version=None,
                           status=200, response_time_ms=None, server=None):
        """Log recommendation with retry logic for database locks"""
        max_retries = 3
        retry_delay = 0.1  # Start with 100ms

        for attempt in range(max_retries):
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO recommendations (user_id, recommended_items, timestamp, model_version,
                                                status, response_time_ms, server)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, recommended_items, timestamp, model_version,
                      status, response_time_ms, server))
                self.db_connection.commit()
                return  # Success, exit
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    print(f"Database lock on attempt {attempt + 1}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to log recommendation after {max_retries} attempts: {e}")
                    # Don't raise - we don't want to crash the recommendation service
                    # The recommendation still succeeded, just logging failed
