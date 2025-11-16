import sqlite3


class Telemetry:
    
    def __init__(self, conn):
        self.db_connection = conn
        self._setup()

    def _setup(self):
        """Create tables if they don't exist"""
        cursor = self.db_connection.cursor()
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
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp
            ON recommendations(timestamp)
        """)
        self.db_connection.commit()
        
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
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT INTO recommendations (user_id, recommended_items, timestamp, model_version,
                                        status, response_time_ms, server)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, recommended_items, timestamp, model_version,
              status, response_time_ms, server))
        self.db_connection.commit()
