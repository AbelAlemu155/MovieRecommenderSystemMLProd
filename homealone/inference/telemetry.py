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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Remove UNIQUE constraint from user_id to allow multiple entries
        self.db_connection.commit()
        
    def log(self, event_type, details):
        if event_type not in ['recommendation']:
            raise ValueError("Invalid event type for telemetry logging.")

        if event_type == 'recommendation':
            user_id = details['user_id']
            recommended_items = details['recommended_items']
            timestamp = details.get('timestamp')
            model_version = details.get('model_version')
            self._recommendation_log(user_id, recommended_items, timestamp, model_version)

    def _recommendation_log(self, user_id, recommended_items, timestamp, model_version=None):
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT INTO recommendations (user_id, recommended_items, timestamp, model_version)
            VALUES (?, ?, ?, ?)
        """, (user_id, recommended_items, timestamp, model_version))
        self.db_connection.commit()
