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
                user_id TEXT NOT NULL UNIQUE,
                recommended_items TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db_connection.commit()
        
    def log(self, event_type, details):
        if event_type not in ['recommendation']:
            raise ValueError("Invalid event type for telemetry logging.")

        if event_type == 'recommendation':
            user_id, recommended_items, timestamp = details['user_id'], details['recommended_items'], details['timestamp']
            self._recommendation_log(user_id, recommended_items, timestamp)

    def _recommendation_log(self, user_id, recommended_items, timestamp):
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT INTO recommendations (user_id, recommended_items, timestamp)
            VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET
            recommended_items=excluded.recommended_items, timestamp=excluded.timestamp
        """, (user_id, recommended_items, timestamp))
        self.db_connection.commit()
<<<<<<< HEAD

    
=======
>>>>>>> e22ab9087cedcf91932fb026dea9c8a11004a088
