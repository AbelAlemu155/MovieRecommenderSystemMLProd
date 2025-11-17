# homealone/inference/telemetry.py

import sqlite3
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Telemetry:
    """
    Handles telemetry logging for the recommendation system.
    
    Logs recommendation events to a SQLite database with UPSERT behavior
    (updates existing records for the same user_id).
    """
    
    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize Telemetry with a database connection.
        
        Args:
            conn: SQLite database connection
        """
        self.conn = conn
        self._create_tables()
    
    def _create_tables(self):
        """Create the recommendations table if it doesn't exist."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                recommended_items TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        logger.info("Recommendations table created/verified")
    
    def log(self, event_type: str, details: Dict[str, Any]):
        """
        Log a telemetry event.
        
        Args:
            event_type: Type of event (currently only 'recommendation')
            details: Dictionary containing event details
                - user_id: User ID (required)
                - recommended_items: Comma-separated movie IDs (required)
                - timestamp: Event timestamp (optional, defaults to CURRENT_TIMESTAMP)
        
        Raises:
            ValueError: If event_type is invalid
            KeyError: If required fields are missing from details
        """
        # Validate event type
        if not event_type or event_type != 'recommendation':
            raise ValueError(f"Invalid event type: {event_type}. Only 'recommendation' is supported.")
        
        # Extract required fields
        user_id = str(details['user_id'])  # Convert to string
        recommended_items = details['recommended_items']
        timestamp = details.get('timestamp')  # Optional
        
        # Validate recommended_items is not None
        if recommended_items is None:
            raise TypeError("recommended_items cannot be None")
        
        # Convert to string if needed
        recommended_items = str(recommended_items)
        
        # Log the recommendation with UPSERT behavior
        self._log_recommendation(user_id, recommended_items, timestamp)
    
    def _log_recommendation(self, user_id: str, recommended_items: str, timestamp: str = None):
        """
        Log a recommendation event with UPSERT (INSERT or UPDATE on conflict).
        
        Args:
            user_id: User ID
            recommended_items: Comma-separated list of recommended movie IDs
            timestamp: Event timestamp (optional)
        """
        cursor = self.conn.cursor()
        
        if timestamp is None:
            # Use CURRENT_TIMESTAMP
            cursor.execute("""
                INSERT INTO recommendations (user_id, recommended_items, timestamp)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) 
                DO UPDATE SET 
                    recommended_items = excluded.recommended_items,
                    timestamp = CURRENT_TIMESTAMP
            """, (user_id, recommended_items))
        else:
            # Use provided timestamp
            cursor.execute("""
                INSERT INTO recommendations (user_id, recommended_items, timestamp)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) 
                DO UPDATE SET 
                    recommended_items = excluded.recommended_items,
                    timestamp = excluded.timestamp
            """, (user_id, recommended_items, timestamp))
        
        self.conn.commit()
        logger.debug(f"Logged recommendation for user {user_id}")
