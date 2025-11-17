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
                user_id INTEGER NOT NULL,
                recommended_items TEXT,
                timestamp DATETIME NOT NULL,
                model_version TEXT,
                data_version TEXT,
                git_commit TEXT,
                mlflow_run_id TEXT,
                UNIQUE(user_id, timestamp)
            )
        """)
        self.db_connection.commit()
        
    def log(self, event_type, details):
        if event_type not in ['recommendation']:
            raise ValueError("Invalid event type for telemetry logging.")

        if event_type == 'recommendation':
            user_id, recommended_items, timestamp = details['user_id'], details['recommended_items'], details['timestamp']
            model_version = details.get('model_version')
            data_version = details.get('data_version')
            git_commit = details.get('git_commit')
            mlflow_run_id = details.get('mlflow_run_id')
            self._recommendation_log(user_id, recommended_items, timestamp, model_version, data_version, git_commit, mlflow_run_id)

    def _recommendation_log(self, user_id, recommended_items, timestamp, model_version, data_version, git_commit, mlflow_run_id):
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO recommendations (user_id, recommended_items, timestamp, model_version, data_version, git_commit, mlflow_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, recommended_items, timestamp, model_version, data_version, git_commit, mlflow_run_id))
        self.db_connection.commit()
