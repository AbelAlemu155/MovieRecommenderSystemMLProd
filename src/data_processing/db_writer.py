import pandas as pd
import sqlite3
import os
import glob
from dotenv import load_dotenv

class DBWriter:
    def __init__(self):
        load_dotenv()
        self.db_path = os.getenv('DB_PATH', 'src/db/home_alone.db')
        self.data_dir = os.getenv('DATA_DIR', 'src/data_processing/data_queues/')
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watch_history (
                user_id INTEGER,
                movie_id TEXT,
                total_watch_time INTEGER DEFAULT 0,
                start_time TEXT,
                last_watch TEXT,
                PRIMARY KEY (user_id, movie_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER,
                movie_id TEXT,
                rating REAL DEFAULT NULL,
                rated_at TEXT,
                PRIMARY KEY (user_id, movie_id)
            )
        ''')
        self.conn.commit()

    def update_watch_history(self, data):
        cursor = self.conn.cursor()
        for _, row in data.iterrows():
            query = f"""
                INSERT INTO watch_history (user_id, movie_id, total_watch_time, start_time, last_watch)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET
                    total_watch_time = watch_history.total_watch_time + excluded.total_watch_time,
                    start_time = CASE 
                        WHEN excluded.start_time < watch_history.start_time THEN excluded.start_time
                        ELSE watch_history.start_time
                    END,
                    last_watch = CASE 
                        WHEN excluded.last_watch > watch_history.last_watch THEN excluded.last_watch
                        ELSE watch_history.last_watch
                    END;
            """
            cursor.execute(query, (
                row['userid'],
                row['movieid'],
                row['minute'],
                row['start_time'],
                row['last_watch']
            ))
        self.conn.commit()

    def update_ratings(self, data):
        cursor = self.conn.cursor()
        for _, row in data.iterrows():
            query = f"""
                INSERT INTO ratings (user_id, movie_id, rating, rated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET
                    rating = excluded.rating,
                    rated_at = excluded.rated_at;
            """
            cursor.execute(query, (
                row['userid'],
                row['movieid'],
                row['rating'],
                row['rated_at']
            ))
        self.conn.commit()

    def process_files(self):
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        for file in all_files:
            print(f"Processing {file}")
            if 'movies' in file:
                df = pd.read_csv(file)
                grouped = df.groupby(['userid', 'movieid']).agg({'minute': 'count', 'time': ['min', 'max']}).reset_index()
                grouped.columns = ['userid', 'movieid', 'minute', 'start_time', 'last_watch']
                self.update_watch_history(grouped)
                os.remove(file)
            elif 'ratings' in file:
                df = pd.read_csv(file)
                grouped = df.groupby(['userid', 'movieid']).agg({'time': 'max', 'rating': 'max'}).reset_index()
                grouped.columns = ['userid', 'movieid', 'rated_at', 'rating']
                self.update_ratings(grouped)
                os.remove(file)
        print("All files processed.")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    writer = DBWriter()
    writer.process_files()
    writer.close()