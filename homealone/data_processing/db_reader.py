import pandas as pd
import sqlite3
import os
import glob
from dotenv import load_dotenv


class DBReader:
    def __init__(self, db_path=None):
        load_dotenv()
        self.db_path = db_path or os.getenv('DB_PATH', 'src/db/home_alone.db')
        self.conn = sqlite3.connect(self.db_path)

    def get_watch_history(self, user_id=None, movie_id=None):
        query = "SELECT * FROM watch_history"
        params = []
        if user_id is not None and movie_id is not None:
            query += " WHERE user_id = ? AND movie_id = ?"
            params = [user_id, movie_id]
        elif user_id is not None:
            query += " WHERE user_id = ?"
            params = [user_id]
        elif movie_id is not None:
            query += " WHERE movie_id = ?"
            params = [movie_id]
        df = pd.read_sql_query(query, self.conn, params=params)
        return df

    def get_ratings(self, user_id=None, movie_id=None):
        query = "SELECT * FROM ratings"
        params = []
        if user_id is not None and movie_id is not None:
            query += " WHERE user_id = ? AND movie_id = ?"
            params = [user_id, movie_id]
        elif user_id is not None:
            query += " WHERE user_id = ?"
            params = [user_id]
        elif movie_id is not None:
            query += " WHERE movie_id = ?"
            params = [movie_id]
        df = pd.read_sql_query(query, self.conn, params=params)
        return df

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    
    # Example usage of DBReader
    reader = DBReader()
    print(reader.get_watch_history())
    print(reader.get_ratings())
    reader.close()