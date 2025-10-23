import pandas as pd
import sqlite3
import os
import glob
from dotenv import load_dotenv
import time
from tqdm import tqdm

class DBWriter:
    def __init__(self):
        load_dotenv()
        # self.db_path = os.getenv('DB_PATH', 'homealone/db/home_alone.db')
        self.db_path = '/app/db/home_alone.db'
        # self.data_dir = os.getenv('DATA_QUEUE_DIR', 'homealone/data_processing/data_queues/')
        self.data_dir = "/app/data_queues/"
        
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movies (
            movie_id TEXT PRIMARY KEY,
            tmdb_id TEXT,
            imdb_id TEXT,
            adult BOOLEAN,
            genres TEXT,
            overview TEXT,
            original_language TEXT,
            popularity REAL,
            production_companies TEXT,
            release_date TEXT,
            revenue INTEGER,
            budget INTEGER,
            runtime INTEGER,
            spoken_languages TEXT,
            belongs_to_collection TEXT,
            vote_average REAL,
            vote_count INTEGER,
            video BOOLEAN,
            title TEXT,
            production_countries TEXT,
            status TEXT,
            tagline TEXT
            )
        ''')
        
        self.conn.commit()

    def insert_or_update_movie(self, movie_data):
        cursor = self.conn.cursor()
        
        query = f"""
            INSERT INTO movies (movie_id, tmdb_id, imdb_id, adult, genres, overview, original_language, popularity,
                                production_companies, release_date, revenue, budget, runtime, spoken_languages,
                                belongs_to_collection, vote_average, vote_count, video, title, production_countries,
                                status, tagline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(movie_id) DO UPDATE SET
                tmdb_id = excluded.tmdb_id,
                imdb_id = excluded.imdb_id,
                adult = excluded.adult,
                genres = excluded.genres,
                overview = excluded.overview,
                original_language = excluded.original_language,
                popularity = excluded.popularity,
                production_companies = excluded.production_companies,
                release_date = excluded.release_date,
                revenue = excluded.revenue,
                budget = excluded.budget,
                runtime = excluded.runtime,
                spoken_languages = excluded.spoken_languages,
                belongs_to_collection = excluded.belongs_to_collection,
                vote_average = excluded.vote_average,
                vote_count = excluded.vote_count,
                video = excluded.video,
                title = excluded.title,
                production_countries = excluded.production_countries,
                status = excluded.status,
                tagline = excluded.tagline
            WHERE movies.movie_id = excluded.movie_id;
        """
        cursor.execute(query, (
            movie_data['movie_id'],
            movie_data['tmdb_id'],
            movie_data['imdb_id'],
            movie_data['adult'],
            movie_data['genres'],
            movie_data['overview'],
            movie_data['original_language'],
            movie_data['popularity'],
            movie_data['production_companies'],
            movie_data['release_date'],
            movie_data['revenue'],
            movie_data['budget'],
            movie_data['runtime'],
            movie_data['spoken_languages'],
            movie_data['belongs_to_collection'],
            movie_data['vote_average'],
            movie_data['vote_count'],
            movie_data['video'],
            movie_data['title'],
            movie_data['production_countries'],
            movie_data['status'],
            movie_data['tagline']
        ))
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
        
        
        for file in tqdm(all_files, desc="Processing data files", unit="file"):
        # for file in all_files:
            
            if 'movies' in file or 'ratings' in file:
                df = pd.read_csv(file)
                print(f"Processing {file}")
                if 'movies' in file:
                    print("Updating watch history...")
                    grouped = df.groupby(['userid', 'movieid']).agg({'minute': 'count', 'time': ['min', 'max']}).reset_index()
                    grouped.columns = ['userid', 'movieid', 'minute', 'start_time', 'last_watch']
                    self.update_watch_history(grouped)
                    os.remove(file)
                elif 'ratings' in file:
                    print("Updating ratings...")
                    grouped = df.groupby(['userid', 'movieid']).agg({'time': 'max', 'rating': 'max'}).reset_index()
                    grouped.columns = ['userid', 'movieid', 'rated_at', 'rating']
                    self.update_ratings(grouped)
                    os.remove(file)
        print("All files processed.")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    while True:
        print("Starting DB Writer...")
        writer = DBWriter()
        writer.process_files()
        writer.close()
        time.sleep(5)