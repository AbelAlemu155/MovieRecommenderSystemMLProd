import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import glob
from data_processing.movie_data import MovieDataProcessor

class MovieDBWriter:
    
    def __init__(self):
        load_dotenv()
        self.db_path = os.getenv('DB_PATH', 'src/db/home_alone.db')
        self.data_dir = os.getenv('DATA_DIR', 'src/data_processing/data_queues/')
        self.cache_dir = os.getenv('CACHE_DIR', 'src/data_processing/cache/')
        
        
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        
        self.movie_columns = [
            'movie_id', 'tmdb_id', 'imdb_id', 'adult', 'genres', 'overview', 'original_language', 'popularity',
            'production_companies', 'release_date', 'revenue', 'budget', 'runtime', 'spoken_languages',
            'belongs_to_collection', 'vote_average', 'vote_count', 'video', 'title', 'production_countries',
            'status', 'tagline'
        ]
        
    def _cache_movie_id_to_csv(self, movie_id):
        cache_file = os.path.join(self.cache_dir, 'movieids.csv')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        df = pd.DataFrame([{'movie_id': movie_id}])
        if os.path.exists(cache_file):
            df_existing = pd.read_csv(cache_file)
            df = pd.concat([df_existing, df]).drop_duplicates().reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        
    def _create_tables(self):
        cursor = self.conn.cursor()
       
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

    
    def process_files(self):
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        for file in all_files:
            if 'movieids' in file:
                print(f"Processing {file}")
                df = pd.read_csv(file)
                movie_ids = df['movie_id'].unique().tolist()
                print(f"Found {len(movie_ids)} unique movie IDs.")
                for movie_id in movie_ids:
                    processor = MovieDataProcessor(movie_id)
                    movie_details = processor.fetch_movie_details()
                    # print("---Movie Details---")
                    # print(movie_details)
                    movie_details_tmdb = processor.fetch_movie_details_tmdb(movie_details.get('tmdb_id'))
                    # print("---TMDb Details---")
                    # print(movie_details_tmdb)
                    movie_data = {**movie_details, **movie_details_tmdb, 'movie_id': movie_id}
                    # print("---Combined Movie Data---")
                    # print(movie_data)
                    # pick only the columns we want
                    movie_data = {key: movie_data.get(key) for key in self.movie_columns}
                    # print("---Filtered Movie Data---")
                    # print(movie_data)
                    self.insert_or_update_movie(movie_data)
                    
                    # add movie_id to cache
                    self._cache_movie_id_to_csv(movie_id)
                    print(f"cached movie ID: {movie_id}")
                print(f"Finished processing {file}")
                os.remove(file)
                break

        print("All files processed.")
    
    
    
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


    def close(self):
        self.conn.close()
        
if __name__ == "__main__":
    writer = MovieDBWriter()
    writer.process_files()
    writer.close()