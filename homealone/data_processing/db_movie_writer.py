import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import glob
from movie_data import MovieDataProcessor
import time

from homealone.data_processing.movie_data import MovieDataProcessor
# import schema validation manager and schemas
import logging
from homealone.data_processing.schema_validation.validator import ValidationManager
from homealone.data_processing.schema_validation.schemas import MovieMetadata, RatingRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MovieDBWriter:
    
    def __init__(self):
        load_dotenv()
        self.db_path = os.getenv('DB_PATH', 'homealone/db/home_alone.db')
        self.data_dir = os.getenv('DATA_DIR', 'homealone/data_processing/data_queues/')
        self.cache_dir = os.getenv('CACHE_DIR', 'homealone/data_processing/cache/')
        
        
        self.conn = sqlite3.connect(self.db_path)
        self.validator = ValidationManager()
        logger.info("MovieDBWriter initialized with ValidationManager")
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
    
    def movie_id_in_cache(self, movie_id):
        cache_file = os.path.join(self.cache_dir, 'movieids.csv')
        if not os.path.exists(cache_file):
            return False
        df = pd.read_csv(cache_file)
        return movie_id in df['movie_id'].values
    
    def validate_movie_data(self, movie_data):
        is_valid, validated_obj, error_msg = self.validator.validate_record(movie_data, MovieMetadata)
        if not is_valid:
            logger.error(f"Movie metadata validation failed: {error_msg}. Data: {movie_data}")
        return is_valid

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
        """
        Process all movieids CSV files, fetch movie metadata, and write to database with validation.
        """
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        if not all_files:
            logger.info("No CSV files to process")
            return

        logger.info(f"Found {len(all_files)} CSV files to process")

        for file in all_files:
            try:
                if 'movieids' in file:
                    logger.info(f"Processing {file}")
                    df = pd.read_csv(file)
                    movie_ids = df['movie_id'].unique().tolist()
                    logger.info(f"Found {len(movie_ids)} unique movie IDs")

                    success_count = 0
                    skipped_count = 0
                    failed_count = 0

                    for movie_id in movie_ids:
                        try:
                            # Check if already cached
                            if self.movie_id_in_cache(movie_id):
                                logger.debug(f"Skipping cached movie ID: {movie_id}")
                                skipped_count += 1
                                continue

                            # Fetch movie details from multiple sources
                            processor = MovieDataProcessor(movie_id)
                            movie_details = processor.fetch_movie_details()
                            movie_details_tmdb = processor.fetch_movie_details_tmdb(movie_details.get('tmdb_id'))

                            # Combine and filter movie data
                            movie_data = {**movie_details, **movie_details_tmdb, 'movie_id': movie_id}
                            movie_data = {key: movie_data.get(key) for key in self.movie_columns}

                            # Validate movie data
                            is_valid = self.validate_movie_data(movie_data)

                            # Insert/update with validation
                            if is_valid:
                                self.insert_or_update_movie(movie_data)
                                # Add to cache only if successfully inserted
                                self._cache_movie_id_to_csv(movie_id)
                                success_count += 1
                            else:
                                failed_count += 1

                        except Exception as e:
                            logger.error(f"Error processing movie ID {movie_id}: {e}", exc_info=True)
                            failed_count += 1
                            continue

                    # Log summary for this file
                    logger.info(f"File {file} summary: {success_count} inserted, {skipped_count} skipped, {failed_count} failed")

                    # Remove processed file
                    try:
                        os.remove(file)
                        logger.debug(f"Removed processed file: {file}")
                    except OSError as e:
                        logger.error(f"Failed to remove file {file}: {e}")

            except pd.errors.EmptyDataError:
                logger.warning(f"Skipping empty CSV file: {file}")
                try:
                    os.remove(file)
                except OSError as e:
                    logger.error(f"Failed to remove empty file {file}: {e}")
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}", exc_info=True)

        logger.info("All files processed.")
    
    
    
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
        
    def get_validation_stats(self) -> dict:
        """
        Get validation statistics from the validator.

        Returns:
            dict: Validation metrics
        """
        return self.validator.get_stats()


if __name__ == "__main__":
    logger.info("=== Starting MovieDBWriter with Schema Validation ===")
    try:
        writer = MovieDBWriter()
        writer.process_files()
        stats = writer.get_validation_stats()
        logger.info(f"Final validation statistics: {stats}")
        writer.close()
        logger.info("MovieDBWriter completed successfully")
    except KeyboardInterrupt:
        logger.info("MovieDBWriter interrupted by user")
    except Exception as e:
        logger.error(f"Error in MovieDBWriter: {e}", exc_info=True)