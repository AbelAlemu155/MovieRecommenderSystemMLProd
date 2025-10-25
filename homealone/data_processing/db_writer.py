import pandas as pd
import sqlite3
import os
import glob
from dotenv import load_dotenv
import time
from tqdm import tqdm
# import schema validation manager and schemas
import logging
from homealone.data_processing.schema_validation.validator import ValidationManager
from homealone.data_processing.schema_validation.schemas import WatchHistoryRecord, RatingRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        # initialize validation manager
        self.validator = ValidationManager()
        logger.info("DBWriter initialized with ValidationManager")
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

    def validate_db_data(self, movie_data, schema):
        valid_records, invalid_records = self.validator.validate_batch(
            movie_data, schema)
        return valid_records, invalid_records

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
        """
        Process all CSV files in data queue directory and write to database with validation.
        """
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
                except OSError as e:
                    logger.error(f"Failed to remove empty file {file}: {e}")
            except Exception as e:
                logger.error(
                    f"Error processing file {file}: {e}", exc_info=True)

        logger.info("All files processed.")

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
    logger.info("=== Starting DBWriter with Schema Validation ===")
    while True:
        try:
            logger.info("Starting DB Writer cycle...")
            writer = DBWriter()
            writer.process_files()
            stats = writer.get_validation_stats()
            logger.info(f"Validation statistics: {stats}")
            writer.close()
            logger.debug(
                "DB Writer cycle completed. Sleeping for 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("DBWriter interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in DBWriter main loop: {e}", exc_info=True)
            time.sleep(5)  # Wait before retry
