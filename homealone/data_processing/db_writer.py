import pandas as pd
import sqlite3
import os
import glob
from dotenv import load_dotenv
import time
from tqdm import tqdm

# import schema validation manager and schemas
import logging
from schema_validation.validator import ValidationManager
from schema_validation.schemas import WatchHistoryRecord, RatingRecord

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

        # Check if there are no files to process. Log and return early.
        if not all_files:
            logger.info("No CSV files to process")
            return

        logger.info(f"Found {len(all_files)} CSV files to process")

        # for file in all_files:
        for file in tqdm(all_files, desc="Processing data files", unit="file"):
            try:
                if 'movies' in file or 'ratings' in file:
                    df = pd.read_csv(file)
                    print(f"Processing {file}")
                    if 'movies' in file:
                        # Aggregate movie watch data
                        grouped = df.groupby(['userid', 'movieid']).agg(
                            {'minute': 'count', 'time': ['min', 'max']}).reset_index()
                        grouped.columns = ['userid', 'movieid',
                                           'minute', 'start_time', 'last_watch']

                        # Rename columns to match schema
                        grouped_renamed = grouped.rename(columns={
                            'userid': 'user_id',
                            'movieid': 'movie_id',
                            'minute': 'total_watch_time'
                        })

                        # Convert DataFrame to list of dictionaries (which the batch validation function expects)
                        records_list = grouped_renamed.to_dict('records')
                        valid_records, invalid_records = self.validate_db_data(
                            records_list, WatchHistoryRecord)
                        if valid_records:
                            # Convert validated pydantic models back to DataFrame
                            valid_df = pd.DataFrame(
                                [record.model_dump() for record in valid_records])
                            # Rename columns back for database insertion
                            valid_df = valid_df.rename(columns={
                                'user_id': 'userid',
                                'movie_id': 'movieid',
                                'total_watch_time': 'minute'
                            })
                            self.update_watch_history(valid_df)
                        if invalid_records:
                            logger.warning(
                                f"Found {len(invalid_records)} invalid watch history records")

                    elif 'ratings' in file:
                       # Aggregate rating data
                        grouped = df.groupby(['userid', 'movieid']).agg(
                            {'time': 'max', 'rating': 'max'}).reset_index()
                        grouped.columns = [
                            'userid', 'movieid', 'rated_at', 'rating']
                        # Rename columns to match schema
                        grouped_renamed = grouped.rename(columns={
                            'userid': 'user_id',
                            'movieid': 'movie_id'
                        })
                        # Convert DataFrame to list of dictionaries
                        records_list = grouped_renamed.to_dict('records')
                        valid_records, invalid_records = self.validate_db_data(
                            records_list, RatingRecord)
                        if valid_records:
                            # Convert validated pydantic models back to DataFrame
                            valid_df = pd.DataFrame(
                                [record.model_dump() for record in valid_records])
                            # Rename columns back for database insertion
                            valid_df = valid_df.rename(columns={
                                'user_id': 'userid',
                                'movie_id': 'movieid'
                            })
                            self.update_ratings(valid_df)
                        if invalid_records:
                            logger.warning(
                                f"Found {len(invalid_records)} invalid rating records")

                    # Remove file after successful processing
                    try:
                        os.remove(file)
                        logger.debug(f"Removed processed file: {file}")
                    except OSError as e:
                        logger.error(
                            f"Failed to remove empty file {file}: {e}")
                
            # if the file is empty, log and remove it
            except pd.errors.EmptyDataError:
                logger.warning(f"Skipping empty CSV file: {file}")
                try:
                    os.remove(file)
                except OSError as e:
                    logger.error(f"Failed to remove empty file {file}: {e}")
            
            # catch all other exceptions to prevent crash
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
