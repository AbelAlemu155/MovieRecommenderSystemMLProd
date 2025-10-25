import pytest
from unittest.mock import MagicMock, Mock, patch, mock_open, call
import pandas as pd
import sqlite3
import os
import sys

# Mock schema_validation module before importing DBWriter
sys.modules['schema_validation'] = MagicMock()
sys.modules['schema_validation.validator'] = MagicMock()
sys.modules['schema_validation.schemas'] = MagicMock()


from homealone.data_processing.db_writer import DBWriter


# ============================================================================
# Initialization Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_db_writer_initializes_with_paths(mock_validator_class, mock_makedirs, mock_connect):
    """Test DBWriter initializes with correct paths"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_validator = MagicMock()
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()

    assert writer.db_path == '/app/db/home_alone.db'
    assert writer.data_dir == '/app/data_queues/'
    assert writer.conn == mock_conn
    assert writer.validator == mock_validator


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_db_writer_creates_directories(mock_validator_class, mock_makedirs, mock_connect):
    """Test DBWriter creates necessary directories"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    writer = DBWriter()

    # Should create directories for db_path and data_dir
    assert mock_makedirs.call_count == 2
    mock_makedirs.assert_any_call('/app/db', exist_ok=True)
    mock_makedirs.assert_any_call('/app/data_queues/', exist_ok=True)


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_db_writer_creates_tables_on_init(mock_validator_class, mock_makedirs, mock_connect):
    """Test DBWriter creates tables during initialization"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()

    # Should execute CREATE TABLE statements
    assert mock_cursor.execute.call_count == 3
    mock_conn.commit.assert_called()


# ============================================================================
# Table Creation Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_create_tables_creates_watch_history(mock_validator_class, mock_makedirs, mock_connect):
    """Test watch_history table is created with correct schema"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()

    # Find the watch_history table creation call
    watch_history_calls = [call for call in mock_cursor.execute.call_args_list
                          if 'watch_history' in str(call)]
    assert len(watch_history_calls) > 0

    # Verify columns
    watch_history_sql = str(watch_history_calls[0])
    assert 'user_id' in watch_history_sql
    assert 'movie_id' in watch_history_sql
    assert 'total_watch_time' in watch_history_sql


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_create_tables_creates_ratings(mock_validator_class, mock_makedirs, mock_connect):
    """Test ratings table is created with correct schema"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()

    # Find the ratings table creation call
    ratings_calls = [call for call in mock_cursor.execute.call_args_list
                    if 'ratings' in str(call) and 'CREATE' in str(call)]
    assert len(ratings_calls) > 0

    # Verify columns
    ratings_sql = str(ratings_calls[0])
    assert 'user_id' in ratings_sql
    assert 'movie_id' in ratings_sql
    assert 'rating' in ratings_sql


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_create_tables_creates_movies(mock_validator_class, mock_makedirs, mock_connect):
    """Test movies table is created with correct schema"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()

    # Find the movies table creation call
    movies_calls = [call for call in mock_cursor.execute.call_args_list
                   if 'movies' in str(call) and 'CREATE' in str(call)]
    assert len(movies_calls) > 0


# ============================================================================
# Validation Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_validate_db_data_returns_valid_invalid_records(mock_validator_class, mock_makedirs, mock_connect, sample_watch_history_data):
    """Test validate_db_data separates valid and invalid records"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    mock_validator = MagicMock()
    valid_records = [MagicMock()]
    invalid_records = [MagicMock()]
    mock_validator.validate_batch.return_value = (valid_records, invalid_records)
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    valid, invalid = writer.validate_db_data(sample_watch_history_data, "WatchHistoryRecord")

    assert valid == valid_records
    assert invalid == invalid_records
    mock_validator.validate_batch.assert_called_once_with(sample_watch_history_data, "WatchHistoryRecord")


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_validate_db_data_handles_empty_data(mock_validator_class, mock_makedirs, mock_connect):
    """Test validate_db_data handles empty data"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    mock_validator = MagicMock()
    mock_validator.validate_batch.return_value = ([], [])
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    valid, invalid = writer.validate_db_data([], "WatchHistoryRecord")

    assert valid == []
    assert invalid == []


# ============================================================================
# Watch History Update Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_update_watch_history_inserts_new_record(mock_validator_class, mock_makedirs, mock_connect, sample_watch_history_df):
    """Test update_watch_history inserts new watch history record"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    writer.update_watch_history(sample_watch_history_df)

    # Should execute INSERT query
    assert mock_cursor.execute.call_count >= 4  # 3 for table creation + 1 for insert
    mock_conn.commit.assert_called()


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_update_watch_history_with_multiple_rows(mock_validator_class, mock_makedirs, mock_connect, multiple_watch_history_df):
    """Test update_watch_history handles multiple rows"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    writer.update_watch_history(multiple_watch_history_df)

    # Should execute 3 times for table creation + 2 times for data
    assert mock_cursor.execute.call_count >= 5


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_update_watch_history_uses_upsert(mock_validator_class, mock_makedirs, mock_connect, sample_watch_history_df):
    """Test update_watch_history uses INSERT...ON CONFLICT"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    writer.update_watch_history(sample_watch_history_df)

    # Check that ON CONFLICT is used
    execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
    watch_history_inserts = [call for call in execute_calls if 'INSERT INTO watch_history' in call]

    assert len(watch_history_inserts) > 0
    assert 'ON CONFLICT' in watch_history_inserts[0]


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_update_watch_history_with_empty_dataframe(mock_validator_class, mock_makedirs, mock_connect):
    """Test update_watch_history handles empty DataFrame"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    empty_df = pd.DataFrame(columns=['userid', 'movieid', 'minute', 'start_time', 'last_watch'])

    # Should not error with empty DataFrame
    writer.update_watch_history(empty_df)


# ============================================================================
# Ratings Update Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_update_ratings_inserts_new_record(mock_validator_class, mock_makedirs, mock_connect, sample_ratings_df):
    """Test update_ratings inserts new rating record"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    writer.update_ratings(sample_ratings_df)

    # Should execute INSERT query
    assert mock_cursor.execute.call_count >= 4  # 3 for table creation + 1 for insert
    mock_conn.commit.assert_called()


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_update_ratings_with_multiple_rows(mock_validator_class, mock_makedirs, mock_connect, multiple_ratings_df):
    """Test update_ratings handles multiple rows"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    writer.update_ratings(multiple_ratings_df)

    # Should execute 3 times for table creation + 3 times for data
    assert mock_cursor.execute.call_count >= 6


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_update_ratings_uses_upsert(mock_validator_class, mock_makedirs, mock_connect, sample_ratings_df):
    """Test update_ratings uses INSERT...ON CONFLICT"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    writer.update_ratings(sample_ratings_df)

    # Check that ON CONFLICT is used
    execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
    ratings_inserts = [call for call in execute_calls if 'INSERT INTO ratings' in call]

    assert len(ratings_inserts) > 0
    assert 'ON CONFLICT' in ratings_inserts[0]


@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
@pytest.mark.parametrize("rating_value", ["1", "2", "3", "4", "5"])
def test_update_ratings_all_rating_values(mock_validator_class, mock_makedirs, mock_connect, rating_value):
    """Test update_ratings handles all valid rating values"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    df = pd.DataFrame([{
        'userid': 'user123',
        'movieid': 'movie456',
        'rating': rating_value,
        'rated_at': '2023-10-25 10:30:15'
    }])

    writer.update_ratings(df)

    assert mock_cursor.execute.call_count >= 4


# ============================================================================
# Process Files Tests - No Files
# ============================================================================

@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_handles_no_files(mock_validator_class, mock_makedirs, mock_connect, mock_glob):
    """Test process_files handles empty directory"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_glob.return_value = []

    writer = DBWriter()
    writer.process_files()

    # Should log but not process anything
    mock_glob.assert_called_once()


# ============================================================================
# Process Files Tests - Movie Files
# ============================================================================

@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_processes_movie_file(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_movie_csv_df):
    """Test process_files processes movie CSV file"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = ['/app/data_queues/movies_001.csv']
    mock_read_csv.return_value = sample_movie_csv_df

    mock_validator = MagicMock()
    mock_record = MagicMock()
    # Return complete WatchHistoryRecord schema fields
    mock_record.model_dump.return_value = {
        'user_id': 'user123',
        'movie_id': 'movie456',
        'total_watch_time': 2,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    mock_validator.validate_batch.return_value = ([mock_record], [])
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()

    # Should read the CSV
    mock_read_csv.assert_called_once_with('/app/data_queues/movies_001.csv')

    # Should validate the data
    mock_validator.validate_batch.assert_called_once()

    # Should remove the file
    mock_remove.assert_called_once_with('/app/data_queues/movies_001.csv')


@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_aggregates_movie_data(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_movie_csv_df):
    """Test process_files aggregates movie watch data correctly"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = ['/app/data_queues/movies_001.csv']
    mock_read_csv.return_value = sample_movie_csv_df

    mock_validator = MagicMock()
    mock_record = MagicMock()
    # Return complete WatchHistoryRecord schema fields
    mock_record.model_dump.return_value = {
        'user_id': 'user123',
        'movie_id': 'movie456',
        'total_watch_time': 2,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    mock_validator.validate_batch.return_value = ([mock_record], [])
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()

    # Verify validation was called with aggregated data
    call_args = mock_validator.validate_batch.call_args[0][0]
    assert isinstance(call_args, list)


# ============================================================================
# Process Files Tests - Rating Files
# ============================================================================

@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_processes_rating_file(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_rating_csv_df):
    """Test process_files processes rating CSV file"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = ['/app/data_queues/ratings_001.csv']
    mock_read_csv.return_value = sample_rating_csv_df

    mock_validator = MagicMock()
    mock_record = MagicMock()
    mock_record.model_dump.return_value = {
        'user_id': 'user789',
        'movie_id': 'movie101',
        'rating': '4',
        'rated_at': '2023-10-25 10:31:20'
    }
    mock_validator.validate_batch.return_value = ([mock_record], [])
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()

    # Should read the CSV
    mock_read_csv.assert_called_once_with('/app/data_queues/ratings_001.csv')

    # Should validate the data
    mock_validator.validate_batch.assert_called_once()

    # Should remove the file
    mock_remove.assert_called_once_with('/app/data_queues/ratings_001.csv')


@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_aggregates_rating_data(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_rating_csv_df):
    """Test process_files aggregates rating data correctly"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = ['/app/data_queues/ratings_001.csv']
    mock_read_csv.return_value = sample_rating_csv_df

    mock_validator = MagicMock()
    mock_record = MagicMock()
    mock_record.model_dump.return_value = {
        'user_id': 'user789',
        'movie_id': 'movie101',
        'rating': '4',
        'rated_at': '2023-10-25 10:31:20'
    }
    mock_validator.validate_batch.return_value = ([mock_record], [])
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()

    # Verify validation was called
    mock_validator.validate_batch.assert_called_once()


# ============================================================================
# Process Files Tests - Multiple Files
# ============================================================================

@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_handles_multiple_files(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_movie_csv_df, sample_rating_csv_df):
    """Test process_files handles multiple CSV files"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = [
        '/app/data_queues/movies_001.csv',
        '/app/data_queues/ratings_001.csv'
    ]
    mock_read_csv.side_effect = [sample_movie_csv_df, sample_rating_csv_df]

    mock_validator = MagicMock()

    # Create separate mock records for watch history and ratings
    mock_watch_record = MagicMock()
    mock_watch_record.model_dump.return_value = {
        'user_id': 'user123',
        'movie_id': 'movie456',
        'total_watch_time': 2,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }

    mock_rating_record = MagicMock()
    mock_rating_record.model_dump.return_value = {
        'user_id': 'user789',
        'movie_id': 'movie101',
        'rating': '4',
        'rated_at': '2023-10-25 10:31:20'
    }

    # Return different records for each validation call
    mock_validator.validate_batch.side_effect = [
        ([mock_watch_record], []),
        ([mock_rating_record], [])
    ]
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()

    # Should process both files
    assert mock_read_csv.call_count == 2
    assert mock_remove.call_count == 2


# ============================================================================
# Process Files Tests - Error Handling
# ============================================================================

@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_handles_empty_csv(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob):
    """Test process_files handles empty CSV files"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    mock_glob.return_value = ['/app/data_queues/movies_empty.csv']
    mock_read_csv.side_effect = pd.errors.EmptyDataError()

    writer = DBWriter()
    writer.process_files()

    # Should attempt to remove the empty file
    mock_remove.assert_called_once_with('/app/data_queues/movies_empty.csv')


@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_continues_on_error(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_movie_csv_df):
    """Test process_files continues processing after error"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = [
        '/app/data_queues/movies_bad.csv',
        '/app/data_queues/movies_good.csv'
    ]
    # First file throws error, second succeeds
    mock_read_csv.side_effect = [Exception("File error"), sample_movie_csv_df]

    mock_validator = MagicMock()
    mock_record = MagicMock()
    mock_record.model_dump.return_value = {
        'user_id': 'user123',
        'movie_id': 'movie456',
        'total_watch_time': 2,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    mock_validator.validate_batch.return_value = ([mock_record], [])
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()

    # Should attempt both files
    assert mock_read_csv.call_count == 2


@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_handles_validation_failures(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_movie_csv_df):
    """Test process_files handles validation failures gracefully"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = ['/app/data_queues/movies_001.csv']
    mock_read_csv.return_value = sample_movie_csv_df

    mock_validator = MagicMock()
    # All records invalid
    mock_validator.validate_batch.return_value = ([], [MagicMock(), MagicMock()])
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()

    # Should still remove the file
    mock_remove.assert_called_once_with('/app/data_queues/movies_001.csv')


# ============================================================================
# Close Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_close_closes_connection(mock_validator_class, mock_makedirs, mock_connect):
    """Test close method closes database connection"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    writer = DBWriter()
    writer.close()

    mock_conn.close.assert_called_once()


# ============================================================================
# Validation Stats Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_get_validation_stats_returns_stats(mock_validator_class, mock_makedirs, mock_connect):
    """Test get_validation_stats returns validator statistics"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    mock_validator = MagicMock()
    expected_stats = {'total': 100, 'valid': 95, 'invalid': 5}
    mock_validator.get_stats.return_value = expected_stats
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    stats = writer.get_validation_stats()

    assert stats == expected_stats
    mock_validator.get_stats.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_full_pipeline_movies_and_ratings(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob, sample_movie_csv_df, sample_rating_csv_df):
    """Test complete pipeline with both movie and rating files"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    mock_glob.return_value = [
        '/app/data_queues/movies_001.csv',
        '/app/data_queues/ratings_001.csv'
    ]
    mock_read_csv.side_effect = [sample_movie_csv_df, sample_rating_csv_df]

    mock_validator = MagicMock()

    # Create separate mock records for watch history and ratings
    mock_watch_record = MagicMock()
    mock_watch_record.model_dump.return_value = {
        'user_id': 'user123',
        'movie_id': 'movie456',
        'total_watch_time': 2,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }

    mock_rating_record = MagicMock()
    mock_rating_record.model_dump.return_value = {
        'user_id': 'user789',
        'movie_id': 'movie101',
        'rating': '4',
        'rated_at': '2023-10-25 10:31:20'
    }

    # Return different records for each validation call
    mock_validator.validate_batch.side_effect = [
        ([mock_watch_record], []),
        ([mock_rating_record], [])
    ]
    mock_validator_class.return_value = mock_validator

    writer = DBWriter()
    writer.process_files()
    stats = writer.get_validation_stats()
    writer.close()

    # Should process both files
    assert mock_read_csv.call_count == 2
    assert mock_remove.call_count == 2

    # Should get stats and close
    mock_validator.get_stats.assert_called_once()
    mock_conn.close.assert_called_once()


# ============================================================================
# Edge Case Tests
# ============================================================================

@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_ignores_non_csv_files(mock_validator_class, mock_makedirs, mock_connect, mock_glob):
    """Test process_files only processes CSV files"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    # glob is configured to only return .csv files
    mock_glob.return_value = []

    writer = DBWriter()
    writer.process_files()

    # glob should be called with *.csv pattern
    mock_glob.assert_called_once_with(os.path.join('/app/data_queues/', '*.csv'))


@patch('homealone.data_processing.db_writer.glob.glob')
@patch('homealone.data_processing.db_writer.pd.read_csv')
@patch('homealone.data_processing.db_writer.os.remove')
@patch('homealone.data_processing.db_writer.sqlite3.connect')
@patch('homealone.data_processing.db_writer.os.makedirs')
@patch('homealone.data_processing.db_writer.ValidationManager')
def test_process_files_skips_unknown_file_types(mock_validator_class, mock_makedirs, mock_connect, mock_remove, mock_read_csv, mock_glob):
    """Test process_files skips files that don't match movies or ratings"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    mock_glob.return_value = ['/app/data_queues/other_001.csv']
    mock_read_csv.return_value = pd.DataFrame({'data': [1, 2, 3]})

    writer = DBWriter()
    writer.process_files()

    # Should read the file but not process it
    mock_read_csv.assert_not_called()
