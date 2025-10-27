"""
Tests for ValidationManager.
Tests the validation logic in schema_validation/validator.py
"""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from homealone.data_processing.schema_validation.validator import ValidationManager
from homealone.data_processing.schema_validation.schemas import (
    MovieEvent,
    RatingEvent,
    WatchHistoryRecord,
    RatingRecord
)


# ============================================================================
# Initialization Tests
# ============================================================================

def test_validation_manager_initializes():
    """Test ValidationManager initializes with correct defaults"""
    manager = ValidationManager()

    assert manager.failed_records == []
    assert manager.validation_stats['total_records'] == 0
    assert manager.validation_stats['valid_records'] == 0
    assert manager.validation_stats['invalid_records'] == 0


# ============================================================================
# Single Record Validation Tests
# ============================================================================

def test_validate_record_valid_movie_event():
    """Test validate_record accepts valid MovieEvent"""
    manager = ValidationManager()
    record = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'minute': 5
    }

    is_valid, validated, error = manager.validate_record(record, MovieEvent)

    assert is_valid is True
    assert validated is not None
    assert isinstance(validated, MovieEvent)
    assert validated.userid == 123
    assert error == ""
    assert manager.validation_stats['total_records'] == 1
    assert manager.validation_stats['valid_records'] == 1
    assert manager.validation_stats['invalid_records'] == 0


def test_validate_record_invalid_movie_event():
    """Test validate_record rejects invalid MovieEvent"""
    manager = ValidationManager()
    record = {
        'time': '2023-10-25 10:30:15',
        'userid': -1,  # Invalid: negative userid
        'movieid': 'movie456',
        'minute': 5
    }

    is_valid, validated, error = manager.validate_record(record, MovieEvent)

    assert is_valid is False
    assert validated is None
    assert error != ""
    assert 'userid must be positive' in error
    assert manager.validation_stats['total_records'] == 1
    assert manager.validation_stats['valid_records'] == 0
    assert manager.validation_stats['invalid_records'] == 1


def test_validate_record_stores_failed_record():
    """Test validate_record stores failed records"""
    manager = ValidationManager()
    record = {
        'time': '2023-10-25 10:30:15',
        'userid': 0,  # Invalid
        'movieid': 'movie456',
        'minute': 5
    }

    manager.validate_record(record, MovieEvent)

    assert len(manager.failed_records) == 1
    failed = manager.failed_records[0]
    assert failed['schema'] == 'MovieEvent'
    assert failed['record'] == record
    assert 'userid must be positive' in failed['error']


def test_validate_record_valid_rating_event():
    """Test validate_record accepts valid RatingEvent"""
    manager = ValidationManager()
    record = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'rating': 4.5
    }

    is_valid, validated, error = manager.validate_record(record, RatingEvent)

    assert is_valid is True
    assert isinstance(validated, RatingEvent)
    assert validated.rating == 4.5


def test_validate_record_invalid_rating_event():
    """Test validate_record rejects invalid RatingEvent"""
    manager = ValidationManager()
    record = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'rating': 6.0  # Invalid: out of range
    }

    is_valid, validated, error = manager.validate_record(record, RatingEvent)

    assert is_valid is False
    assert validated is None
    assert 'rating must be between 0 and 5' in error


def test_validate_record_missing_required_field():
    """Test validate_record rejects record with missing field"""
    manager = ValidationManager()
    record = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456'
        # missing 'minute'
    }

    is_valid, validated, error = manager.validate_record(record, MovieEvent)

    assert is_valid is False
    assert validated is None
    assert error != ""


# ============================================================================
# Batch Validation Tests
# ============================================================================

def test_validate_batch_all_valid():
    """Test validate_batch with all valid records"""
    manager = ValidationManager()
    records = [
        {
            'time': '2023-10-25 10:30:15',
            'userid': 123,
            'movieid': 'movie456',
            'minute': 5
        },
        {
            'time': '2023-10-25 10:30:16',
            'userid': 456,
            'movieid': 'movie789',
            'minute': 10
        }
    ]

    valid, failed = manager.validate_batch(records, MovieEvent)

    assert len(valid) == 2
    assert len(failed) == 0
    assert all(isinstance(v, MovieEvent) for v in valid)
    assert manager.validation_stats['total_records'] == 2
    assert manager.validation_stats['valid_records'] == 2
    assert manager.validation_stats['invalid_records'] == 0


def test_validate_batch_all_invalid():
    """Test validate_batch with all invalid records"""
    manager = ValidationManager()
    records = [
        {
            'time': '2023-10-25 10:30:15',
            'userid': -1,  # Invalid
            'movieid': 'movie456',
            'minute': 5
        },
        {
            'time': '2023-10-25 10:30:16',
            'userid': 0,  # Invalid
            'movieid': 'movie789',
            'minute': 10
        }
    ]

    valid, failed = manager.validate_batch(records, MovieEvent)

    assert len(valid) == 0
    assert len(failed) == 2
    assert manager.validation_stats['total_records'] == 2
    assert manager.validation_stats['valid_records'] == 0
    assert manager.validation_stats['invalid_records'] == 2


def test_validate_batch_mixed_validity():
    """Test validate_batch with mix of valid and invalid records"""
    manager = ValidationManager()
    records = [
        {
            'time': '2023-10-25 10:30:15',
            'userid': 123,  # Valid
            'movieid': 'movie456',
            'minute': 5
        },
        {
            'time': '2023-10-25 10:30:16',
            'userid': -1,  # Invalid
            'movieid': 'movie789',
            'minute': 10
        },
        {
            'time': '2023-10-25 10:30:17',
            'userid': 789,  # Valid
            'movieid': 'movie101',
            'minute': 3
        }
    ]

    valid, failed = manager.validate_batch(records, MovieEvent)

    assert len(valid) == 2
    assert len(failed) == 1
    assert valid[0].userid == 123
    assert valid[1].userid == 789
    assert failed[0]['record']['userid'] == -1
    assert manager.validation_stats['total_records'] == 3
    assert manager.validation_stats['valid_records'] == 2
    assert manager.validation_stats['invalid_records'] == 1


def test_validate_batch_empty_list():
    """Test validate_batch with empty list"""
    manager = ValidationManager()
    records = []

    valid, failed = manager.validate_batch(records, MovieEvent)

    assert len(valid) == 0
    assert len(failed) == 0
    assert manager.validation_stats['total_records'] == 0


def test_validate_batch_with_rating_events():
    """Test validate_batch with RatingEvent schema"""
    manager = ValidationManager()
    records = [
        {
            'time': '2023-10-25 10:30:15',
            'userid': 123,
            'movieid': 'movie456',
            'rating': 4.5
        },
        {
            'time': '2023-10-25 10:30:16',
            'userid': 456,
            'movieid': 'movie789',
            'rating': 10.0  # Invalid
        }
    ]

    valid, failed = manager.validate_batch(records, RatingEvent)

    assert len(valid) == 1
    assert len(failed) == 1
    assert valid[0].rating == 4.5


def test_validate_batch_with_watch_history_records():
    """Test validate_batch with WatchHistoryRecord schema"""
    manager = ValidationManager()
    records = [
        {
            'user_id': 123,
            'movie_id': 'movie456',
            'total_watch_time': 50,
            'start_time': '2023-10-25 10:30:15',
            'last_watch': '2023-10-25 10:35:20'
        },
        {
            'user_id': 456,
            'movie_id': 'movie789',
            'total_watch_time': 100,
            'start_time': '2023-10-25 11:00:00',
            'last_watch': '2023-10-25 11:10:00'
        }
    ]

    valid, failed = manager.validate_batch(records, WatchHistoryRecord)

    assert len(valid) == 2
    assert len(failed) == 0
    assert all(isinstance(v, WatchHistoryRecord) for v in valid)


# ============================================================================
# Statistics Tests
# ============================================================================

def test_get_stats_initial_state():
    """Test get_stats returns correct initial statistics"""
    manager = ValidationManager()

    stats = manager.get_stats()

    assert stats['total_records'] == 0
    assert stats['valid_records'] == 0
    assert stats['invalid_records'] == 0
    assert stats['valid_percentage'] == 0
    assert stats['failed_records_count'] == 0


def test_get_stats_after_validation():
    """Test get_stats returns correct statistics after validation"""
    manager = ValidationManager()
    records = [
        {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': -1, 'movieid': 'movie789', 'minute': 10},
        {'time': '2023-10-25 10:30:17', 'userid': 789, 'movieid': 'movie101', 'minute': 3},
        {'time': '2023-10-25 10:30:18', 'userid': 0, 'movieid': 'movie202', 'minute': 7}
    ]

    manager.validate_batch(records, MovieEvent)
    stats = manager.get_stats()

    assert stats['total_records'] == 4
    assert stats['valid_records'] == 2
    assert stats['invalid_records'] == 2
    assert stats['valid_percentage'] == 50.0
    assert stats['failed_records_count'] == 2


def test_get_stats_valid_percentage_calculation():
    """Test get_stats calculates valid_percentage correctly"""
    manager = ValidationManager()
    records = [
        {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': 456, 'movieid': 'movie789', 'minute': 10},
        {'time': '2023-10-25 10:30:17', 'userid': 789, 'movieid': 'movie101', 'minute': 3},
        {'time': '2023-10-25 10:30:18', 'userid': -1, 'movieid': 'movie202', 'minute': 7}
    ]

    manager.validate_batch(records, MovieEvent)
    stats = manager.get_stats()

    assert stats['valid_percentage'] == 75.0


def test_get_stats_all_valid():
    """Test get_stats with 100% valid records"""
    manager = ValidationManager()
    records = [
        {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': 456, 'movieid': 'movie789', 'minute': 10}
    ]

    manager.validate_batch(records, MovieEvent)
    stats = manager.get_stats()

    assert stats['valid_percentage'] == 100.0
    assert stats['failed_records_count'] == 0


def test_get_stats_all_invalid():
    """Test get_stats with 0% valid records"""
    manager = ValidationManager()
    records = [
        {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': 0, 'movieid': 'movie789', 'minute': 10}
    ]

    manager.validate_batch(records, MovieEvent)
    stats = manager.get_stats()

    assert stats['valid_percentage'] == 0.0
    assert stats['failed_records_count'] == 2


# ============================================================================
# Failed Records Management Tests
# ============================================================================

def test_get_failed_records_empty():
    """Test get_failed_records returns empty list initially"""
    manager = ValidationManager()

    failed = manager.get_failed_records()

    assert failed == []


def test_get_failed_records_after_failures():
    """Test get_failed_records returns all failed records"""
    manager = ValidationManager()
    records = [
        {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': 0, 'movieid': 'movie789', 'minute': 10}
    ]

    manager.validate_batch(records, MovieEvent)
    failed = manager.get_failed_records()

    assert len(failed) == 2
    assert all('schema' in f and 'record' in f and 'error' in f for f in failed)


def test_get_failed_records_contains_schema_info():
    """Test get_failed_records includes schema name"""
    manager = ValidationManager()
    record = {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5}

    manager.validate_record(record, MovieEvent)
    failed = manager.get_failed_records()

    assert failed[0]['schema'] == 'MovieEvent'


def test_get_failed_records_contains_original_record():
    """Test get_failed_records includes original record data"""
    manager = ValidationManager()
    record = {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5}

    manager.validate_record(record, MovieEvent)
    failed = manager.get_failed_records()

    assert failed[0]['record'] == record


def test_get_failed_records_contains_error_message():
    """Test get_failed_records includes error message"""
    manager = ValidationManager()
    record = {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5}

    manager.validate_record(record, MovieEvent)
    failed = manager.get_failed_records()

    assert 'userid must be positive' in failed[0]['error']


def test_clear_failed_records():
    """Test clear_failed_records empties the failed records list"""
    manager = ValidationManager()
    records = [
        {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': 0, 'movieid': 'movie789', 'minute': 10}
    ]

    manager.validate_batch(records, MovieEvent)
    assert len(manager.get_failed_records()) == 2

    manager.clear_failed_records()

    assert manager.get_failed_records() == []
    # Stats should remain unchanged
    assert manager.validation_stats['invalid_records'] == 2


def test_clear_failed_records_multiple_times():
    """Test clear_failed_records can be called multiple times"""
    manager = ValidationManager()
    record = {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5}

    manager.validate_record(record, MovieEvent)
    manager.clear_failed_records()
    manager.clear_failed_records()  # Should not error

    assert manager.get_failed_records() == []


# ============================================================================
# Cumulative Statistics Tests
# ============================================================================

def test_validation_stats_are_cumulative():
    """Test that validation stats accumulate across multiple validations"""
    manager = ValidationManager()

    # First batch
    records1 = [
        {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': -1, 'movieid': 'movie789', 'minute': 10}
    ]
    manager.validate_batch(records1, MovieEvent)

    # Second batch
    records2 = [
        {'time': '2023-10-25 10:30:17', 'userid': 456, 'movieid': 'movie101', 'minute': 3},
        {'time': '2023-10-25 10:30:18', 'userid': 789, 'movieid': 'movie202', 'minute': 7}
    ]
    manager.validate_batch(records2, MovieEvent)

    stats = manager.get_stats()

    assert stats['total_records'] == 4
    assert stats['valid_records'] == 3
    assert stats['invalid_records'] == 1


def test_failed_records_are_cumulative():
    """Test that failed records accumulate"""
    manager = ValidationManager()

    record1 = {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5}
    manager.validate_record(record1, MovieEvent)

    record2 = {'time': '2023-10-25 10:30:16', 'userid': 0, 'movieid': 'movie789', 'minute': 10}
    manager.validate_record(record2, MovieEvent)

    assert len(manager.get_failed_records()) == 2


# ============================================================================
# Logging Tests
# ============================================================================

@patch('homealone.data_processing.schema_validation.validator.logger')
def test_validate_record_logs_warning_on_failure(mock_logger):
    """Test validate_record logs warning when validation fails"""
    manager = ValidationManager()
    record = {'time': '2023-10-25 10:30:15', 'userid': -1, 'movieid': 'movie456', 'minute': 5}

    manager.validate_record(record, MovieEvent)

    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[0][0]
    assert 'Validation failed' in call_args
    assert 'MovieEvent' in call_args


@patch('homealone.data_processing.schema_validation.validator.logger')
def test_validate_record_does_not_log_on_success(mock_logger):
    """Test validate_record does not log on successful validation"""
    manager = ValidationManager()
    record = {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'minute': 5}

    manager.validate_record(record, MovieEvent)

    mock_logger.warning.assert_not_called()


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_validation_workflow():
    """Test complete validation workflow with multiple schemas"""
    manager = ValidationManager()

    # Validate movie events
    movie_records = [
        {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'minute': 5},
        {'time': '2023-10-25 10:30:16', 'userid': -1, 'movieid': 'movie789', 'minute': 10}
    ]
    valid_movies, failed_movies = manager.validate_batch(movie_records, MovieEvent)

    # Validate rating events
    rating_records = [
        {'time': '2023-10-25 10:30:17', 'userid': 123, 'movieid': 'movie456', 'rating': 4.5},
        {'time': '2023-10-25 10:30:18', 'userid': 456, 'movieid': 'movie789', 'rating': 10.0}
    ]
    valid_ratings, failed_ratings = manager.validate_batch(rating_records, RatingEvent)

    # Check results
    assert len(valid_movies) == 1
    assert len(failed_movies) == 1
    assert len(valid_ratings) == 1
    assert len(failed_ratings) == 1

    # Check cumulative stats
    stats = manager.get_stats()
    assert stats['total_records'] == 4
    assert stats['valid_records'] == 2
    assert stats['invalid_records'] == 2
    assert stats['valid_percentage'] == 50.0

    # Check failed records
    failed = manager.get_failed_records()
    assert len(failed) == 2

    # Clear and verify
    manager.clear_failed_records()
    assert len(manager.get_failed_records()) == 0


def test_validation_with_different_schemas_simultaneously():
    """Test validation manager can handle different schemas in same session"""
    manager = ValidationManager()

    # Movie event
    movie = {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'minute': 5}
    is_valid_movie, _, _ = manager.validate_record(movie, MovieEvent)

    # Rating event
    rating = {'time': '2023-10-25 10:30:15', 'userid': 123, 'movieid': 'movie456', 'rating': 4.5}
    is_valid_rating, _, _ = manager.validate_record(rating, RatingEvent)

    # Watch history record
    watch = {
        'user_id': 123,
        'movie_id': 'movie456',
        'total_watch_time': 50,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    is_valid_watch, _, _ = manager.validate_record(watch, WatchHistoryRecord)

    # All should be valid
    assert is_valid_movie is True
    assert is_valid_rating is True
    assert is_valid_watch is True

    # Stats should reflect all validations
    assert manager.validation_stats['total_records'] == 3
    assert manager.validation_stats['valid_records'] == 3
