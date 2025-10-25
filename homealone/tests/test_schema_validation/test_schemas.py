"""
Tests for schema validation schemas.
Tests all Pydantic models in schema_validation/schemas.py
"""

import pytest
from pydantic import ValidationError
from homealone.data_processing.schema_validation.schemas import (
    MovieEvent,
    RatingEvent,
    RecommendationEvent,
    WatchHistoryRecord,
    RatingRecord,
    MovieMetadata
)


# ============================================================================
# MovieEvent Tests
# ============================================================================

def test_movie_event_valid():
    """Test MovieEvent with valid data"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'minute': 5
    }
    event = MovieEvent(**data)
    assert event.userid == 123
    assert event.movieid == 'movie456'
    assert event.minute == 5
    assert event.time == '2023-10-25 10:30:15'


def test_movie_event_userid_must_be_positive():
    """Test MovieEvent rejects non-positive userid"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 0,
        'movieid': 'movie456',
        'minute': 5
    }
    with pytest.raises(ValidationError) as exc_info:
        MovieEvent(**data)
    assert 'userid must be positive' in str(exc_info.value)


def test_movie_event_userid_negative():
    """Test MovieEvent rejects negative userid"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': -1,
        'movieid': 'movie456',
        'minute': 5
    }
    with pytest.raises(ValidationError) as exc_info:
        MovieEvent(**data)
    assert 'userid must be positive' in str(exc_info.value)


def test_movie_event_movieid_cannot_be_empty():
    """Test MovieEvent rejects empty movieid"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': '',
        'minute': 5
    }
    with pytest.raises(ValidationError) as exc_info:
        MovieEvent(**data)
    assert 'movieid must not be empty' in str(exc_info.value)


def test_movie_event_minute_must_be_non_negative():
    """Test MovieEvent rejects negative minute"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'minute': -1
    }
    with pytest.raises(ValidationError) as exc_info:
        MovieEvent(**data)
    assert 'minute must be non-negative' in str(exc_info.value)


def test_movie_event_minute_zero_is_valid():
    """Test MovieEvent accepts minute = 0"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'minute': 0
    }
    event = MovieEvent(**data)
    assert event.minute == 0


def test_movie_event_missing_required_field():
    """Test MovieEvent rejects missing required fields"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456'
        # missing 'minute'
    }
    with pytest.raises(ValidationError):
        MovieEvent(**data)


# ============================================================================
# RatingEvent Tests
# ============================================================================

def test_rating_event_valid():
    """Test RatingEvent with valid data"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'rating': 4.5
    }
    event = RatingEvent(**data)
    assert event.userid == 123
    assert event.movieid == 'movie456'
    assert event.rating == 4.5
    assert event.time == '2023-10-25 10:30:15'


@pytest.mark.parametrize("rating_value", [0, 1, 2, 3, 4, 5, 0.0, 2.5, 5.0])
def test_rating_event_valid_rating_range(rating_value):
    """Test RatingEvent accepts all valid ratings (0-5)"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'rating': rating_value
    }
    event = RatingEvent(**data)
    assert event.rating == rating_value


@pytest.mark.parametrize("rating_value", [-1, -0.1, 5.1, 6, 10])
def test_rating_event_invalid_rating_range(rating_value):
    """Test RatingEvent rejects ratings outside 0-5 range"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': 'movie456',
        'rating': rating_value
    }
    with pytest.raises(ValidationError) as exc_info:
        RatingEvent(**data)
    assert 'rating must be between 0 and 5' in str(exc_info.value)


def test_rating_event_userid_must_be_positive():
    """Test RatingEvent rejects non-positive userid"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 0,
        'movieid': 'movie456',
        'rating': 4.0
    }
    with pytest.raises(ValidationError) as exc_info:
        RatingEvent(**data)
    assert 'userid must be positive' in str(exc_info.value)


def test_rating_event_movieid_cannot_be_empty():
    """Test RatingEvent rejects empty movieid"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'movieid': '',
        'rating': 4.0
    }
    with pytest.raises(ValidationError) as exc_info:
        RatingEvent(**data)
    assert 'movieid must not be empty' in str(exc_info.value)


# ============================================================================
# RecommendationEvent Tests
# ============================================================================

def test_recommendation_event_valid():
    """Test RecommendationEvent with valid data"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'server': 'server1',
        'status': 200,
        'recommendations': 'movie1,movie2,movie3',
        'responsetime': 150.5
    }
    event = RecommendationEvent(**data)
    assert event.userid == 123
    assert event.server == 'server1'
    assert event.status == 200
    assert event.recommendations == 'movie1,movie2,movie3'
    assert event.responsetime == 150.5


@pytest.mark.parametrize("status_code", [200, 404, 500])
def test_recommendation_event_valid_status_codes(status_code):
    """Test RecommendationEvent accepts valid status codes"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'server': 'server1',
        'status': status_code,
        'recommendations': 'movie1,movie2',
        'responsetime': 100.0
    }
    event = RecommendationEvent(**data)
    assert event.status == status_code


@pytest.mark.parametrize("status_code", [100, 201, 400, 403, 503])
def test_recommendation_event_invalid_status_codes(status_code):
    """Test RecommendationEvent rejects invalid status codes"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 123,
        'server': 'server1',
        'status': status_code,
        'recommendations': 'movie1,movie2',
        'responsetime': 100.0
    }
    with pytest.raises(ValidationError) as exc_info:
        RecommendationEvent(**data)
    assert 'not in allowed values' in str(exc_info.value)


def test_recommendation_event_userid_must_be_positive():
    """Test RecommendationEvent rejects non-positive userid"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': -1,
        'server': 'server1',
        'status': 200,
        'recommendations': 'movie1,movie2',
        'responsetime': 100.0
    }
    with pytest.raises(ValidationError) as exc_info:
        RecommendationEvent(**data)
    assert 'userid must be positive' in str(exc_info.value)


# ============================================================================
# WatchHistoryRecord Tests
# ============================================================================

def test_watch_history_record_valid():
    """Test WatchHistoryRecord with valid data"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'total_watch_time': 50,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    record = WatchHistoryRecord(**data)
    assert record.user_id == 123
    assert record.movie_id == 'movie456'
    assert record.total_watch_time == 50
    assert record.start_time == '2023-10-25 10:30:15'
    assert record.last_watch == '2023-10-25 10:35:20'


def test_watch_history_record_default_watch_time():
    """Test WatchHistoryRecord defaults total_watch_time to 0"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    record = WatchHistoryRecord(**data)
    assert record.total_watch_time == 0


def test_watch_history_record_user_id_must_be_positive():
    """Test WatchHistoryRecord rejects non-positive user_id"""
    data = {
        'user_id': 0,
        'movie_id': 'movie456',
        'total_watch_time': 50,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    with pytest.raises(ValidationError) as exc_info:
        WatchHistoryRecord(**data)
    assert 'user_id must be positive' in str(exc_info.value)


def test_watch_history_record_movie_id_cannot_be_empty():
    """Test WatchHistoryRecord rejects empty movie_id"""
    data = {
        'user_id': 123,
        'movie_id': '',
        'total_watch_time': 50,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    with pytest.raises(ValidationError) as exc_info:
        WatchHistoryRecord(**data)
    assert 'movie_id must not be empty' in str(exc_info.value)


def test_watch_history_record_watch_time_must_be_non_negative():
    """Test WatchHistoryRecord rejects negative total_watch_time"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'total_watch_time': -1,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    with pytest.raises(ValidationError) as exc_info:
        WatchHistoryRecord(**data)
    assert 'total_watch_time must be non-negative' in str(exc_info.value)


def test_watch_history_record_zero_watch_time_is_valid():
    """Test WatchHistoryRecord accepts total_watch_time = 0"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'total_watch_time': 0,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    record = WatchHistoryRecord(**data)
    assert record.total_watch_time == 0


# ============================================================================
# RatingRecord Tests
# ============================================================================

def test_rating_record_valid():
    """Test RatingRecord with valid data"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'rating': 4.5,
        'rated_at': '2023-10-25 10:30:15'
    }
    record = RatingRecord(**data)
    assert record.user_id == 123
    assert record.movie_id == 'movie456'
    assert record.rating == 4.5
    assert record.rated_at == '2023-10-25 10:30:15'


def test_rating_record_rating_is_optional():
    """Test RatingRecord allows None rating"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'rated_at': '2023-10-25 10:30:15'
    }
    record = RatingRecord(**data)
    assert record.rating is None


def test_rating_record_rating_none_explicit():
    """Test RatingRecord accepts explicit None rating"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'rating': None,
        'rated_at': '2023-10-25 10:30:15'
    }
    record = RatingRecord(**data)
    assert record.rating is None


@pytest.mark.parametrize("rating_value", [0, 1, 2, 3, 4, 5, 0.0, 2.5, 5.0])
def test_rating_record_valid_rating_range(rating_value):
    """Test RatingRecord accepts all valid ratings (0-5)"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'rating': rating_value,
        'rated_at': '2023-10-25 10:30:15'
    }
    record = RatingRecord(**data)
    assert record.rating == rating_value


@pytest.mark.parametrize("rating_value", [-1, -0.1, 5.1, 6, 10])
def test_rating_record_invalid_rating_range(rating_value):
    """Test RatingRecord rejects ratings outside 0-5 range"""
    data = {
        'user_id': 123,
        'movie_id': 'movie456',
        'rating': rating_value,
        'rated_at': '2023-10-25 10:30:15'
    }
    with pytest.raises(ValidationError) as exc_info:
        RatingRecord(**data)
    assert 'rating must be between 0 and 5' in str(exc_info.value)


# ============================================================================
# MovieMetadata Tests
# ============================================================================

def test_movie_metadata_minimal():
    """Test MovieMetadata with only required field"""
    data = {
        'movie_id': 'movie123'
    }
    metadata = MovieMetadata(**data)
    assert metadata.movie_id == 'movie123'
    assert metadata.tmdb_id is None
    assert metadata.title is None


def test_movie_metadata_full():
    """Test MovieMetadata with all fields"""
    data = {
        'movie_id': 'movie123',
        'tmdb_id': 550,
        'imdb_id': 'tt0137523',
        'adult': False,
        'genres': 'Drama,Thriller',
        'overview': 'A great movie',
        'original_language': 'en',
        'popularity': 85.5,
        'production_companies': 'Company A',
        'release_date': '1999-10-15',
        'revenue': 100000000,
        'budget': 63000000,
        'runtime': 139,
        'spoken_languages': 'English',
        'belongs_to_collection': None,
        'vote_average': 8.4,
        'vote_count': 20000,
        'video': False,
        'title': 'Fight Club',
        'production_countries': 'USA',
        'status': 'Released',
        'tagline': 'Mischief. Mayhem. Soap.'
    }
    metadata = MovieMetadata(**data)
    assert metadata.movie_id == 'movie123'
    assert metadata.tmdb_id == 550
    assert metadata.title == 'Fight Club'
    assert metadata.runtime == 139


def test_movie_metadata_partial_fields():
    """Test MovieMetadata with some optional fields"""
    data = {
        'movie_id': 'movie456',
        'title': 'The Matrix',
        'runtime': 136,
        'vote_average': 8.7
    }
    metadata = MovieMetadata(**data)
    assert metadata.movie_id == 'movie456'
    assert metadata.title == 'The Matrix'
    assert metadata.runtime == 136
    assert metadata.vote_average == 8.7
    assert metadata.tmdb_id is None
    assert metadata.budget is None


def test_movie_metadata_all_optional_fields_none():
    """Test MovieMetadata accepts None for all optional fields"""
    data = {
        'movie_id': 'movie789',
        'tmdb_id': None,
        'imdb_id': None,
        'adult': None,
        'genres': None,
        'overview': None,
        'original_language': None,
        'popularity': None,
        'production_companies': None,
        'release_date': None,
        'revenue': None,
        'budget': None,
        'runtime': None,
        'spoken_languages': None,
        'belongs_to_collection': None,
        'vote_average': None,
        'vote_count': None,
        'video': None,
        'title': None,
        'production_countries': None,
        'status': None,
        'tagline': None
    }
    metadata = MovieMetadata(**data)
    assert metadata.movie_id == 'movie789'
    assert all([
        metadata.tmdb_id is None,
        metadata.imdb_id is None,
        metadata.title is None
    ])


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_movie_event_type_coercion():
    """Test MovieEvent coerces string to int where possible"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': '123',  # String that can be coerced to int
        'movieid': 'movie456',
        'minute': '5'  # String that can be coerced to int
    }
    event = MovieEvent(**data)
    assert event.userid == 123
    assert event.minute == 5


def test_rating_event_type_coercion():
    """Test RatingEvent coerces string to numeric types"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': '123',
        'movieid': 'movie456',
        'rating': '4.5'
    }
    event = RatingEvent(**data)
    assert event.userid == 123
    assert event.rating == 4.5


def test_watch_history_record_type_coercion():
    """Test WatchHistoryRecord coerces string types"""
    data = {
        'user_id': '123',
        'movie_id': 'movie456',
        'total_watch_time': '50',
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }
    record = WatchHistoryRecord(**data)
    assert record.user_id == 123
    assert record.total_watch_time == 50


def test_invalid_type_raises_validation_error():
    """Test that completely invalid types raise ValidationError"""
    data = {
        'time': '2023-10-25 10:30:15',
        'userid': 'not_a_number',  # Cannot be coerced to int
        'movieid': 'movie456',
        'minute': 5
    }
    with pytest.raises(ValidationError):
        MovieEvent(**data)
