from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
import re

# ===== Helper regex =====
MOVIE_ID_RE = r"^(?!\+)(?!.*\+\+)(?!.*\+$)[\w\-.'&:()]+(?:\+[\w\-.'&:()]+)*$"
MOVIE_ID_MAXLEN = 150

# ===== Raw Event Schemas (from Kafka) =====
class MovieEvent(BaseModel):
    """Schema for movie watching events"""
    time: str
    userid: int
    movieid: str
    minute: int

    @field_validator('userid')
    def userid_positive(cls, v):
        if v <= 0:
            raise ValueError('userid must be positive')
        return v

    @field_validator('movieid')
    def movieid_valid(cls, v):
        if not isinstance(v, str):  # normalize
            v = str(v)
        v = v.strip()
        if not v:
            raise ValueError('movieid must not be empty')
        if len(v) > MOVIE_ID_MAXLEN:
            raise ValueError('movieid too long')
        if not re.match(MOVIE_ID_RE, v, flags=re.UNICODE):
            raise ValueError(
                'movieid is malformed (invalid + usage or characters)')
        return v

    @field_validator('minute')
    def minute_valid(cls, v):
        if v < 0:
            raise ValueError('minute must be non-negative')
        return v


class RatingEvent(BaseModel):
    """Schema for user rating events"""
    time: str
    userid: int
    movieid: str
    rating: float

    @field_validator('userid')
    def userid_positive(cls, v):
        if v <= 0:
            raise ValueError('userid must be positive')
        return v

    @field_validator('movieid')
    def movieid_valid(cls, v):
        if not isinstance(v, str):  # normalize
            v = str(v)
        v = v.strip()
        if not v:
            raise ValueError('movieid must not be empty')
        if len(v) > MOVIE_ID_MAXLEN:
            raise ValueError('movieid too long')
        if not re.match(MOVIE_ID_RE, v, flags=re.UNICODE):
            raise ValueError(
                'movieid is malformed (invalid + usage or characters)')
        return v

    @field_validator('rating')
    def rating_range(cls, v):
        if not (0 <= v <= 5):
            raise ValueError('rating must be between 0 and 5')
        return v


class RecommendationEvent(BaseModel):
    """Schema for recommendation request events"""
    time: str
    userid: int
    server: str
    status: int
    recommendations: str  # comma-separated movie IDs
    responsetime: float

    @field_validator('userid')
    def userid_positive(cls, v):
        if v <= 0:
            raise ValueError('userid must be positive')
        return v

    @field_validator('status')
    def status_valid(cls, v):
        if v not in [200, 404, 500]:
            raise ValueError(f'status {v} not in allowed values')
        return v

    @field_validator('recommendations')
    def recommendations_ids_valid(cls, v):
        ids = [s.strip() for s in v.split(',')] if v else []
        # Empty list is allowed for certain statuses; see cross-field rule below
        for mid in ids:
            if not re.match(MOVIE_ID_RE, mid, flags=re.UNICODE):
                raise ValueError(
                    f"recommendations contain malformed movie_id: {mid}")
        # Keep normalized, de-duplicated string
        return ",".join(dict.fromkeys(ids))  # preserves order, removes dupes

    @field_validator('responsetime')
    def response_time_nonneg(cls, v):
        if v < 0 or v > 60_000:  # prevent absurd latencies in ms
            raise ValueError('responsetime out of sensible range')
        return v


# ===== Aggregated Database Schemas =====
class WatchHistoryRecord(BaseModel):
    """Schema for watch_history table"""
    user_id: int
    movie_id: str
    total_watch_time: int = 0
    start_time: str
    last_watch: str

    @field_validator('user_id')
    def user_id_positive(cls, v):
        if v <= 0:
            raise ValueError('user_id must be positive')
        return v

    @field_validator('movie_id')
    def movieid_valid(cls, v):
        if not isinstance(v, str):  # normalize
            v = str(v)
        v = v.strip()
        if not v:
            raise ValueError('movie_id must not be empty')
        if len(v) > MOVIE_ID_MAXLEN:
            raise ValueError('movieid too long')
        if not re.match(MOVIE_ID_RE, v, flags=re.UNICODE):
            raise ValueError(
                'movieid is malformed (invalid + usage or characters)')
        return v

    @field_validator('total_watch_time')
    def watch_time_valid(cls, v):
        if v < 0:
            raise ValueError('total_watch_time must be non-negative')
        return v


class RatingRecord(BaseModel):
    """Schema for ratings table"""
    user_id: int
    movie_id: str
    rating: Optional[float] = None
    rated_at: str

    @field_validator('rating')
    def rating_range(cls, v):
        if v is not None and not (0 <= v <= 5):
            raise ValueError('rating must be between 0 and 5')
        return v

    @field_validator('user_id')
    def user_id_positive(cls, v):
        if v <= 0:
            raise ValueError('user_id must be positive')
        return v

    @field_validator('movie_id')
    def movieid_valid(cls, v):
        if not isinstance(v, str):  # normalize
            v = str(v)
        v = v.strip()
        if not v:
            raise ValueError('movie_id must not be empty')
        if len(v) > MOVIE_ID_MAXLEN:
            raise ValueError('movieid too long')
        if not re.match(MOVIE_ID_RE, v, flags=re.UNICODE):
            raise ValueError(
                'movieid is malformed (invalid + usage or characters)')
        return v


class MovieMetadata(BaseModel):
    """Schema for movies table"""
    movie_id: str
    tmdb_id: Optional[int] = None
    imdb_id: Optional[str] = None
    adult: Optional[bool] = None
    genres: Optional[str] = None
    overview: Optional[str] = None
    original_language: Optional[str] = None
    popularity: Optional[float] = None
    production_companies: Optional[str] = None
    release_date: Optional[str] = None
    revenue: Optional[int] = None
    budget: Optional[int] = None
    runtime: Optional[int] = None
    spoken_languages: Optional[str] = None
    belongs_to_collection: Optional[str] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    video: Optional[bool] = None
    title: Optional[str] = None
    production_countries: Optional[str] = None
    status: Optional[str] = None
    tagline: Optional[str] = None

    @field_validator('tmdb_id')
    def tmdb_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('tmdb_id must be positive')
        return v


    @field_validator('imdb_id')
    def imdb_pattern(cls, v):
        if v is not None and not re.fullmatch(r"tt\d{7,8}", v):
            raise ValueError('imdb_id must look like tt1234567')
        return v
