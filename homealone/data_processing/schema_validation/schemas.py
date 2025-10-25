from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime

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
        if not v:
            raise ValueError('movieid must not be empty')
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
        if not v:
            raise ValueError('movieid must not be empty')
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
    def movie_id_valid(cls, v):
        if not v:
            raise ValueError('movie_id must not be empty')
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