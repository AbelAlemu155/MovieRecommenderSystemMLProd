# """
# Tests for feature transformation functions in the training pipeline.

# These tests cover:
# 1. Data loading and basic transformations (user_id, movie_id, rating)
# 2. Surprise Reader transformations (rating scale normalization)
# 3. Dataset creation and trainset/testset building
# 4. ID mapping between raw IDs and internal trainset IDs
# """

# import pytest
# import pandas as pd
# import sqlite3
# import tempfile
# import os
# from surprise import Dataset, Reader


# # ============================================================================
# # Fixtures
# # ============================================================================

# @pytest.fixture
# def sample_ratings_data():
#     """Sample ratings data for testing transformations"""
#     return pd.DataFrame([
#         {'user_id': 'user1', 'movie_id': 'movie1', 'rating': 5},
#         {'user_id': 'user1', 'movie_id': 'movie2', 'rating': 4},
#         {'user_id': 'user2', 'movie_id': 'movie1', 'rating': 3},
#         {'user_id': 'user2', 'movie_id': 'movie3', 'rating': 2},
#         {'user_id': 'user3', 'movie_id': 'movie2', 'rating': 5},
#         {'user_id': 'user3', 'movie_id': 'movie3', 'rating': 1},
#     ])


# @pytest.fixture
# def sample_ratings_with_edge_cases():
#     """Sample ratings with edge cases (boundary values, duplicates)"""
#     return pd.DataFrame([
#         {'user_id': 'user1', 'movie_id': 'movie1', 'rating': 1},  # Min rating
#         {'user_id': 'user1', 'movie_id': 'movie2', 'rating': 5},  # Max rating
#         {'user_id': 'user2', 'movie_id': 'movie1', 'rating': 3},  # Duplicate user-movie
#         {'user_id': 'user2', 'movie_id': 'movie1', 'rating': 4},  # Duplicate (should keep last)
#         {'user_id': 'user3', 'movie_id': 'movie3', 'rating': 2.5}, # Float rating
#     ])


# @pytest.fixture
# def temp_db_with_ratings(sample_ratings_data):
#     """Create a temporary SQLite database with ratings table"""
#     # Create temporary database
#     fd, db_path = tempfile.mkstemp(suffix='.db')
#     os.close(fd)

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Create ratings table
#     cursor.execute('''
#         CREATE TABLE ratings (
#             user_id TEXT NOT NULL,
#             movie_id TEXT NOT NULL,
#             rating REAL NOT NULL,
#             rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     ''')

#     # Insert sample data
#     for _, row in sample_ratings_data.iterrows():
#         cursor.execute(
#             'INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)',
#             (row['user_id'], row['movie_id'], row['rating'])
#         )

#     conn.commit()
#     conn.close()

#     yield db_path

#     # Cleanup
#     if os.path.exists(db_path):
#         os.remove(db_path)


# # ============================================================================
# # Test: Data Loading from SQLite
# # ============================================================================

# def test_load_ratings_from_db(temp_db_with_ratings):
#     """Test loading ratings data from SQLite database"""
#     conn = sqlite3.connect(temp_db_with_ratings)

#     # Load data (same as train.py line 34)
#     query = "SELECT user_id, movie_id, rating FROM ratings"
#     df = pd.read_sql_query(query, conn)
#     conn.close()

#     # Assertions
#     assert len(df) == 6, "Should load 6 ratings"
#     assert list(df.columns) == ['user_id', 'movie_id', 'rating'], "Should have correct columns"
#     assert df['rating'].min() >= 1, "Ratings should be >= 1"
#     assert df['rating'].max() <= 5, "Ratings should be <= 5"
#     assert df['user_id'].notna().all(), "No null user_ids"
#     assert df['movie_id'].notna().all(), "No null movie_ids"


# def test_dataframe_structure(sample_ratings_data):
#     """Test that DataFrame has correct structure for Surprise"""
#     df = sample_ratings_data[['user_id', 'movie_id', 'rating']]

#     assert len(df.columns) == 3, "Should have exactly 3 columns"
#     assert 'user_id' in df.columns, "Should have user_id column"
#     assert 'movie_id' in df.columns, "Should have movie_id column"
#     assert 'rating' in df.columns, "Should have rating column"


# # ============================================================================
# # Test: Surprise Reader Transformation
# # ============================================================================

# def test_reader_rating_scale():
#     """Test that Reader correctly defines rating scale (1-5)"""
#     # This is the transformation in train.py line 40
#     reader = Reader(rating_scale=(1, 5))

#     assert reader.rating_scale == (1, 5), "Rating scale should be (1, 5)"


# def test_dataset_creation_from_dataframe(sample_ratings_data):
#     """Test creating Surprise Dataset from DataFrame"""
#     # This mirrors train.py lines 40-41
#     reader = Reader(rating_scale=(1, 5))
#     df = sample_ratings_data[['user_id', 'movie_id', 'rating']]
#     data = Dataset.load_from_df(df, reader)

#     assert data is not None, "Dataset should be created"
#     # Dataset should store the raw data
#     assert hasattr(data, 'raw_ratings'), "Dataset should have raw_ratings attribute"


# def test_dataset_with_edge_case_ratings(sample_ratings_with_edge_cases):
#     """Test dataset creation with boundary and edge case ratings"""
#     reader = Reader(rating_scale=(1, 5))
#     df = sample_ratings_with_edge_cases[['user_id', 'movie_id', 'rating']]
#     data = Dataset.load_from_df(df, reader)

#     # Should handle min/max ratings
#     assert data is not None

#     # Build trainset to verify transformations
#     trainset = data.build_full_trainset()
#     assert trainset.n_users > 0
#     assert trainset.n_items > 0


# # ============================================================================
# # Test: Trainset Building
# # ============================================================================

# def test_build_full_trainset(sample_ratings_data):
#     """Test building full trainset from dataset"""
#     # This is train.py line 44
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     assert trainset is not None, "Trainset should be created"
#     assert trainset.n_users == 3, "Should have 3 unique users"
#     assert trainset.n_items == 3, "Should have 3 unique items"
#     assert trainset.n_ratings == 6, "Should have 6 ratings"


# def test_trainset_rating_scale_transformation(sample_ratings_data):
#     """Test that trainset maintains rating scale information"""
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # Trainset should store rating scale
#     assert trainset.rating_scale == (1, 5), "Trainset should maintain rating scale"


# def test_build_anti_testset(sample_ratings_data):
#     """Test building anti-testset (unrated items) for evaluation"""
#     # This is train.py line 46
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()
#     testset = trainset.build_anti_testset()

#     assert testset is not None, "Testset should be created"
#     assert len(testset) > 0, "Anti-testset should contain unrated items"

#     # Anti-testset should contain items NOT in trainset
#     # With 3 users and 3 items, we have 9 possible pairs, 6 rated, so 3 unrated
#     assert len(testset) == 3, "Should have 3 unrated user-item pairs"


# # ============================================================================
# # Test: ID Mapping Transformations
# # ============================================================================

# def test_raw_to_inner_id_mapping(sample_ratings_data):
#     """Test transformation from raw IDs to internal trainset IDs"""
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # Test user ID mapping
#     raw_user_id = 'user1'
#     inner_user_id = trainset.to_inner_uid(raw_user_id)
#     assert isinstance(inner_user_id, int), "Inner user ID should be integer"
#     assert 0 <= inner_user_id < trainset.n_users, "Inner ID should be in valid range"

#     # Test movie ID mapping
#     raw_movie_id = 'movie1'
#     inner_movie_id = trainset.to_inner_iid(raw_movie_id)
#     assert isinstance(inner_movie_id, int), "Inner item ID should be integer"
#     assert 0 <= inner_movie_id < trainset.n_items, "Inner ID should be in valid range"


# def test_inner_to_raw_id_mapping(sample_ratings_data):
#     """Test reverse transformation from internal IDs to raw IDs"""
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # Test round-trip user ID conversion
#     raw_user_id = 'user2'
#     inner_id = trainset.to_inner_uid(raw_user_id)
#     converted_back = trainset.to_raw_uid(inner_id)
#     assert converted_back == raw_user_id, "Should convert back to original raw ID"

#     # Test round-trip item ID conversion
#     raw_movie_id = 'movie2'
#     inner_id = trainset.to_inner_iid(raw_movie_id)
#     converted_back = trainset.to_raw_iid(inner_id)
#     assert converted_back == raw_movie_id, "Should convert back to original raw ID"


# def test_all_raw_ids_have_inner_mapping(sample_ratings_data):
#     """Test that all raw IDs in dataset have corresponding inner IDs"""
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # All unique users should be mappable
#     unique_users = sample_ratings_data['user_id'].unique()
#     for user_id in unique_users:
#         try:
#             inner_id = trainset.to_inner_uid(user_id)
#             assert inner_id >= 0
#         except ValueError:
#             pytest.fail(f"User ID {user_id} should have inner mapping")

#     # All unique movies should be mappable
#     unique_movies = sample_ratings_data['movie_id'].unique()
#     for movie_id in unique_movies:
#         try:
#             inner_id = trainset.to_inner_iid(movie_id)
#             assert inner_id >= 0
#         except ValueError:
#             pytest.fail(f"Movie ID {movie_id} should have inner mapping")


# def test_unknown_id_raises_error(sample_ratings_data):
#     """Test that unknown IDs raise ValueError"""
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # Unknown user should raise ValueError
#     with pytest.raises(ValueError):
#         trainset.to_inner_uid('unknown_user')

#     # Unknown movie should raise ValueError
#     with pytest.raises(ValueError):
#         trainset.to_inner_iid('unknown_movie')


# # ============================================================================
# # Test: User-Item Matrix Structure
# # ============================================================================

# def test_trainset_user_item_structure(sample_ratings_data):
#     """Test that trainset creates correct user-item rating structure"""
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # Check user-item ratings structure
#     for user_inner_id in range(trainset.n_users):
#         user_ratings = trainset.ur[user_inner_id]
#         assert isinstance(user_ratings, list), "User ratings should be list"

#         for item_inner_id, rating in user_ratings:
#             assert isinstance(item_inner_id, int), "Item ID should be integer"
#             assert isinstance(rating, (int, float)), "Rating should be numeric"
#             assert 1 <= rating <= 5, "Rating should be in scale (1-5)"


# def test_global_mean_rating_computation(sample_ratings_data):
#     """Test global mean rating computation"""
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(sample_ratings_data[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # Trainset should compute global mean
#     global_mean = trainset.global_mean
#     expected_mean = sample_ratings_data['rating'].mean()

#     assert abs(global_mean - expected_mean) < 0.01, "Global mean should match DataFrame mean"


# # ============================================================================
# # Test: Data Validation
# # ============================================================================

# def test_reject_invalid_rating_scale():
#     """Test that ratings outside scale are handled properly"""
#     df = pd.DataFrame([
#         {'user_id': 'user1', 'movie_id': 'movie1', 'rating': 6},  # Invalid: > 5
#         {'user_id': 'user2', 'movie_id': 'movie2', 'rating': 0},  # Invalid: < 1
#     ])

#     reader = Reader(rating_scale=(1, 5))

#     # Surprise will clip or handle out-of-range ratings
#     # This test ensures the data can still be loaded
#     data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # The dataset should still be created (Surprise handles this internally)
#     assert trainset.n_ratings == 2


# def test_empty_dataframe_handling():
#     """Test handling of empty DataFrame"""
#     df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
#     reader = Reader(rating_scale=(1, 5))

#     data = Dataset.load_from_df(df, reader)
#     trainset = data.build_full_trainset()

#     assert trainset.n_users == 0, "Empty data should have 0 users"
#     assert trainset.n_items == 0, "Empty data should have 0 items"
#     assert trainset.n_ratings == 0, "Empty data should have 0 ratings"


# def test_single_rating_handling():
#     """Test handling of dataset with single rating"""
#     df = pd.DataFrame([
#         {'user_id': 'user1', 'movie_id': 'movie1', 'rating': 4}
#     ])

#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     assert trainset.n_users == 1
#     assert trainset.n_items == 1
#     assert trainset.n_ratings == 1
#     assert trainset.global_mean == 4.0


# # ============================================================================
# # Test: Integration with Training Class
# # ============================================================================

# def test_training_class_load_data_transformation(temp_db_with_ratings):
#     """Test the complete load_data transformation from Training class"""
#     # Mock the Training class load_data method
#     conn = sqlite3.connect(temp_db_with_ratings)
#     query = "SELECT user_id, movie_id, rating FROM ratings"
#     df = pd.read_sql_query(query, conn)
#     conn.close()

#     # Apply transformations (train.py lines 40-44)
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
#     trainset = data.build_full_trainset()

#     # Verify complete transformation pipeline
#     assert trainset is not None
#     assert trainset.n_ratings == len(df)
#     assert trainset.rating_scale == (1, 5)

#     # Verify all original users and items are preserved
#     assert trainset.n_users == df['user_id'].nunique()
#     assert trainset.n_items == df['movie_id'].nunique()
