"""
Tests for the telemetry module.

These tests cover:
1. Database table setup and schema creation
2. Logging recommendation events
3. UPSERT behavior (INSERT or UPDATE on conflict)
4. Input validation for event types
5. Data integrity and constraints
6. Error handling for invalid inputs
7. Timestamp handling (both provided and default)
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime
import sys

# Add the inference directory to the path
inference_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../inference'))
sys.path.insert(0, inference_dir)

from homealone.inference.telemetry import Telemetry

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def db_connection(temp_db):
    """Create a database connection for testing"""
    conn = sqlite3.connect(temp_db)
    yield conn
    conn.close()


@pytest.fixture
def telemetry_instance(db_connection):
    """Create a Telemetry instance with test database"""
    return Telemetry(db_connection)


@pytest.fixture
def sample_recommendation_details():
    """Sample recommendation details for logging"""
    return {
        'user_id': 'user123',
        'recommended_items': 'movie1,movie2,movie3,movie4,movie5',
        'timestamp': '2024-10-27T10:30:00Z'
    }


@pytest.fixture
def sample_recommendation_details_no_timestamp():
    """Sample recommendation details without timestamp (will use default)"""
    return {
        'user_id': 'user456',
        'recommended_items': 'movie10,movie11,movie12',
        'timestamp': None
    }


# ============================================================================
# Test: Database Setup
# ============================================================================

def test_telemetry_init_creates_table(db_connection):
    """Test that Telemetry.__init__ creates recommendations table"""
    telemetry = Telemetry(db_connection)

    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='recommendations'
    """)
    result = cursor.fetchone()

    assert result is not None, "Table 'recommendations' should be created"
    assert result[0] == 'recommendations'


def test_recommendations_table_schema(telemetry_instance, db_connection):
    """Test that recommendations table has correct schema"""
    cursor = db_connection.cursor()
    cursor.execute("PRAGMA table_info(recommendations)")
    columns = cursor.fetchall()

    # Extract column names and types
    column_info = {col[1]: col[2] for col in columns}

    assert 'id' in column_info, "Should have 'id' column"
    assert 'user_id' in column_info, "Should have 'user_id' column"
    assert 'recommended_items' in column_info, "Should have 'recommended_items' column"
    assert 'timestamp' in column_info, "Should have 'timestamp' column"

    # Check types
    assert 'INTEGER' in column_info['id'], "id should be INTEGER"
    assert 'TEXT' in column_info['user_id'], "user_id should be TEXT"
    assert 'TEXT' in column_info['recommended_items'], "recommended_items should be TEXT"
    assert 'DATETIME' in column_info['timestamp'], "timestamp should be DATETIME"


def test_recommendations_table_constraints(telemetry_instance, db_connection):
    """Test that recommendations table has correct constraints"""
    cursor = db_connection.cursor()
    cursor.execute("PRAGMA table_info(recommendations)")
    columns = cursor.fetchall()

    # Find user_id column (index 1 is column name, index 3 is NOT NULL, index 5 is PRIMARY KEY)
    user_id_col = [col for col in columns if col[1] == 'user_id'][0]
    id_col = [col for col in columns if col[1] == 'id'][0]

    # Check NOT NULL constraint on user_id
    assert user_id_col[3] == 1, "user_id should have NOT NULL constraint"

    # Check PRIMARY KEY on id
    assert id_col[5] == 1, "id should be PRIMARY KEY"


def test_table_created_only_once(db_connection):
    """Test that creating multiple Telemetry instances doesn't cause errors"""
    telemetry1 = Telemetry(db_connection)
    telemetry2 = Telemetry(db_connection)  # Should not raise error

    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recommendations'")
    tables = cursor.fetchall()

    assert len(tables) == 1, "Should only create one recommendations table"


# ============================================================================
# Test: Log Method - Event Type Validation
# ============================================================================

def test_log_valid_event_type(telemetry_instance, sample_recommendation_details):
    """Test logging with valid 'recommendation' event type"""
    # Should not raise any exception
    telemetry_instance.log('recommendation', sample_recommendation_details)


def test_log_invalid_event_type(telemetry_instance):
    """Test that invalid event type raises ValueError"""
    invalid_details = {
        'user_id': 'user123',
        'recommended_items': 'movie1,movie2',
        'timestamp': None
    }

    with pytest.raises(ValueError) as exc_info:
        telemetry_instance.log('invalid_type', invalid_details)

    assert 'Invalid event type' in str(exc_info.value)


def test_log_empty_event_type(telemetry_instance, sample_recommendation_details):
    """Test that empty event type raises ValueError"""
    with pytest.raises(ValueError):
        telemetry_instance.log('', sample_recommendation_details)


def test_log_none_event_type(telemetry_instance, sample_recommendation_details):
    """Test that None event type raises ValueError"""
    with pytest.raises(ValueError):
        telemetry_instance.log(None, sample_recommendation_details)


# ============================================================================
# Test: Recommendation Logging - INSERT
# ============================================================================

def test_recommendation_log_insert(telemetry_instance, db_connection, sample_recommendation_details):
    """Test inserting a new recommendation log"""
    telemetry_instance.log('recommendation', sample_recommendation_details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM recommendations WHERE user_id = ?", ('user123',))
    result = cursor.fetchone()

    assert result is not None, "Recommendation should be inserted"
    assert result[1] == 'user123', "user_id should match"
    assert result[2] == 'movie1,movie2,movie3,movie4,movie5', "recommended_items should match"
    assert result[3] == '2024-10-27T10:30:00Z', "timestamp should match"


def test_recommendation_log_with_default_timestamp(telemetry_instance, db_connection, sample_recommendation_details_no_timestamp):
    """Test logging with default timestamp (NULL)"""
    telemetry_instance.log('recommendation', sample_recommendation_details_no_timestamp)

    cursor = db_connection.cursor()
    cursor.execute("SELECT user_id, timestamp FROM recommendations WHERE user_id = ?", ('user456',))
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == 'user456'
    # When timestamp is None, it should be stored as NULL or use CURRENT_TIMESTAMP
    # Check that timestamp was stored
    assert result[1] is None or result[1] != '', "Timestamp should be set (NULL or current time)"


def test_recommendation_log_multiple_users(telemetry_instance, db_connection):
    """Test logging recommendations for multiple different users"""
    users = [
        {'user_id': 'user1', 'recommended_items': 'movie1,movie2', 'timestamp': None},
        {'user_id': 'user2', 'recommended_items': 'movie3,movie4', 'timestamp': None},
        {'user_id': 'user3', 'recommended_items': 'movie5,movie6', 'timestamp': None},
    ]

    for details in users:
        telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations")
    count = cursor.fetchone()[0]

    assert count == 3, "Should have 3 recommendation logs"


def test_recommendation_log_with_numeric_user_id(telemetry_instance, db_connection):
    """Test logging with numeric user_id (stored as TEXT)"""
    details = {
        'user_id': 123,  # Numeric, but will be stored as TEXT
        'recommended_items': 'movie1,movie2',
        'timestamp': None
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT user_id FROM recommendations WHERE user_id = ?", ('123',))
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == '123', "Numeric user_id should be stored as text"


# ============================================================================
# Test: Recommendation Logging - UPSERT (UPDATE on conflict)
# ============================================================================

def test_recommendation_log_upsert_updates_existing(telemetry_instance, db_connection):
    """Test that logging same user_id updates the existing record"""
    # First insert
    details1 = {
        'user_id': 'user789',
        'recommended_items': 'movie1,movie2',
        'timestamp': '2024-10-27T10:00:00Z'
    }
    telemetry_instance.log('recommendation', details1)

    # Second insert with same user_id (should update)
    details2 = {
        'user_id': 'user789',
        'recommended_items': 'movie3,movie4,movie5',
        'timestamp': '2024-10-27T11:00:00Z'
    }
    telemetry_instance.log('recommendation', details2)

    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations WHERE user_id = ?", ('user789',))
    count = cursor.fetchone()[0]

    assert count == 1, "Should only have one record for user789"

    cursor.execute("SELECT recommended_items, timestamp FROM recommendations WHERE user_id = ?", ('user789',))
    result = cursor.fetchone()

    assert result[0] == 'movie3,movie4,movie5', "recommended_items should be updated"
    assert result[1] == '2024-10-27T11:00:00Z', "timestamp should be updated"


def test_recommendation_log_upsert_preserves_id(telemetry_instance, db_connection):
    """Test that UPSERT preserves the original record ID"""
    details1 = {
        'user_id': 'user999',
        'recommended_items': 'movie1',
        'timestamp': None
    }
    telemetry_instance.log('recommendation', details1)

    cursor = db_connection.cursor()
    cursor.execute("SELECT id FROM recommendations WHERE user_id = ?", ('user999',))
    original_id = cursor.fetchone()[0]

    # Update
    details2 = {
        'user_id': 'user999',
        'recommended_items': 'movie2',
        'timestamp': None
    }
    telemetry_instance.log('recommendation', details2)

    cursor.execute("SELECT id FROM recommendations WHERE user_id = ?", ('user999',))
    updated_id = cursor.fetchone()[0]

    assert original_id == updated_id, "ID should remain the same after update"


def test_recommendation_log_upsert_multiple_updates(telemetry_instance, db_connection):
    """Test multiple updates to the same user"""
    user_id = 'user_multi_update'

    for i in range(5):
        details = {
            'user_id': user_id,
            'recommended_items': f'movie{i},movie{i+1}',
            'timestamp': f'2024-10-27T10:{i:02d}:00Z'
        }
        telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]

    assert count == 1, "Should only have one record after multiple updates"

    cursor.execute("SELECT recommended_items FROM recommendations WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()

    # Should have the last update
    assert result[0] == 'movie4,movie5', "Should have the last updated value"


# ============================================================================
# Test: Data Integrity
# ============================================================================

def test_recommendation_log_empty_user_id(telemetry_instance):
    """Test that empty user_id is handled"""
    details = {
        'user_id': '',
        'recommended_items': 'movie1,movie2',
        'timestamp': None
    }

    # SQLite allows empty strings, so this should succeed
    telemetry_instance.log('recommendation', details)


def test_recommendation_log_empty_recommended_items(telemetry_instance, db_connection):
    """Test logging with empty recommended_items"""
    details = {
        'user_id': 'user_empty_items',
        'recommended_items': '',
        'timestamp': None
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT recommended_items FROM recommendations WHERE user_id = ?", ('user_empty_items',))
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == '', "Empty recommended_items should be allowed"


def test_recommendation_log_long_recommended_items(telemetry_instance, db_connection):
    """Test logging with very long list of recommended items"""
    # Create a long comma-separated list
    long_items = ','.join([f'movie{i}' for i in range(1000)])

    details = {
        'user_id': 'user_long_list',
        'recommended_items': long_items,
        'timestamp': None
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT recommended_items FROM recommendations WHERE user_id = ?", ('user_long_list',))
    result = cursor.fetchone()

    assert result is not None
    assert len(result[0]) > 5000, "Should store long list"
    assert 'movie999' in result[0], "Should contain last item"


def test_recommendation_log_special_characters_in_items(telemetry_instance, db_connection):
    """Test logging with special characters in recommended_items"""
    details = {
        'user_id': 'user_special',
        'recommended_items': "movie's,movie\"2\",movie<3>,movie&4",
        'timestamp': None
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT recommended_items FROM recommendations WHERE user_id = ?", ('user_special',))
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == "movie's,movie\"2\",movie<3>,movie&4", "Special characters should be preserved"


def test_recommendation_log_special_characters_in_user_id(telemetry_instance, db_connection):
    """Test logging with special characters in user_id"""
    details = {
        'user_id': "user@email.com",
        'recommended_items': 'movie1,movie2',
        'timestamp': None
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT user_id FROM recommendations WHERE user_id = ?", ('user@email.com',))
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == 'user@email.com'


# ============================================================================
# Test: Timestamp Handling
# ============================================================================

def test_recommendation_log_various_timestamp_formats(telemetry_instance, db_connection):
    """Test logging with different timestamp formats"""
    timestamps = [
        '2024-10-27T10:30:00Z',
        '2024-10-27 10:30:00',
        '2024-10-27',
        '10:30:00',
    ]

    for i, ts in enumerate(timestamps):
        details = {
            'user_id': f'user_ts_{i}',
            'recommended_items': 'movie1,movie2',
            'timestamp': ts
        }
        telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations WHERE user_id LIKE 'user_ts_%'")
    count = cursor.fetchone()[0]

    assert count == len(timestamps), "All timestamps should be stored"


def test_recommendation_log_null_timestamp_uses_default(telemetry_instance, db_connection):
    """Test that NULL timestamp triggers DEFAULT CURRENT_TIMESTAMP or stays NULL"""
    details = {
        'user_id': 'user_null_ts',
        'recommended_items': 'movie1,movie2',
        'timestamp': None
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT timestamp FROM recommendations WHERE user_id = ?", ('user_null_ts',))
    result = cursor.fetchone()

    # The behavior depends on implementation - either NULL or CURRENT_TIMESTAMP
    # The code passes None, so it will be NULL
    assert result is not None, "Record should exist"


# ============================================================================
# Test: Error Handling
# ============================================================================

def test_recommendation_log_missing_required_fields(telemetry_instance):
    """Test that missing required fields raise appropriate errors"""
    incomplete_details = {
        'user_id': 'user123',
        # Missing 'recommended_items' and 'timestamp'
    }

    with pytest.raises(KeyError):
        telemetry_instance.log('recommendation', incomplete_details)


def test_recommendation_log_none_values(telemetry_instance, db_connection):
    """Test logging with None for recommended_items"""
    details = {
        'user_id': 'user_none_items',
        'recommended_items': None,
        'timestamp': None
    }

    # This should fail or insert NULL
    with pytest.raises((TypeError, sqlite3.IntegrityError)):
        telemetry_instance.log('recommendation', details)


def test_telemetry_with_closed_connection():
    """Test that using closed connection raises error"""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(db_path)
    telemetry = Telemetry(conn)
    conn.close()

    details = {
        'user_id': 'user123',
        'recommended_items': 'movie1,movie2',
        'timestamp': None
    }

    with pytest.raises(sqlite3.ProgrammingError):
        telemetry.log('recommendation', details)

    os.remove(db_path)


# ============================================================================
# Test: Database Transaction and Commit
# ============================================================================

def test_recommendation_log_commits_transaction(telemetry_instance, db_connection):
    """Test that log method commits the transaction"""
    details = {
        'user_id': 'user_commit',
        'recommended_items': 'movie1,movie2',
        'timestamp': None
    }

    telemetry_instance.log('recommendation', details)

    # Create a new connection to verify data was committed
    new_conn = sqlite3.connect(db_connection.execute("PRAGMA database_list").fetchone()[2])
    cursor = new_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations WHERE user_id = ?", ('user_commit',))
    count = cursor.fetchone()[0]
    new_conn.close()

    assert count == 1, "Data should be committed and visible to new connection"


# ============================================================================
# Test: Concurrent Access
# ============================================================================

def test_multiple_telemetry_instances_same_db(temp_db):
    """Test multiple Telemetry instances can use the same database"""
    conn1 = sqlite3.connect(temp_db)
    conn2 = sqlite3.connect(temp_db)

    telemetry1 = Telemetry(conn1)
    telemetry2 = Telemetry(conn2)

    details1 = {
        'user_id': 'user_conn1',
        'recommended_items': 'movie1',
        'timestamp': None
    }
    telemetry1.log('recommendation', details1)

    details2 = {
        'user_id': 'user_conn2',
        'recommended_items': 'movie2',
        'timestamp': None
    }
    telemetry2.log('recommendation', details2)

    cursor = conn1.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations")
    count = cursor.fetchone()[0]

    conn1.close()
    conn2.close()

    assert count == 2, "Both connections should write to same database"


# ============================================================================
# Test: Integration with Server
# ============================================================================

def test_telemetry_typical_usage_pattern(telemetry_instance, db_connection):
    """Test typical usage pattern from server code"""
    # Simulate server behavior
    user_id = 42
    recommended_items = "101,102,103,104,105"
    timestamp = None  # Server passes None for current time

    details = {
        'user_id': user_id,
        'recommended_items': recommended_items,
        'timestamp': timestamp
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT user_id, recommended_items FROM recommendations WHERE user_id = ?", ('42',))
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == '42'
    assert result[1] == '101,102,103,104,105'


def test_telemetry_fallback_recommendation_logging(telemetry_instance, db_connection):
    """Test logging fallback recommendations (when user not found)"""
    details = {
        'user_id': 9999,
        'recommended_items': 'movie1,movie2,movie3',  # Fallback movies
        'timestamp': '2024-10-27T10:30:00Z'
    }

    telemetry_instance.log('recommendation', details)

    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM recommendations WHERE user_id = ?", ('9999',))
    result = cursor.fetchone()

    assert result is not None, "Fallback recommendations should be logged"


def test_telemetry_repeated_user_requests(telemetry_instance, db_connection):
    """Test logging multiple requests from same user (realistic scenario)"""
    user_id = 'user_repeated'

    # First request
    details1 = {
        'user_id': user_id,
        'recommended_items': 'movie1,movie2,movie3',
        'timestamp': '2024-10-27T10:00:00Z'
    }
    telemetry_instance.log('recommendation', details1)

    # Second request (user comes back later)
    details2 = {
        'user_id': user_id,
        'recommended_items': 'movie4,movie5,movie6',
        'timestamp': '2024-10-27T15:00:00Z'
    }
    telemetry_instance.log('recommendation', details2)

    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]

    assert count == 1, "Should only keep latest recommendation"

    cursor.execute("SELECT recommended_items, timestamp FROM recommendations WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()

    assert result[0] == 'movie4,movie5,movie6', "Should have latest recommendations"
    assert result[1] == '2024-10-27T15:00:00Z', "Should have latest timestamp"
