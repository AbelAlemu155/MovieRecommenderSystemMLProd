# Test Suite Documentation

## Schema Validation Module Tests

### 1. `test_schemas.py` — [homealone/tests/test_schema validation/test_schemas.py](homealone/tests/test_schema validation/test_schemas.py)
Tests all Pydantic schemas with complete coverage.

**Total: 38 tests**

#### **MovieEvent Tests** (9 tests)
- Valid event creation
- Positive userid validation (rejects zero and negative)
- Non-empty movieid validation
- Non-negative minute validation (accepts zero)
- Missing required fields rejection
- Type coercion (string to int)

#### **RatingEvent Tests** (6 tests)
- Valid event creation
- Rating range validation (0-5) with parametrized tests
- Invalid rating range rejection (negative, > 5)
- Userid must be positive
- Movieid cannot be empty

#### **RecommendationEvent Tests** (4 tests)
- Valid event creation
- Status code validation (200, 404, 500 only)
- Invalid status code rejection (201, 400, 403, 503)
- Userid must be positive

#### **WatchHistoryRecord Tests** (6 tests)
- Valid record creation
- Default total_watch_time (0)
- User_id must be positive
- Movie_id cannot be empty
- Watch time must be non-negative
- Zero watch time is valid

#### **RatingRecord Tests** (5 tests)
- Valid record creation
- Optional rating field (can be None)
- Rating range validation (0-5) with parametrized tests
- Invalid rating rejection

#### **MovieMetadata Tests** (4 tests)
- Minimal required fields (movie_id only)
- Full metadata with all optional fields
- Partial optional fields
- All optional fields as None

#### **Edge Cases** (4 tests)
- Type coercion (string to int/float)
- Invalid type handling (non-coercible strings)

---

### 2. `test_validator.py` — [homealone/tests/test_schema validation/test_validator.py](homealone/tests/test_schema validation/test_validator.py)
Tests the ValidationManager class functionality.

**Total: 32 tests**

#### **Initialization Tests** (1 test)
- Correct initialization with default values

#### **Single Record Validation Tests** (7 tests)
- Valid MovieEvent acceptance
- Invalid MovieEvent rejection
- Failed record storage
- Valid RatingEvent acceptance
- Invalid RatingEvent rejection
- Missing required field handling
- Return values (is_valid, validated_object, error_message)

#### **Batch Validation Tests** (6 tests)
- All valid records processing
- All invalid records handling
- Mixed valid/invalid records separation
- Empty list handling
- RatingEvent batch validation
- WatchHistoryRecord batch validation

#### **Statistics Tests** (5 tests)
- Initial state (all zeros)
- Statistics after validation
- Valid percentage calculation (50%, 75%)
- 100% valid records
- 0% valid records

#### **Failed Records Management Tests** (7 tests)
- Empty list initially
- Failed records retrieval after failures
- Schema name storage
- Original record preservation
- Error message inclusion
- Clearing failed records
- Multiple clears (idempotent)

#### **Cumulative Statistics Tests** (2 tests)
- Stats accumulation across multiple batches
- Failed records accumulation

#### **Logging Tests** (2 tests)
- Warning logged on validation failure (mocked)
- No logging on successful validation

#### **Integration Tests** (2 tests)
- Full validation workflow (multiple schemas)
- Multiple schema types simultaneously

---

## Data Processing Module Tests

### 3. `test_kafka_pipeline.py` — [homealone/tests/test_data_processing/test_kafka_pipeline.py](homealone/tests/test_data_processing/test_kafka_pipeline.py)
Tests the KafkaSSHClient class for Kafka connectivity and SSH tunnel management.

**Total: 28 tests**

#### **Initialization Tests** (2 tests)
- Client initializes with all parameters (username, host, password, ports)
- Client initializes without password (SSH key auth)

#### **Tunnel Detection Tests** (5 tests)
- Detects existing SSH tunnel
- Returns False when no tunnel exists
- Handles psutil.AccessDenied gracefully
- Ignores SSH tunnels on different ports
- Correctly matches username@host in command line

#### **Tunnel Start Tests** (2 tests)
- Uses sshpass when password provided
- Uses plain SSH when no password (key-based auth)

#### **Tunnel Orchestration Tests** (2 tests)
- Starts tunnel when not running
- Skips starting when already running

#### **Consume (Streaming) Tests** (5 tests)
- Yields UTF-8 decoded messages
- Uses correct default parameters (topic, group_id, auto_offset_reset)
- Commits and closes consumer on exception
- Handles UTF-8 special characters
- Ensures tunnel before consuming

#### **Consume Batch Tests** (5 tests)
- Returns exactly batch_size messages
- Handles fewer messages than requested
- Sets consumer_timeout_ms correctly
- Handles exceptions gracefully (returns empty list)
- Commits after batch consumption

#### **Close Tests** (2 tests)
- Terminates SSH process when present
- Handles case when no SSH process exists

#### **Integration Tests** (2 tests)
- Full workflow: start tunnel, consume, close
- Sequential batch consumption

---

### 4. `test_process.py` — [homealone/tests/test_data_processing/test_process.py](homealone/tests/test_data_processing/test_process.py)
Tests the DataProcessor class for log parsing and data queue management.

**Total: 30 tests**

#### **Initialization Tests** (2 tests)
- Initializes correctly with KafkaSSHClient
- Creates DataQueue instance

#### **Movie Log Processing Tests** (5 tests)
- Processes basic movie log
- Matches expected data structure
- Handles various log formats (parametrized)
- Extracts movieid and minute correctly
- Handles special characters in movieid

#### **Rating Log Processing Tests** (5 tests)
- Processes basic rating log
- Matches expected data structure
- Handles all rating values (1-5) with parametrized tests
- Extracts movieid and rating correctly
- Handles various log formats

#### **Recommendation Request Processing Tests** (4 tests)
- Processes basic recommendation log
- Matches expected data structure
- Handles various HTTP status codes (parametrized)
- Extracts server, status, recommendations, response time

#### **Invalid Log Processing Tests** (2 tests)
- Returns None for invalid logs
- Handles various invalid formats (parametrized: empty, malformed, wrong method)

#### **Print Statement Tests** (2 tests)
- Prints status message for valid logs
- Prints status even for invalid logs

#### **Run Method Tests** (7 tests)
- Processes single message
- Processes multiple messages
- Skips invalid messages (doesn't enqueue)
- Handles empty consumer
- Enqueues all message types (movie, rating, recommendation)
- Verifies kafka client consume is called
- Verifies DataQueue enqueue_data is called

#### **Integration Tests** (2 tests)
- Full pipeline with mixed messages
- Prints for all messages (valid and invalid)

#### **Edge Case Tests** (3 tests)
- Handles empty string
- Handles whitespace-only string
- Handles large minute values (999)

---

### 5. `test_db_writer.py` — [homealone/tests/test_data_processing/test_db_writer.py](homealone/tests/test_data_processing/test_db_writer.py)
Tests the DBWriter class for database operations and file processing.

**Total: 37 tests**

#### **Initialization Tests** (3 tests)
- Initializes with correct paths (/app/db/home_alone.db, /app/data_queues/)
- Creates necessary directories
- Creates tables during initialization

#### **Table Creation Tests** (3 tests)
- Creates watch_history table with correct schema
- Creates ratings table with correct schema
- Creates movies table

#### **Validation Tests** (2 tests)
- Separates valid and invalid records using ValidationManager
- Handles empty data

#### **Watch History Update Tests** (5 tests)
- Inserts new watch history record
- Handles multiple rows
- Uses INSERT...ON CONFLICT (upsert)
- Handles empty DataFrame gracefully
- Commits changes to database

#### **Ratings Update Tests** (4 tests)
- Inserts new rating record
- Handles multiple rows
- Uses INSERT...ON CONFLICT (upsert)
- Handles all rating values (1-5) with parametrized tests

#### **Process Files Tests - No Files** (1 test)
- Handles empty directory (no CSV files)

#### **Process Files Tests - Movie Files** (2 tests)
- Processes movie CSV file
- Aggregates movie watch data correctly
- Validates and removes processed file

#### **Process Files Tests - Rating Files** (2 tests)
- Processes rating CSV file
- Aggregates rating data correctly
- Validates and removes processed file

#### **Process Files Tests - Multiple Files** (1 test)
- Handles multiple CSV files (movies and ratings)
- Returns different schema records for each type

#### **Process Files Tests - Error Handling** (3 tests)
- Handles empty CSV files (EmptyDataError)
- Continues processing after error
- Handles validation failures gracefully

#### **Close Tests** (1 test)
- Closes database connection

#### **Validation Stats Tests** (1 test)
- Returns ValidationManager statistics

#### **Integration Tests** (1 test)
- Full pipeline: process files, get stats, close

#### **Edge Case Tests** (2 tests)
- Ignores non-CSV files (glob pattern)
- Skips unknown file types (not movies or ratings)

---

## Test Summary

### Total Test Count by Module:
- **Schema Validation**: 70 tests (38 schemas + 32 validator)
- **Data Processing**: 95 tests (28 kafka + 30 process + 37 db_writer)

### **Grand Total: 165 tests**

### Key Testing Patterns Used:
1. **Parametrized tests** - Testing multiple values efficiently
2. **Fixtures** - Reusable test data (conftest.py)
3. **Mocking** - For external dependencies (Kafka, SSH, database, logger)
4. **Comprehensive edge cases** - Empty values, special characters, boundary conditions
5. **Integration tests** - End-to-end workflows
6. **Error handling** - Exception handling and graceful degradation
7. **Statistical validation** - Verifying calculations and counters