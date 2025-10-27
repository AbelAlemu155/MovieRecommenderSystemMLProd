import pytest
from unittest.mock import MagicMock, Mock, patch
import sys

# Mock modules before importing DataProcessor
# These need to be mocked because process.py imports them directly
sys.modules['kafka_pipeline'] = MagicMock()
sys.modules['data_queue'] = MagicMock()
sys.modules['schema_validation'] = MagicMock()
sys.modules['schema_validation.validator'] = MagicMock()
sys.modules['schema_validation.schemas'] = MagicMock()

from homealone.data_processing.process import DataProcessor


# ============================================================================
# Initialization Tests
# ============================================================================

def test_data_processor_initializes_with_kafka_client(mock_kafka_client):
    """Test DataProcessor initializes correctly with KafkaSSHClient"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    assert processor.kafka_client == mock_kafka_client
    assert processor.data_queue is not None


@patch('homealone.data_processing.process.dq.DataQueue')
def test_data_processor_creates_data_queue(mock_queue_class, mock_kafka_client):
    """Test that DataQueue is instantiated during initialization"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    mock_queue_class.assert_called_once()


# ============================================================================
# Movie Log Processing Tests
# ============================================================================

def test_process_movie_log_basic(mock_kafka_client, sample_movie_log):
    """Test processing basic movie log"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(sample_movie_log)
    
    assert result is not None
    assert result["type"] == "movie"
    assert result["userid"] == "user123"
    assert result["movieid"] == "movie456"
    assert result["minute"] == "12"


def test_process_movie_log_matches_expected_structure(mock_kafka_client, sample_movie_log, expected_movie_data):
    """Test movie log produces expected data structure"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(sample_movie_log)
    
    assert result == expected_movie_data


@pytest.mark.parametrize("log_line,expected_movieid,expected_minute", [
    ("2023-10-25 10:30:15,user123,GET /data/m/movie456/12.mpg", "movie456", "12"),
    ("2024-01-01 15:20:00,user999,GET /data/m/action123/5.mpg", "action123", "5"),
    ("2024-02-15 09:00:00,userABC,GET /data/m/comedy-789/99.mpg", "comedy-789", "99"),
])
def test_process_movie_log_various_formats(mock_kafka_client, log_line, expected_movieid, expected_minute):
    """Test movie log processing with various formats"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(log_line)
    
    assert result["type"] == "movie"
    assert result["movieid"] == expected_movieid
    assert result["minute"] == expected_minute


def test_process_movie_log_with_special_characters(mock_kafka_client):
    """Test movie log with special characters in movieid"""
    log = "2023-10-25 10:30:15,user123,GET /data/m/movie-special_456/12.mpg"
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(log)
    
    assert result is not None
    assert result["movieid"] == "movie-special_456"


# ============================================================================
# Rating Log Processing Tests
# ============================================================================

def test_process_rating_log_basic(mock_kafka_client, sample_rating_log):
    """Test processing basic rating log"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(sample_rating_log)
    
    assert result is not None
    assert result["type"] == "rating"
    assert result["userid"] == "user789"
    assert result["movieid"] == "movie101"
    assert result["rating"] == "4"


def test_process_rating_log_matches_expected_structure(mock_kafka_client, sample_rating_log, expected_rating_data):
    """Test rating log produces expected data structure"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(sample_rating_log)
    
    assert result == expected_rating_data


@pytest.mark.parametrize("rating_value", ["1", "2", "3", "4", "5"])
def test_process_rating_log_all_rating_values(mock_kafka_client, rating_value):
    """Test rating log processing with all valid rating values"""
    log = f"2023-10-25 10:31:20,user789,GET /rate/movie101={rating_value}"
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(log)
    
    assert result is not None
    assert result["type"] == "rating"
    assert result["rating"] == rating_value


@pytest.mark.parametrize("log_line,expected_movieid,expected_rating", [
    ("2023-10-25 10:31:20,user789,GET /rate/movie101=4", "movie101", "4"),
    ("2024-01-15 14:20:30,userXYZ,GET /rate/action-999=5", "action-999", "5"),
    ("2024-03-10 08:45:00,user555,GET /rate/comedy_123=3", "comedy_123", "3"),
])
def test_process_rating_log_various_formats(mock_kafka_client, log_line, expected_movieid, expected_rating):
    """Test rating log processing with various formats"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(log_line)
    
    assert result["type"] == "rating"
    assert result["movieid"] == expected_movieid
    assert result["rating"] == expected_rating


# ============================================================================
# Recommendation Request Processing Tests
# ============================================================================

def test_process_recommendation_log_basic(mock_kafka_client, sample_recommendation_log):
    """Test processing basic recommendation request log"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(sample_recommendation_log)
    
    assert result is not None
    assert result["type"] == "recommendation_request"
    assert result["userid"] == "user555"
    assert result["server"] == "server1"
    assert result["status"] == "200"
    assert result["recommendations"] == "movie1,movie2,movie3"
    assert result["responsetime"] == 150.0


def test_process_recommendation_log_matches_expected_structure(
    mock_kafka_client, sample_recommendation_log, expected_recommendation_data
):
    """Test recommendation log produces expected data structure"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(sample_recommendation_log)
    
    assert result == expected_recommendation_data


@pytest.mark.parametrize("status_code", ["200", "404", "500", "503"])
def test_process_recommendation_log_various_status_codes(mock_kafka_client, status_code):
    """Test recommendation log with different HTTP status codes"""
    log = f"2023-10-25 10:32:00,user555,recommendation request server1, status {status_code}, result: movies, 100ms"
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(log)
    
    assert result is not None
    assert result["type"] == "recommendation_request"
    assert result["status"] == status_code


@pytest.mark.parametrize("log_line,expected_server,expected_responsetime", [
    (
        "2023-10-25 10:32:00,user555,recommendation request server1, status 200, result: m1,m2, 50 ms",
        "server1", 50.0
    ),
    (
        "2024-01-10 14:00:00,user123,recommendation request server2, status 200, result: none, 250 ms",
        "server2", 250.0
    ),
    (
        "2024-02-20 18:30:00,userXYZ,recommendation request prod-server, status 200, result: a,b,c,d, 1500 ms",
        "prod-server", 1500.0
    ),
])
def test_process_recommendation_log_various_formats(
    mock_kafka_client, log_line, expected_server, expected_responsetime
):
    """Test recommendation log processing with various formats"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(log_line)
    
    assert result["type"] == "recommendation_request"
    assert result["server"] == expected_server
    assert result["responsetime"] == expected_responsetime


# ============================================================================
# Invalid Log Processing Tests
# ============================================================================

def test_process_invalid_log_returns_none(mock_kafka_client, invalid_log):
    """Test that invalid log returns None"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(invalid_log)
    
    assert result is None


@pytest.mark.parametrize("invalid_input", [
    "invalid,log,data,format",
    "completely random text",
    "",
    "2023-10-25 10:30:15,user123,INVALID",
    "GET /data/m/movie456/12.mpg",  # Missing time and userid
    "2023-10-25 10:30:15,user123,POST /data/m/movie456/12.mpg",  # Wrong HTTP method
    "user123,GET /rate/movie101=4",  # Missing timestamp
])
def test_process_various_invalid_logs(mock_kafka_client, invalid_input):
    """Test that various invalid log formats return None"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(invalid_input)
    
    assert result is None


# ============================================================================
# Print Statement Tests
# ============================================================================

@patch('builtins.print')
def test_process_data_prints_status(mock_print, mock_kafka_client, sample_movie_log):
    """Test that processing prints status message"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    processor.process_data(sample_movie_log)
    
    mock_print.assert_called_once_with("Done processing data: ", sample_movie_log)


@patch('builtins.print')
def test_process_data_prints_for_invalid_log(mock_print, mock_kafka_client, invalid_log):
    """Test that processing prints even for invalid logs"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    processor.process_data(invalid_log)
    
    mock_print.assert_called_once_with("Done processing data: ", invalid_log)


# ============================================================================
# Run Method Tests
# ============================================================================

@patch('homealone.data_processing.process.ValidationManager')
@patch('homealone.data_processing.process.dq.DataQueue')
def test_run_processes_single_message(mock_queue_class, mock_validator_class, mock_kafka_client):
    """Test run method processes a single message"""
    mock_queue_instance = MagicMock()
    mock_queue_class.return_value = mock_queue_instance

    # Mock validator to return valid data
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_record.return_value = (True, {}, None)
    mock_validator_class.return_value = mock_validator_instance

    mock_kafka_client.consume.return_value = iter([
        "2023-10-25 10:30:15,user123,GET /data/m/movie456/12.mpg"
    ])

    processor = DataProcessor(kafka_client=mock_kafka_client)
    processor.run()
    
    # Verify kafka client was called
    mock_kafka_client.consume.assert_called_once()
    
    # Verify data was enqueued
    mock_queue_instance.enqueue_data.assert_called_once()
    
    # Verify enqueued data structure
    call_args = mock_queue_instance.enqueue_data.call_args[0][0]
    assert call_args["type"] == "movie"
    assert call_args["userid"] == "user123"


@patch('homealone.data_processing.process.ValidationManager')
@patch('homealone.data_processing.process.dq.DataQueue')
def test_run_processes_multiple_messages(mock_queue_class, mock_validator_class, mock_kafka_client, sample_log_batch):
    """Test run method processes multiple messages"""
    mock_queue_instance = MagicMock()
    mock_queue_class.return_value = mock_queue_instance

    # Mock validator to return valid data
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_record.return_value = (True, {}, None)
    mock_validator_class.return_value = mock_validator_instance

    mock_kafka_client.consume.return_value = iter(sample_log_batch)

    processor = DataProcessor(kafka_client=mock_kafka_client)
    processor.run()
    
    # Should process 4 valid messages (skip 1 invalid)
    assert mock_queue_instance.enqueue_data.call_count == 4


@patch('homealone.data_processing.process.ValidationManager')
@patch('homealone.data_processing.process.dq.DataQueue')
def test_run_skips_invalid_messages(mock_queue_class, mock_validator_class, mock_kafka_client):
    """Test run method skips invalid messages"""
    mock_queue_instance = MagicMock()
    mock_queue_class.return_value = mock_queue_instance

    # Mock validator to return valid data
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_record.return_value = (True, {}, None)
    mock_validator_class.return_value = mock_validator_instance

    messages = [
        "2023-10-25 10:30:15,user123,GET /data/m/movie456/12.mpg",
        "invalid,log,data",
        "2023-10-25 10:31:20,user789,GET /rate/movie101=4"
    ]
    mock_kafka_client.consume.return_value = iter(messages)

    processor = DataProcessor(kafka_client=mock_kafka_client)
    processor.run()
    
    # Should only enqueue 2 valid messages
    assert mock_queue_instance.enqueue_data.call_count == 2


@patch('homealone.data_processing.process.ValidationManager')
@patch('homealone.data_processing.process.dq.DataQueue')
def test_run_handles_empty_consumer(mock_queue_class, mock_validator_class, mock_kafka_client):
    """Test run method handles empty message stream"""
    mock_queue_instance = MagicMock()
    mock_queue_class.return_value = mock_queue_instance

    # Mock validator to return valid data
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_record.return_value = (True, {}, None)
    mock_validator_class.return_value = mock_validator_instance

    mock_kafka_client.consume.return_value = iter([])

    processor = DataProcessor(kafka_client=mock_kafka_client)
    processor.run()

    # No messages should be enqueued
    mock_queue_instance.enqueue_data.assert_not_called()


@patch('homealone.data_processing.process.ValidationManager')
@patch('homealone.data_processing.process.dq.DataQueue')
def test_run_enqueues_all_message_types(mock_queue_class, mock_validator_class, mock_kafka_client):
    """Test run method enqueues all three message types correctly"""
    mock_queue_instance = MagicMock()
    mock_queue_class.return_value = mock_queue_instance

    # Mock validator to return valid data
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_record.return_value = (True, {}, None)
    mock_validator_class.return_value = mock_validator_instance

    messages = [
        "2023-10-25 10:30:15,user1,GET /data/m/movie1/1.mpg",
        "2023-10-25 10:30:16,user2,GET /rate/movie2=5",
        "2023-10-25 10:30:17,user3,recommendation request server1, status 200, result: m1, 50ms",
    ]
    mock_kafka_client.consume.return_value = iter(messages)

    processor = DataProcessor(kafka_client=mock_kafka_client)
    processor.run()
    
    # Verify all 3 messages were enqueued
    assert mock_queue_instance.enqueue_data.call_count == 3
    
    # Verify different types were processed
    calls = mock_queue_instance.enqueue_data.call_args_list
    types_processed = [call[0][0]["type"] for call in calls]
    
    assert "movie" in types_processed
    assert "rating" in types_processed
    assert "recommendation_request" in types_processed


# ============================================================================
# Integration Tests
# ============================================================================

@patch('homealone.data_processing.process.ValidationManager')
@patch('homealone.data_processing.process.dq.DataQueue')
def test_full_pipeline_integration(mock_queue_class, mock_validator_class, mock_kafka_client, sample_log_batch):
    """Test complete pipeline with mixed messages"""
    mock_queue_instance = MagicMock()
    mock_queue_class.return_value = mock_queue_instance

    # Mock validator to return valid data
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_record.return_value = (True, {}, None)
    mock_validator_class.return_value = mock_validator_instance

    mock_kafka_client.consume.return_value = iter(sample_log_batch)

    processor = DataProcessor(kafka_client=mock_kafka_client)
    processor.run()
    
    # Should process 4 valid messages (skip 1 invalid)
    assert mock_queue_instance.enqueue_data.call_count == 4
    
    # Verify queue was called for each valid message
    calls = mock_queue_instance.enqueue_data.call_args_list
    
    # First message should be movie type
    assert calls[0][0][0]["type"] == "movie"
    assert calls[0][0][0]["movieid"] == "movie1"
    
    # Second message should be rating type
    assert calls[1][0][0]["type"] == "rating"
    assert calls[1][0][0]["rating"] == "5"
    
    # Third message should be recommendation type
    assert calls[2][0][0]["type"] == "recommendation_request"
    
    # Fourth message should be movie type
    assert calls[3][0][0]["type"] == "movie"
    assert calls[3][0][0]["movieid"] == "movie3"


@patch('homealone.data_processing.process.ValidationManager')
@patch('homealone.data_processing.process.dq.DataQueue')
@patch('builtins.print')
def test_run_prints_for_all_messages(mock_print, mock_queue_class, mock_validator_class, mock_kafka_client):
    """Test that run prints status for all messages"""
    mock_queue_instance = MagicMock()
    mock_queue_class.return_value = mock_queue_instance

    # Mock validator to return valid data
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate_record.return_value = (True, {}, None)
    mock_validator_class.return_value = mock_validator_instance

    messages = [
        "2023-10-25 10:30:15,user1,GET /data/m/movie1/1.mpg",
        "invalid,log,data",
    ]
    mock_kafka_client.consume.return_value = iter(messages)

    processor = DataProcessor(kafka_client=mock_kafka_client)
    processor.run()

    # Should print for both messages (valid and invalid)
    assert mock_print.call_count == 2


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_process_data_with_empty_string(mock_kafka_client):
    """Test processing empty string"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data("")
    
    assert result is None


def test_process_data_with_whitespace_only(mock_kafka_client):
    """Test processing whitespace-only string"""
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data("   ")
    
    assert result is None


def test_process_movie_log_with_large_minute_value(mock_kafka_client):
    """Test movie log with large minute value"""
    log = "2023-10-25 10:30:15,user123,GET /data/m/movie456/999.mpg"
    processor = DataProcessor(kafka_client=mock_kafka_client)
    
    result = processor.process_data(log)
    
    assert result is not None
    assert result["minute"] == "999"
