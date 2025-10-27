import pytest
from unittest.mock import MagicMock, Mock

# Kafka Pipeline Fixtures
@pytest.fixture
def kafka_client_params():
    """Standard KafkaSSHClient initialization parameters"""
    return {
        "username": "testuser",
        "host": "testhost.com",
        "password": "testpass",
        "local_port": "9092",
        "remote_port": "9092"
    }


@pytest.fixture
def mock_kafka_client():
    """Mock KafkaSSHClient for testing"""
    client = MagicMock()
    client.consume = MagicMock()
    client.consume_batch = MagicMock()
    client.ensure_tunnel = MagicMock()
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_kafka_message():
    """Factory for creating mock Kafka messages"""
    def _create_message(value: str):
        msg = MagicMock()
        msg.value = value.encode('utf-8')
        return msg
    return _create_message


@pytest.fixture
def mock_kafka_consumer():
    """Mock KafkaConsumer instance"""
    consumer = MagicMock()
    consumer.commit = MagicMock()
    consumer.close = MagicMock()
    consumer.subscribe = MagicMock()
    return consumer

@pytest.fixture
def mock_running_ssh_tunnel():
    """Mock a running SSH tunnel process for psutil.process_iter"""
    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": 12345,
        "cmdline": [
            "ssh", "-L", "9092:localhost:9092",
            "testuser@testhost.com", "-NT"
        ]
    }
    return [mock_proc]

# ============================================================================
# Data Processor Fixtures
# ============================================================================

@pytest.fixture
def mock_data_queue():
    """Mock DataQueue for testing"""
    queue = MagicMock()
    queue.enqueue_data = MagicMock()
    queue.dequeue_data = MagicMock()
    queue.is_empty = MagicMock(return_value=False)
    queue.size = MagicMock(return_value=0)
    return queue


@pytest.fixture
def sample_movie_log():
    """Sample movie log data"""
    return "2023-10-25 10:30:15,user123,GET /data/m/movie456/12.mpg"


@pytest.fixture
def sample_rating_log():
    """Sample rating log data"""
    return "2023-10-25 10:31:20,user789,GET /rate/movie101=4"


@pytest.fixture
def sample_recommendation_log():
    """Sample recommendation request log"""
    return "2023-10-25 10:32:00,user555,recommendation request server1, status 200, result: movie1,movie2,movie3, 150 ms"


@pytest.fixture
def invalid_log():
    """Invalid log that doesn't match any pattern"""
    return "invalid,log,data,format"


@pytest.fixture
def sample_log_batch():
    """Batch of mixed log messages for testing"""
    return [
        "2023-10-25 10:30:15,user1,GET /data/m/movie1/1.mpg",
        "2023-10-25 10:30:16,user2,GET /rate/movie2=5",
        "invalid message",
        "2023-10-25 10:30:17,user3,recommendation request server1, status 200, result: m1,m2, 50 ms",
        "2023-10-25 10:30:18,user1,GET /data/m/movie3/10.mpg",
    ]


@pytest.fixture
def expected_movie_data():
    """Expected processed movie data structure"""
    return {
        "type": "movie",
        "time": "2023-10-25 10:30:15",
        "userid": "user123",
        "movieid": "movie456",
        "minute": "12"
    }


@pytest.fixture
def expected_rating_data():
    """Expected processed rating data structure"""
    return {
        "type": "rating",
        "time": "2023-10-25 10:31:20",
        "userid": "user789",
        "movieid": "movie101",
        "rating": "4"
    }


@pytest.fixture
def expected_recommendation_data():
    """Expected processed recommendation data structure"""
    return {
        "type": "recommendation_request",
        "time": "2023-10-25 10:32:00",
        "userid": "user555",
        "server": "server1",
        "status": "200",
        "recommendations": "movie1,movie2,movie3",
        "responsetime": 150.0
    }



# ============================================================================
# DB Writer Fixtures
# ============================================================================

@pytest.fixture
def sample_watch_history_data():
    """Sample watch history data for validation"""
    return [
        {
            'user_id': 'user123',
            'movie_id': 'movie456',
            'total_watch_time': 5,
            'start_time': '2023-10-25 10:30:15',
            'last_watch': '2023-10-25 10:35:20'
        }
    ]


@pytest.fixture
def sample_watch_history_df():
    """Sample watch history DataFrame"""
    import pandas as pd
    return pd.DataFrame([{
        'userid': 'user123',
        'movieid': 'movie456',
        'minute': 5,
        'start_time': '2023-10-25 10:30:15',
        'last_watch': '2023-10-25 10:35:20'
    }])


@pytest.fixture
def multiple_watch_history_df():
    """Multiple watch history records DataFrame"""
    import pandas as pd
    return pd.DataFrame([
        {
            'userid': 'user123',
            'movieid': 'movie456',
            'minute': 5,
            'start_time': '2023-10-25 10:30:15',
            'last_watch': '2023-10-25 10:35:20'
        },
        {
            'userid': 'user789',
            'movieid': 'movie101',
            'minute': 10,
            'start_time': '2023-10-25 11:00:00',
            'last_watch': '2023-10-25 11:10:00'
        }
    ])


@pytest.fixture
def sample_ratings_df():
    """Sample ratings DataFrame"""
    import pandas as pd
    return pd.DataFrame([{
        'userid': 'user789',
        'movieid': 'movie101',
        'rating': '4',
        'rated_at': '2023-10-25 10:31:20'
    }])


@pytest.fixture
def multiple_ratings_df():
    """Multiple ratings records DataFrame"""
    import pandas as pd
    return pd.DataFrame([
        {
            'userid': 'user789',
            'movieid': 'movie101',
            'rating': '4',
            'rated_at': '2023-10-25 10:31:20'
        },
        {
            'userid': 'user123',
            'movieid': 'movie456',
            'rating': '5',
            'rated_at': '2023-10-25 10:32:00'
        },
        {
            'userid': 'user555',
            'movieid': 'movie789',
            'rating': '3',
            'rated_at': '2023-10-25 10:33:15'
        }
    ])


@pytest.fixture
def sample_movie_csv_df():
    """Sample movie CSV DataFrame with raw data"""
    import pandas as pd
    return pd.DataFrame([
        {
            'userid': 'user123',
            'movieid': 'movie456',
            'minute': 1,
            'time': '2023-10-25 10:30:15'
        },
        {
            'userid': 'user123',
            'movieid': 'movie456',
            'minute': 2,
            'time': '2023-10-25 10:35:20'
        }
    ])


@pytest.fixture
def sample_rating_csv_df():
    """Sample rating CSV DataFrame with raw data"""
    import pandas as pd
    return pd.DataFrame([
        {
            'userid': 'user789',
            'movieid': 'movie101',
            'rating': '4',
            'time': '2023-10-25 10:31:20'
        }
    ])
