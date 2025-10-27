import pytest
from unittest.mock import MagicMock, Mock, patch
import subprocess
from homealone.data_processing.kafka_pipeline import KafkaSSHClient


# ============================================================================
# Initialization Tests
# ============================================================================

def test_client_initializes_with_all_parameters():
    """Test KafkaSSHClient stores all initialization parameters"""
    client = KafkaSSHClient("user", "host.com", "pass", "9092", "9092")
    
    assert client.username == "user"
    assert client.host == "host.com"
    assert client.password == "pass"
    assert client.local_port == "9092"
    assert client.remote_port == "9092"
    assert client.ssh_proc is None


def test_client_initializes_without_password():
    """Test initialization works with None password (SSH key auth)"""
    client = KafkaSSHClient("user", "host.com", None, "9092", "9092")
    
    assert client.password is None
    assert client.ssh_proc is None


# ============================================================================
# Tunnel Detection Tests
# ============================================================================

@patch('homealone.data_processing.kafka_pipeline.psutil.process_iter')
def test_is_tunnel_running_detects_existing_tunnel(mock_process_iter, kafka_client_params):
    """Test detection of running SSH tunnel"""
    mock_proc = MagicMock()
    mock_proc.info = {
        "cmdline": [
            "ssh", "-L", "9092:localhost:9092",
            "testuser@testhost.com", "-NT"
        ]
    }
    mock_process_iter.return_value = [mock_proc]

    client = KafkaSSHClient(**kafka_client_params)
    assert client._is_tunnel_running() is True



@patch('homealone.data_processing.kafka_pipeline.psutil.process_iter')
def test_is_tunnel_running_returns_false_when_no_tunnel(mock_process_iter, kafka_client_params):
    """Test returns False when no SSH tunnel exists"""
    mock_proc = MagicMock()
    mock_proc.info = {"cmdline": ["python", "script.py"]}
    mock_process_iter.return_value = [mock_proc]
    
    client = KafkaSSHClient(**kafka_client_params)
    assert client._is_tunnel_running() is False


@patch('homealone.data_processing.kafka_pipeline.psutil.process_iter')
def test_is_tunnel_running_handles_access_denied(mock_process_iter, kafka_client_params):
    """Test gracefully handles psutil.AccessDenied"""
    import psutil
    
    mock_proc = MagicMock()
    mock_proc.info.side_effect = psutil.AccessDenied()
    mock_process_iter.return_value = [mock_proc]
    
    client = KafkaSSHClient(**kafka_client_params)
    assert client._is_tunnel_running() is False


@patch('homealone.data_processing.kafka_pipeline.psutil.process_iter')
def test_is_tunnel_running_ignores_different_port(mock_process_iter, kafka_client_params):
    """Test ignores SSH tunnel on different port"""
    mock_proc = MagicMock()
    mock_proc.info = {
        "cmdline": ["ssh", "-L", "9093:localhost:9093", "user@host.com", "-NT"]
    }
    mock_process_iter.return_value = [mock_proc]
    
    client = KafkaSSHClient(**kafka_client_params)
    assert client._is_tunnel_running() is False


# ============================================================================
# Tunnel Start Tests
# ============================================================================

@patch('homealone.data_processing.kafka_pipeline.subprocess.Popen')
def test_start_tunnel_uses_sshpass_with_password(mock_popen):
    """Test SSH tunnel starts with sshpass when password provided"""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    
    client = KafkaSSHClient("user", "host.com", "pass", "9092", "9092")
    client._start_tunnel()
    
    expected_cmd = [
        "sshpass", "-p", "pass",
        "ssh", "-L", "9092:localhost:9092", "user@host.com", "-NT"
    ]
    mock_popen.assert_called_once_with(
        expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    assert client.ssh_proc == mock_proc


@patch('homealone.data_processing.kafka_pipeline.subprocess.Popen')
def test_start_tunnel_uses_ssh_without_password(mock_popen):
    """Test SSH tunnel starts without sshpass when no password"""
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    
    client = KafkaSSHClient("user", "host.com", None, "9092", "9092")
    client._start_tunnel()
    
    expected_cmd = ["ssh", "-L", "9092:localhost:9092", "user@host.com", "-NT"]
    mock_popen.assert_called_once_with(
        expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


# ============================================================================
# Tunnel Orchestration Tests
# ============================================================================

@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient._start_tunnel')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient._is_tunnel_running')
def test_ensure_tunnel_starts_when_not_running(mock_is_running, mock_start, kafka_client_params):
    """Test tunnel starts when not already running"""
    mock_is_running.return_value = False
    
    client = KafkaSSHClient(**kafka_client_params)
    client.ensure_tunnel()
    
    mock_is_running.assert_called_once()
    mock_start.assert_called_once()


@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient._start_tunnel')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient._is_tunnel_running')
def test_ensure_tunnel_skips_when_already_running(mock_is_running, mock_start, kafka_client_params):
    """Test tunnel not started when already running"""
    mock_is_running.return_value = True
    
    client = KafkaSSHClient(**kafka_client_params)
    client.ensure_tunnel()
    
    mock_is_running.assert_called_once()
    mock_start.assert_not_called()


# ============================================================================
# Consume (Streaming) Tests
# ============================================================================

@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_yields_decoded_messages(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test consume yields UTF-8 decoded messages"""
    mock_msg1 = MagicMock(value=b'msg1')
    mock_msg2 = MagicMock(value=b'msg2')
    
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(return_value=iter([mock_msg1, mock_msg2]))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    messages = list(client.consume(topic="test", group_id="group"))
    
    assert messages == ['msg1', 'msg2']
    mock_consumer.commit.assert_called_once()
    mock_consumer.close.assert_called_once()


@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_uses_default_parameters(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test consume uses correct defaults"""
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(return_value=iter([]))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    list(client.consume())
    
    mock_consumer_class.assert_called_once_with(
        "movielog10",
        bootstrap_servers="localhost:9092",
        group_id="team10-consumer-group",
        auto_offset_reset="latest",
        enable_auto_commit=False
    )


@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_commits_and_closes_on_exception(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test consumer closes properly even when exception occurs"""
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(side_effect=Exception("Error"))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    
    with pytest.raises(Exception, match="Error"):
        list(client.consume())
    
    mock_consumer.commit.assert_called_once()
    mock_consumer.close.assert_called_once()


@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_handles_utf8_characters(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test UTF-8 special characters are decoded correctly"""
    mock_msg = MagicMock(value='ñoño'.encode('utf-8'))
    
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(return_value=iter([mock_msg]))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    messages = list(client.consume())
    
    assert messages == ['ñoño']


# ============================================================================
# Consume Batch Tests
# ============================================================================

@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_batch_returns_exact_size(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test consume_batch returns exactly batch_size messages"""
    mock_msgs = [MagicMock(value=f'msg{i}'.encode()) for i in range(15)]
    
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(return_value=iter(mock_msgs))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    messages = client.consume_batch(batch_size=10)
    
    assert len(messages) == 10
    assert messages[0] == 'msg0'
    assert messages[9] == 'msg9'


@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_batch_handles_fewer_messages(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test consume_batch when fewer messages available than requested"""
    mock_msgs = [MagicMock(value=f'msg{i}'.encode()) for i in range(5)]
    
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(return_value=iter(mock_msgs))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    messages = client.consume_batch(batch_size=10)
    
    assert len(messages) == 5


@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_batch_sets_timeout(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test consume_batch sets consumer_timeout_ms"""
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(return_value=iter([]))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    client.consume_batch(batch_size=5)
    
    call_kwargs = mock_consumer_class.call_args[1]
    assert call_kwargs['consumer_timeout_ms'] == 10000


@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.KafkaSSHClient.ensure_tunnel')
def test_consume_batch_handles_exception_gracefully(mock_ensure, mock_consumer_class, kafka_client_params):
    """Test consume_batch returns empty list on exception"""
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(side_effect=Exception("Error"))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient(**kafka_client_params)
    messages = client.consume_batch()
    
    assert messages == []
    mock_consumer.close.assert_called_once()


# ============================================================================
# Close Tests
# ============================================================================

def test_close_terminates_ssh_process():
    """Test close terminates SSH process when present"""
    client = KafkaSSHClient("user", "host.com", "pass", "9092", "9092")
    
    mock_proc = MagicMock()
    client.ssh_proc = mock_proc
    
    client.close()
    
    mock_proc.terminate.assert_called_once()
    assert client.ssh_proc is None


def test_close_handles_no_ssh_process():
    """Test close doesn't error when no SSH process exists"""
    client = KafkaSSHClient("user", "host.com", "pass", "9092", "9092")
    client.ssh_proc = None
    
    # Should not raise
    client.close()


# ============================================================================
# Integration Tests
# ============================================================================

@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.subprocess.Popen')
@patch('homealone.data_processing.kafka_pipeline.psutil.process_iter')
def test_full_workflow_tunnel_consume_close(mock_process_iter, mock_popen, mock_consumer_class):
    """Test complete workflow: start tunnel, consume, close"""
    # No tunnel running
    mock_process_iter.return_value = []
    
    # Mock tunnel
    mock_ssh_proc = MagicMock()
    mock_popen.return_value = mock_ssh_proc
    
    # Mock consumer
    mock_msgs = [MagicMock(value=f'msg{i}'.encode()) for i in range(5)]
    mock_consumer = MagicMock()
    mock_consumer.__iter__ = Mock(return_value=iter(mock_msgs))
    mock_consumer_class.return_value = mock_consumer
    
    client = KafkaSSHClient("user", "host.com", "pass", "9092", "9092")
    messages = list(client.consume())
    client.close()
    
    assert len(messages) == 5
    mock_popen.assert_called_once()  # Tunnel started
    mock_consumer.commit.assert_called_once()
    mock_ssh_proc.terminate.assert_called_once()  # Tunnel closed


@patch('homealone.data_processing.kafka_pipeline.KafkaConsumer')
@patch('homealone.data_processing.kafka_pipeline.psutil.process_iter')
def test_sequential_batch_consumption(mock_process_iter, mock_consumer_class, mock_running_ssh_tunnel):
    """Test consuming multiple batches sequentially"""
    mock_process_iter.return_value = mock_running_ssh_tunnel
    
    # Two separate batches
    mock_msgs1 = [MagicMock(value=f'b1_{i}'.encode()) for i in range(10)]
    mock_msgs2 = [MagicMock(value=f'b2_{i}'.encode()) for i in range(10)]
    
    mock_consumer1 = MagicMock()
    mock_consumer1.__iter__ = Mock(return_value=iter(mock_msgs1))
    
    mock_consumer2 = MagicMock()
    mock_consumer2.__iter__ = Mock(return_value=iter(mock_msgs2))
    
    mock_consumer_class.side_effect = [mock_consumer1, mock_consumer2]
    
    client = KafkaSSHClient("user", "host.com", "pass", "9092", "9092")
    batch1 = client.consume_batch(batch_size=10)
    batch2 = client.consume_batch(batch_size=10)
    
    assert len(batch1) == 10
    assert len(batch2) == 10
    assert batch1[0] == 'b1_0'
    assert batch2[0] == 'b2_0'
    mock_consumer1.commit.assert_called_once()
    mock_consumer2.commit.assert_called_once()
