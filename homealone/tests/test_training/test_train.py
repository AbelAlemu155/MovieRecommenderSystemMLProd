"""
Tests for the Training class - key functionality tests with comprehensive mocking.

Run with: pytest homealone/tests/test_training/test_train.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timedelta
import sys
import pandas as pd
import os

# Mock all external dependencies before any imports
surprise_mock = MagicMock()
sys.modules['surprise'] = surprise_mock
sys.modules['surprise.prediction_algorithms'] = surprise_mock.prediction_algorithms
sys.modules['surprise.prediction_algorithms.matrix_factorization'] = surprise_mock.prediction_algorithms.matrix_factorization
sys.modules['surprise.prediction_algorithms.knns'] = surprise_mock.prediction_algorithms.knns
sys.modules['surprise.model_selection'] = surprise_mock.model_selection
sys.modules['surprise.dump'] = surprise_mock.dump_module

# Create mock classes
SVD_mock = MagicMock()
KNNBasic_mock = MagicMock()
NMF_mock = MagicMock()
Dataset_mock = MagicMock()
Reader_mock = MagicMock()
dump_mock = MagicMock()
accuracy_mock = MagicMock()

surprise_mock.SVD = SVD_mock
surprise_mock.KNNBasic = KNNBasic_mock
surprise_mock.NMF = NMF_mock
surprise_mock.Dataset = Dataset_mock
surprise_mock.Reader = Reader_mock
surprise_mock.dump = dump_mock
surprise_mock.accuracy = accuracy_mock

# Mock provenance modules
data_provenance_mock = MagicMock()
model_provenance_mock = MagicMock()
sys.modules['data_provenance'] = data_provenance_mock
sys.modules['model_provenance'] = model_provenance_mock

DataProvenance_mock = MagicMock()
ModelProvenance_mock = MagicMock()
data_provenance_mock.DataProvenance = DataProvenance_mock
model_provenance_mock.ModelProvenance = ModelProvenance_mock

# Now import Training
from homealone.training.train import Training


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_db():
    """Mock database connection"""
    with patch('homealone.training.train.sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_connect, mock_conn, mock_cursor


@pytest.fixture
def sample_train_df():
    """Sample training dataframe"""
    return pd.DataFrame({
        'user_id': list(range(100)),
        'movie_id': list(range(100, 200)),
        'rating': [4.0] * 100,
        'rated_at': pd.date_range('2025-01-01', periods=100, freq='h')
    })


@pytest.fixture
def sample_test_df():
    """Sample test dataframe"""
    return pd.DataFrame({
        'user_id': list(range(50)),
        'movie_id': list(range(100, 150)),
        'rating': [3.5] * 50,
        'rated_at': pd.date_range('2025-01-05', periods=50, freq='h')
    })


# ============================================================================
# Test: Initialization
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
def test_initialization_creates_tables(mock_sql, mock_db):
    """Test that initialization creates required tables"""
    mock_connect, mock_conn, mock_cursor = mock_db
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    
    # Check that CREATE TABLE statements were executed
    execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
    
    assert any('training_metadata' in call for call in execute_calls)
    assert any('model_deployment' in call for call in execute_calls)
    assert mock_conn.commit.called


@patch('homealone.training.train.pd.read_sql_query')
def test_initialization_sets_attributes(mock_sql, mock_db):
    """Test that initialization sets correct attributes"""
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    
    assert trainer.db_path == 'db/home_alone.db'
    assert trainer.table_name == 'ratings'
    assert trainer.test_percentage == 0.30
    assert trainer.model_path == '/app/models/'
    assert trainer.trainset is None
    assert trainer.training_cutoff is None


@patch('homealone.training.train.pd.read_sql_query')
def test_initialization_creates_provenance_trackers(mock_sql, mock_db):
    """Test that provenance trackers are initialized"""
    mock_sql.return_value = pd.DataFrame()
    
    # Reset mock call counts before creating trainer
    DataProvenance_mock.reset_mock()
    ModelProvenance_mock.reset_mock()
    
    trainer = Training()
    
    assert trainer.model_prov is not None
    assert trainer.data_prov is not None
    # Don't check exact call count - just that they were called
    assert DataProvenance_mock.called
    assert ModelProvenance_mock.called


# ============================================================================
# Test: Data Loading and Splitting
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
def test_get_percentage_based_split_success(mock_sql, mock_db):
    """Test successful data split"""
    # Create sample data with enough rows
    df = pd.DataFrame({
        'user_id': list(range(200)) * 2,  # 400 total rows
        'movie_id': list(range(200)) * 2,
        'rating': [4.0] * 400,
        'rated_at': pd.date_range('2025-01-01', periods=400, freq='h')
    })
    
    mock_sql.return_value = df
    
    trainer = Training()
    train_df, test_df, cutoff = trainer.get_percentage_based_split()
    
    assert train_df is not None
    assert test_df is not None
    assert cutoff is not None
    # Check approximate 70/30 split (allowing for warm-start filtering)
    assert len(train_df) > len(test_df)


@patch('homealone.training.train.pd.read_sql_query')
def test_get_percentage_based_split_empty_data(mock_sql, mock_db):
    """Test that empty data returns None"""
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    train_df, test_df, cutoff = trainer.get_percentage_based_split()
    
    assert train_df is None
    assert test_df is None
    assert cutoff is None


@patch('homealone.training.train.pd.read_sql_query')
def test_get_percentage_based_split_insufficient_data(mock_sql, mock_db):
    """Test that insufficient data returns None"""
    small_df = pd.DataFrame({
        'user_id': list(range(50)),
        'movie_id': list(range(50)),
        'rating': [3.0] * 50,
        'rated_at': pd.date_range('2025-01-01', periods=50, freq='h')
    })
    
    mock_sql.return_value = small_df
    
    trainer = Training()
    train_df, test_df, cutoff = trainer.get_percentage_based_split()
    
    assert train_df is None
    assert test_df is None
    assert cutoff is None


@patch('homealone.training.train.pd.read_sql_query')
def test_load_data_for_training_creates_trainset(mock_sql, mock_db):
    """Test that trainset is created from data"""
    # Create sufficient data
    df = pd.DataFrame({
        'user_id': list(range(200)) * 2,
        'movie_id': list(range(200)) * 2,
        'rating': [4.0] * 400,
        'rated_at': pd.date_range('2025-01-01', periods=400, freq='h')
    })
    
    mock_sql.return_value = df
    
    mock_trainset = MagicMock()
    mock_data = MagicMock()
    mock_data.build_full_trainset.return_value = mock_trainset
    Dataset_mock.load_from_df.return_value = mock_data
    
    trainer = Training()
    train_df, test_df = trainer.load_data_for_training()
    
    assert trainer.trainset is not None
    assert trainer.trainset == mock_trainset
    Dataset_mock.load_from_df.assert_called_once()


# ============================================================================
# Test: Model Training
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
def test_train_svd_creates_and_fits_model(mock_sql, mock_db):
    """Test SVD model training"""
    mock_sql.return_value = pd.DataFrame()
    
    mock_model = MagicMock()
    SVD_mock.return_value = mock_model
    
    trainer = Training()
    trainer.trainset = MagicMock()
    
    result = trainer.train_svd()
    
    SVD_mock.assert_called_once_with(
        n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42
    )
    mock_model.fit.assert_called_once_with(trainer.trainset)
    assert result == mock_model


@patch('homealone.training.train.pd.read_sql_query')
def test_train_nmf_creates_and_fits_model(mock_sql, mock_db):
    """Test NMF model training"""
    mock_sql.return_value = pd.DataFrame()
    
    mock_model = MagicMock()
    NMF_mock.return_value = mock_model
    
    trainer = Training()
    trainer.trainset = MagicMock()
    
    result = trainer.train_nmf()
    
    NMF_mock.assert_called_once_with(n_factors=20, n_epochs=20, random_state=42)
    mock_model.fit.assert_called_once_with(trainer.trainset)
    assert result == mock_model


# ============================================================================
# Test: Model Evaluation
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
def test_evaluate_model_success(mock_sql, mock_db, sample_test_df):
    """Test successful model evaluation"""
    mock_sql.return_value = pd.DataFrame()
    
    mock_model = MagicMock()
    mock_predictions = [MagicMock() for _ in range(50)]
    mock_model.test.return_value = mock_predictions
    accuracy_mock.rmse.return_value = 0.85
    
    trainer = Training()
    rmse = trainer.evaluate_model(mock_model, sample_test_df, 'SVD')
    
    assert rmse == 0.85
    mock_model.test.assert_called_once()
    accuracy_mock.rmse.assert_called_once()


@patch('homealone.training.train.pd.read_sql_query')
def test_evaluate_model_insufficient_data(mock_sql, mock_db):
    """Test evaluation with insufficient data"""
    mock_sql.return_value = pd.DataFrame()
    
    small_test_df = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [101, 102],
        'rating': [4.0, 3.0],
        'rated_at': [datetime.now(), datetime.now()]
    })
    
    trainer = Training()
    rmse = trainer.evaluate_model(MagicMock(), small_test_df, 'SVD')
    
    assert rmse is None


# ============================================================================
# Test: Model Saving
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
def test_save_model_calls_dump(mock_sql, mock_db):
    """Test model saving"""
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    mock_model = MagicMock()
    file_path = '/app/models/test.dump'
    
    result = trainer.save_model(mock_model, file_path)
    
    dump_mock.dump.assert_called_once_with(file_path, algo=mock_model)
    assert result == file_path


@patch('homealone.training.train.pd.read_sql_query')
@patch('builtins.open', create=True)
@patch('pickle.dump')
def test_save_trainset_uses_pickle(mock_pickle, mock_open, mock_sql, mock_db):
    """Test trainset saving"""
    mock_sql.return_value = pd.DataFrame()
    
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    trainer = Training()
    trainer.trainset = {'test': 'trainset'}
    
    result = trainer.save_trainset()
    
    mock_pickle.assert_called_once()
    assert 'trainset.dump' in result


# ============================================================================
# Test: Data Export
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
@patch('os.makedirs')
@patch('os.path.exists')
@patch('os.remove')
@patch('os.symlink')
def test_export_data_to_parquet(mock_symlink, mock_remove, mock_exists, 
                                 mock_makedirs, mock_sql, mock_db, 
                                 sample_train_df, sample_test_df):
    """Test data export to parquet"""
    # First call is for initialization, second is for export (full dataset query)
    mock_sql.side_effect = [pd.DataFrame(), sample_train_df]
    mock_exists.return_value = False
    
    with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
        trainer = Training()
        result = trainer.export_data_to_parquet(sample_train_df, sample_test_df)
        
        assert 'data_hash' in result
        assert 'timestamp' in result
        assert 'full_file' in result
        assert 'train_file' in result
        assert 'test_file' in result
        
        # Verify parquet files were written
        assert mock_to_parquet.call_count == 3  # full, train, test


# ============================================================================
# Test: Git and Versioning
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
@patch('os.getenv')
def test_get_git_commit(mock_getenv, mock_sql, mock_db):
    """Test git commit retrieval"""
    mock_sql.return_value = pd.DataFrame()
    mock_getenv.return_value = 'abc123def456'
    
    trainer = Training()
    commit = trainer.get_git_commit()
    
    assert commit == 'abc123def456'
    mock_getenv.assert_called_once_with('GIT_COMMIT', 'unknown')


@patch('homealone.training.train.pd.read_sql_query')
@patch('builtins.open', create=True)
def test_calculate_model_hash(mock_open, mock_sql, mock_db):
    """Test model hash calculation"""
    mock_sql.return_value = pd.DataFrame()
    mock_file = MagicMock()
    mock_file.read.return_value = b'model_data'
    mock_open.return_value.__enter__.return_value = mock_file
    
    trainer = Training()
    hash_val = trainer.calculate_model_hash('/app/models/test.dump')
    
    assert isinstance(hash_val, str)
    assert len(hash_val) == 8  # MD5 truncated to 8 chars


# ============================================================================
# Test: Provenance Tracking
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
@patch('builtins.open', create=True)
@patch('json.dump')
def test_save_provenance_manifest(mock_json_dump, mock_open, mock_sql, mock_db):
    """Test provenance manifest saving"""
    mock_sql.return_value = pd.DataFrame()
    
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    trainer = Training()
    provenance_data = {
        'model_version': 'v1',
        'data_version': 'd1',
        'git_commit': 'abc123'
    }
    
    result = trainer.save_provenance_manifest(provenance_data)
    
    mock_json_dump.assert_called_once()
    assert 'provenance_manifest.json' in result


# ============================================================================
# Test: Metadata Logging
# ============================================================================

@patch('homealone.training.train.pd.read_sql_query')
def test_log_training_metadata_success(mock_sql, mock_db, sample_train_df, sample_test_df):
    """Test successful metadata logging"""
    mock_connect, mock_conn, mock_cursor = mock_db
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    trainer.training_cutoff = datetime(2025, 1, 5)
    
    evaluation_results = {
        'svd_rmse': 0.85,
        'nmf_rmse': 0.92
    }
    
    trainer.log_training_metadata(
        sample_train_df, sample_test_df, evaluation_results,
        mlflow_run_id='run123',
        data_version='d1',
        git_commit='abc123',
        model_version='v1'
    )
    
    # Verify INSERT was called
    insert_calls = [call for call in mock_cursor.execute.call_args_list 
                   if 'INSERT' in str(call)]
    assert len(insert_calls) >= 2  # training_metadata + model_deployment
    assert mock_conn.commit.called


@patch('homealone.training.train.pd.read_sql_query')
def test_log_training_metadata_identifies_best_model(mock_sql, mock_db, sample_train_df):
    """Test that best model is correctly identified"""
    mock_connect, mock_conn, mock_cursor = mock_db
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    trainer.training_cutoff = datetime(2025, 1, 5)
    
    # NMF has better (lower) RMSE
    evaluation_results = {
        'svd_rmse': 0.95,
        'nmf_rmse': 0.82
    }
    
    trainer.log_training_metadata(
        sample_train_df, None, evaluation_results,
        mlflow_run_id='run123',
        data_version='d1',
        git_commit='abc123',
        model_version='v1'
    )
    
    # Find the INSERT call for training_metadata
    insert_calls = [call for call in mock_cursor.execute.call_args_list 
                   if len(call[0]) > 1 and 'INSERT INTO training_metadata' in str(call[0][0])]
    
    assert len(insert_calls) > 0, "Should have INSERT INTO training_metadata call"
    
    # Check that nmf_rmse was identified as best
    execute_args = insert_calls[0][0][1]
    assert 'nmf_rmse' in execute_args


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
