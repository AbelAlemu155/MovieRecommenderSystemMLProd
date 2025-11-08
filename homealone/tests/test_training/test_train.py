"""
Tests for the Training class - fully mocked with proper patching.

Run with: pytest homealone/tests/test_training/test_train.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
import sys

# Mock surprise before any imports
sys.modules['surprise'] = MagicMock()
sys.modules['surprise.prediction_algorithms'] = MagicMock()


# ============================================================================
# Test: Initialization
# ============================================================================

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_initialization_creates_connection(mock_sql, mock_connect):
    """Test that Training class initializes and connects to database"""
    from homealone.training.train import Training
    
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    trainer = Training()
    
    assert trainer.db_path == 'db/home_alone.db'
    assert trainer.test_percentage == 0.30
    assert trainer.retrain_threshold == 0.05
    mock_connect.assert_called_once_with('db/home_alone.db')


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_initialization_sets_correct_attributes(mock_sql, mock_connect):
    """Test that all attributes are set correctly"""
    from homealone.training.train import Training
    
    mock_connect.return_value = MagicMock()
    
    trainer = Training()
    
    assert trainer.table_name == 'ratings'
    assert trainer.trainset is None
    assert trainer.training_cutoff is None
    assert trainer.model_path == '/app/models/'


# ============================================================================
# Test: Retraining Logic
# ============================================================================

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_should_retrain_first_run(mock_sql, mock_connect):
    """Test should_retrain returns True on first run"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_connect.return_value = MagicMock()
    
    def side_effect_func(query, conn):
        if 'training_metadata' in query:
            # First call: empty metadata (no previous training)
            return pd.DataFrame(columns=['total_samples'])
        elif 'COUNT' in query:
            # Second call: current count
            return pd.DataFrame({'count': [1000]})
        return pd.DataFrame()
    
    mock_sql.side_effect = side_effect_func
    
    trainer = Training()
    should_train, current, new = trainer.should_retrain()
    
    assert should_train is True
    assert current == 1000
    assert new == 0


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_should_retrain_sufficient_data(mock_sql, mock_connect):
    """Test should_retrain returns True with 5% more data"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_connect.return_value = MagicMock()
    
    def side_effect_func(query, conn):
        if 'training_metadata' in query:
            # First call: last training had 1000 samples
            return pd.DataFrame({'total_samples': [1000]})
        elif 'COUNT' in query:
            # Second call: current count is 1060 (6% more)
            return pd.DataFrame({'count': [1060]})
        return pd.DataFrame()
    
    mock_sql.side_effect = side_effect_func
    
    trainer = Training()
    should_train, current, new = trainer.should_retrain()
    
    assert should_train is True
    assert current == 1060
    assert new == 60

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_should_retrain_insufficient_data(mock_sql, mock_connect):
    """Test should_retrain returns False with insufficient data"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_connect.return_value = MagicMock()
    
    def side_effect_func(query, conn):
        if 'training_metadata' in query:
            # First call: last training had 1000 samples
            return pd.DataFrame({'total_samples': [1000]})
        elif 'COUNT' in query:
            # Second call: current count is 1030 (3% more - insufficient)
            return pd.DataFrame({'count': [1030]})
        return pd.DataFrame()
    
    mock_sql.side_effect = side_effect_func
    
    trainer = Training()
    should_train, current, new = trainer.should_retrain()
    
    assert should_train is False
    assert current == 1030
    assert new == 30


# ============================================================================
# Test: Data Loading
# ============================================================================

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('homealone.training.train.pd.to_datetime')
def test_get_percentage_based_split_empty_data(mock_datetime, mock_sql, mock_connect):
    """Test that empty data returns None"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_connect.return_value = MagicMock()
    
    # Return empty DataFrame
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    train_df, test_df, cutoff = trainer.get_percentage_based_split()
    
    assert train_df is None
    assert test_df is None
    assert cutoff is None


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('homealone.training.train.pd.to_datetime')
def test_get_percentage_based_split_insufficient_data(mock_datetime, mock_sql, mock_connect):
    """Test that insufficient data (< 100) returns None"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_connect.return_value = MagicMock()
    
    # Return small DataFrame (50 rows)
    df = pd.DataFrame({
        'user_id': list(range(50)),
        'movie_id': list(range(50)),
        'rating': [3.0] * 50,
        'rated_at': [datetime.now()] * 50
    })
    
    mock_sql.return_value = df
    mock_datetime.return_value = df['rated_at']
    
    trainer = Training()
    train_df, test_df, cutoff = trainer.get_percentage_based_split()
    
    assert train_df is None
    assert test_df is None
    assert cutoff is None


# ============================================================================
# Test: Model Training
# ============================================================================

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('homealone.training.train.SVD')
def test_train_svd_calls_surprise_svd(mock_svd, mock_sql, mock_connect):
    """Test that SVD training calls Surprise SVD correctly"""
    from homealone.training.train import Training
    
    mock_connect.return_value = MagicMock()
    mock_model = MagicMock()
    mock_svd.return_value = mock_model
    
    trainer = Training()
    trainer.trainset = MagicMock()
    
    result = trainer.train_svd()
    
    mock_svd.assert_called_once_with(
        n_factors=20,
        lr_all=0.01,
        reg_all=0.01,
        n_epochs=20,
        random_state=42
    )
    mock_model.fit.assert_called_once_with(trainer.trainset)
    assert result == mock_model


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('homealone.training.train.KNNBasic')
def test_train_knn_calls_surprise_knn(mock_knn, mock_sql, mock_connect):
    """Test that KNN training calls Surprise KNNBasic correctly"""
    from homealone.training.train import Training
    
    mock_connect.return_value = MagicMock()
    mock_model = MagicMock()
    mock_knn.return_value = mock_model
    
    trainer = Training()
    trainer.trainset = MagicMock()
    
    result = trainer.train_knn()
    
    mock_knn.assert_called_once_with(sim_options={'user_based': True})
    mock_model.fit.assert_called_once_with(trainer.trainset)
    assert result == mock_model


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('homealone.training.train.NMF')
def test_train_nmf_calls_surprise_nmf(mock_nmf, mock_sql, mock_connect):
    """Test that NMF training calls Surprise NMF correctly"""
    from homealone.training.train import Training
    
    mock_connect.return_value = MagicMock()
    mock_model = MagicMock()
    mock_nmf.return_value = mock_model
    
    trainer = Training()
    trainer.trainset = MagicMock()
    
    result = trainer.train_nmf()
    
    mock_nmf.assert_called_once_with(n_factors=20, n_epochs=20, random_state=42)
    mock_model.fit.assert_called_once_with(trainer.trainset)
    assert result == mock_model


# ============================================================================
# Test: Model Saving
# ============================================================================

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('homealone.training.train.dump.dump')
def test_save_model_calls_dump(mock_dump, mock_sql, mock_connect):
    """Test that model saving calls surprise dump"""
    from homealone.training.train import Training
    
    mock_connect.return_value = MagicMock()
    
    trainer = Training()
    mock_model = MagicMock()
    
    trainer.save_model(mock_model, '/tmp/test.dump')
    
    mock_dump.assert_called_once_with('/tmp/test.dump', algo=mock_model)


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('pickle.dump')
@patch('builtins.open', create=True)
def test_save_trainset_uses_pickle(mock_open_file, mock_pickle, mock_sql, mock_connect):
    """Test that trainset is saved using pickle"""
    from homealone.training.train import Training
    
    mock_connect.return_value = MagicMock()
    mock_file = MagicMock()
    mock_open_file.return_value.__enter__.return_value = mock_file
    
    trainer = Training()
    trainer.trainset = {'mock': 'trainset'}
    
    trainer.save_trainset()
    
    mock_pickle.assert_called_once()
    assert mock_pickle.call_args[0][0] == trainer.trainset


# ============================================================================
# Test: Model Evaluation
# ============================================================================

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
@patch('homealone.training.train.accuracy.rmse')
def test_evaluate_model_sufficient_data(mock_rmse, mock_sql, mock_connect):
    """Test model evaluation with sufficient data"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_connect.return_value = MagicMock()
    
    # Create real DataFrame for iterrows
    test_df = pd.DataFrame({
        'user_id': list(range(50)),
        'movie_id': list(range(100, 150)),
        'rating': [4.0] * 50
    })
    
    mock_model = MagicMock()
    mock_predictions = [MagicMock() for _ in range(50)]
    mock_model.test.return_value = mock_predictions
    mock_rmse.return_value = 0.85
    
    trainer = Training()
    rmse = trainer.evaluate_model(mock_model, test_df, 'TestModel')
    
    assert rmse == 0.85
    mock_model.test.assert_called_once()


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_evaluate_model_insufficient_data(mock_sql, mock_connect):
    """Test that evaluation returns None with insufficient data"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_connect.return_value = MagicMock()
    
    # Create DataFrame with only 5 rows
    test_df = pd.DataFrame({
        'user_id': list(range(5)),
        'movie_id': list(range(5)),
        'rating': [4.0] * 5
    })
    
    trainer = Training()
    rmse = trainer.evaluate_model(MagicMock(), test_df, 'TestModel')
    
    assert rmse is None


@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_evaluate_model_none_dataframe(mock_sql, mock_connect):
    """Test that evaluation handles None DataFrame"""
    from homealone.training.train import Training
    
    mock_connect.return_value = MagicMock()
    
    trainer = Training()
    rmse = trainer.evaluate_model(MagicMock(), None, 'TestModel')
    
    assert rmse is None


# ============================================================================
# Test: Metadata Logging
# ============================================================================

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_log_training_metadata_success(mock_sql, mock_connect):
    """Test successful metadata logging"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    
    # Mock read_sql_query to return empty (for initialization)
    mock_sql.return_value = pd.DataFrame()
    
    trainer = Training()
    trainer.training_cutoff = datetime(2025, 10, 27)
    
    # Create real DataFrames
    train_df = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [101, 102],
        'rating': [4.0, 3.0],
        'rated_at': [datetime(2025, 10, 25), datetime(2025, 10, 26)]
    })
    
    test_df = pd.DataFrame({
        'user_id': [1],
        'movie_id': [103],
        'rating': [5.0],
        'rated_at': [datetime(2025, 10, 27)]
    })
    
    evaluation_results = {
        'svd_rmse': 0.85,
        'knn_rmse': 0.92,
        'nmf_rmse': 0.88
    }
    
    trainer.log_training_metadata(train_df, test_df, evaluation_results)
    
    # Verify that execute was called with INSERT
    # Check all execute calls for INSERT statements
    insert_found = False
    for call_args in mock_cursor.execute.call_args_list:
        if call_args[0] and 'INSERT' in str(call_args[0][0]):
            insert_found = True
            break
    
    assert insert_found, "INSERT statement was not executed"
    
    # Verify commit was called
    assert mock_conn.commit.called

@patch('homealone.training.train.sqlite3.connect')
@patch('homealone.training.train.pd.read_sql_query')
def test_log_training_metadata_identifies_best_model(mock_sql, mock_connect):
    """Test that best model is correctly identified"""
    from homealone.training.train import Training
    import pandas as pd
    
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    
    trainer = Training()
    trainer.training_cutoff = datetime(2025, 10, 27)
    
    train_df = pd.DataFrame({
        'user_id': [1],
        'movie_id': [101],
        'rating': [4.0],
        'rated_at': [datetime(2025, 10, 25)]
    })
    
    # KNN has best (lowest) RMSE
    evaluation_results = {
        'svd_rmse': 0.95,
        'knn_rmse': 0.82,  # Best
        'nmf_rmse': 0.90
    }
    
    trainer.log_training_metadata(train_df, None, evaluation_results)
    
    # Verify 'knn_rmse' was logged as best
    execute_args = mock_cursor.execute.call_args[0][1]
    best_model = execute_args[-1]  # Last parameter
    assert best_model == 'knn_rmse'


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
