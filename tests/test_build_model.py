import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from anonymity_loss_coefficient.alc.baseline_predictor import BaselinePredictor, OneToOnePredictor
import logging


@pytest.fixture
def sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    
    # Create sample data with both categorical and continuous features
    n_train = 100
    n_test = 50
    
    # Categorical features
    cat_feature1 = np.random.choice(['A', 'B', 'C', 'D'], n_train + n_test)
    cat_feature2 = np.random.choice(['X', 'Y', 'Z'], n_train + n_test)
    
    # Continuous features  
    cont_feature1 = np.random.randn(n_train + n_test)
    cont_feature2 = np.random.randn(n_train + n_test)
    
    # Target variable (categorical)
    target = np.random.choice([0, 1, 2], n_train + n_test)
    
    # Create DataFrame
    df_all = pd.DataFrame({
        'cat_feat1': cat_feature1,
        'cat_feat2': cat_feature2,
        'cont_feat1': cont_feature1,
        'cont_feat2': cont_feature2,
        'target': target
    })
    
    df_train = df_all.iloc[:n_train].copy()
    df_test = df_all.iloc[n_train:].copy()
    
    return df_train, df_test


@pytest.fixture
def column_classifications():
    """Define column classifications."""
    return {
        'cat_feat1': 'categorical',
        'cat_feat2': 'categorical', 
        'cont_feat1': 'continuous',
        'cont_feat2': 'continuous'
    }


@pytest.fixture
def mock_score_interval():
    """Create a mock ScoreInterval object."""
    si = Mock()
    si.compute_best_prc = Mock(return_value={'prc': 0.7})
    return si


@pytest.fixture
def baseline_predictor():
    """Create a BaselinePredictor instance."""
    logger = logging.getLogger('test_logger')
    return BaselinePredictor(logger)


class TestOneToOnePredictor:
    """Test the OneToOnePredictor class."""
    
    def test_init(self, sample_data):
        """Test OneToOnePredictor initialization."""
        df_train, _ = sample_data
        
        otop = OneToOnePredictor(df_train, 'cat_feat1', 'target')
        
        assert otop.feature_name == 'cat_feat1'
        assert otop.target_name == 'target'
        assert isinstance(otop.mapping, dict)
        assert len(otop.mapping) > 0
    
    def test_predict(self, sample_data):
        """Test OneToOnePredictor prediction."""
        df_train, _ = sample_data
        
        otop = OneToOnePredictor(df_train, 'cat_feat1', 'target')
        
        # Test with a known feature value
        feature_values = df_train['cat_feat1'].unique()
        test_value = feature_values[0]
        
        prediction = otop.predict(test_value)
        assert prediction is not None
        
        # Test with unknown feature value
        with pytest.raises(KeyError):
            otop.predict('unknown_value')


class TestBaselinePredictorBuildModel:
    """Test the BaselinePredictor build_model method."""
    
    def test_build_model_initial_call(self, baseline_predictor, sample_data, column_classifications, mock_score_interval):
        """Test initial build_model call with all parameters."""
        df_train, df_test = sample_data
        known_columns = ['cat_feat1', 'cat_feat2', 'cont_feat1', 'cont_feat2']
        secret_column = 'target'
        
        model_name = baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=known_columns,
            secret_column=secret_column,
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        # Check that configuration was stored
        assert baseline_predictor.known_columns == known_columns
        assert baseline_predictor.secret_column == secret_column
        assert baseline_predictor.si == mock_score_interval
        assert baseline_predictor.selected_model_name is not None
        assert baseline_predictor.df_pred_conf is not None
        assert isinstance(model_name, str)
        
        # Check that columns were classified correctly
        expected_onehot = ['cat_feat1', 'cat_feat2']
        expected_continuous = ['cont_feat1', 'cont_feat2']
        assert set(baseline_predictor.onehot_columns) == set(expected_onehot)
        assert set(baseline_predictor.non_onehot_columns) == set(expected_continuous)
    
    def test_build_model_subsequent_call(self, baseline_predictor, sample_data, column_classifications, mock_score_interval):
        """Test subsequent build_model call without optional parameters."""
        df_train, df_test = sample_data
        known_columns = ['cat_feat1', 'cat_feat2', 'cont_feat1', 'cont_feat2']
        secret_column = 'target'
        
        # Initial call
        baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=known_columns,
            secret_column=secret_column,
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        # Store original configuration
        original_model_name = baseline_predictor.selected_model_name
        
        # Subsequent call without optional parameters
        df_train_new = df_train.sample(n=50, random_state=123).copy()
        df_test_new = df_test.sample(n=25, random_state=123).copy()
        
        model_name = baseline_predictor.build_model(
            df_train=df_train_new,
            df_test=df_test_new,
            random_state=42
        )
        
        # Should use stored configuration
        assert model_name == original_model_name
        assert baseline_predictor.df_pred_conf is not None
        
    def test_build_model_without_initial_call(self, baseline_predictor, sample_data):
        """Test that build_model fails when called without initial configuration."""
        df_train, df_test = sample_data
        
        with pytest.raises(ValueError, match="Must call build_model with optional parameters first"):
            baseline_predictor.build_model(
                df_train=df_train,
                df_test=df_test,
                random_state=42
            )
    
    def test_build_model_prediction_dataframe_structure(self, baseline_predictor, sample_data, column_classifications, mock_score_interval):
        """Test that df_pred_conf has the correct structure."""
        df_train, df_test = sample_data
        known_columns = ['cat_feat1', 'cat_feat2', 'cont_feat1', 'cont_feat2']
        secret_column = 'target'
        
        baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=known_columns,
            secret_column=secret_column,
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        # Check df_pred_conf structure
        df_pred_conf = baseline_predictor.df_pred_conf
        assert isinstance(df_pred_conf, pd.DataFrame)
        assert len(df_pred_conf) > 0
        
        required_columns = ['predicted_value', 'prediction', 'confidence']
        for col in required_columns:
            assert col in df_pred_conf.columns
        
        # Check data types
        assert df_pred_conf['prediction'].dtype == bool
        assert pd.api.types.is_numeric_dtype(df_pred_conf['confidence'])


class TestBaselinePredictorPredict:
    """Test the BaselinePredictor predict method."""
    
    def test_predict_success(self, baseline_predictor, sample_data, column_classifications, mock_score_interval):
        """Test successful prediction."""
        df_train, df_test = sample_data
        known_columns = ['cat_feat1', 'cat_feat2', 'cont_feat1', 'cont_feat2']
        secret_column = 'target'
        
        baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=known_columns,
            secret_column=secret_column,
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        # Test prediction
        predicted_value, confidence = baseline_predictor.predict(0)
        
        assert predicted_value is not None
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
    
    def test_predict_without_model(self, baseline_predictor):
        """Test that predict fails when model hasn't been built."""
        with pytest.raises(ValueError, match="Model has not been built yet"):
            baseline_predictor.predict(0)
    
    def test_predict_invalid_index(self, baseline_predictor, sample_data, column_classifications, mock_score_interval):
        """Test prediction with invalid index."""
        df_train, df_test = sample_data
        known_columns = ['cat_feat1', 'cat_feat2', 'cont_feat1', 'cont_feat2']
        secret_column = 'target'
        
        baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=known_columns,
            secret_column=secret_column,
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        # Test with negative index
        with pytest.raises(IndexError):
            baseline_predictor.predict(-1)
        
        # Test with too large index
        with pytest.raises(IndexError):
            baseline_predictor.predict(1000)
    
    def test_predict_all_test_indices(self, baseline_predictor, sample_data, column_classifications, mock_score_interval):
        """Test prediction for all test data indices."""
        df_train, df_test = sample_data
        known_columns = ['cat_feat1', 'cat_feat2', 'cont_feat1', 'cont_feat2']
        secret_column = 'target'
        
        baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=known_columns,
            secret_column=secret_column,
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        # Test prediction for all valid indices
        n_test = len(baseline_predictor.df_pred_conf)
        for i in range(n_test):
            predicted_value, confidence = baseline_predictor.predict(i)
            assert predicted_value is not None
            assert isinstance(confidence, (int, float))
            assert 0 <= confidence <= 1


class TestBaselinePredictorIntegration:
    """Integration tests for BaselinePredictor."""
    
    def test_complete_workflow(self, baseline_predictor, sample_data, column_classifications, mock_score_interval):
        """Test complete workflow: build model, then predict multiple times."""
        df_train, df_test = sample_data
        known_columns = ['cat_feat1', 'cat_feat2', 'cont_feat1', 'cont_feat2']
        secret_column = 'target'
        
        # Initial model build
        model_name1 = baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=known_columns,
            secret_column=secret_column,
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        # Make some predictions
        predictions1 = []
        for i in range(min(5, len(baseline_predictor.df_pred_conf))):
            pred, conf = baseline_predictor.predict(i)
            predictions1.append((pred, conf))
        
        # Rebuild model with new data
        df_train_new = df_train.sample(n=80, random_state=456).copy()
        df_test_new = df_test.sample(n=40, random_state=456).copy()
        
        model_name2 = baseline_predictor.build_model(
            df_train=df_train_new,
            df_test=df_test_new,
            random_state=42
        )
        
        # Should use same model type
        assert model_name1 == model_name2
        
        # Make predictions on new data
        predictions2 = []
        for i in range(min(5, len(baseline_predictor.df_pred_conf))):
            pred, conf = baseline_predictor.predict(i)
            predictions2.append((pred, conf))
        
        # Should have predictions for both datasets
        assert len(predictions1) > 0
        assert len(predictions2) > 0
    
    def test_one_to_one_predictor_selection(self, baseline_predictor, mock_score_interval):
        """Test scenario where OneToOnePredictor is selected."""
        # Create data with perfect correlation between feature and target
        df_train = pd.DataFrame({
            'perfect_feature': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'other_feature': np.random.randn(60),
            'target': [0, 1, 2, 0, 1, 2] * 10
        })
        
        df_test = pd.DataFrame({
            'perfect_feature': ['A', 'B', 'C', 'A', 'B'],
            'other_feature': np.random.randn(5),
            'target': [0, 1, 2, 0, 1]
        })
        
        column_classifications = {
            'perfect_feature': 'categorical',
            'other_feature': 'continuous'
        }
        
        # Mock the score interval to make OneToOnePredictor win
        mock_score_interval.compute_best_prc = Mock(side_effect=[
            {'prc': 0.99},  # OneToOnePredictor score (first call)
            {'prc': 0.70},  # RandomForest score
            {'prc': 0.65},  # ExtraTrees score
            {'prc': 0.60},  # HistGB score
            {'prc': 0.55},  # LogisticRegression score
        ])
        
        model_name = baseline_predictor.build_model(
            df_train=df_train,
            df_test=df_test,
            known_columns=['perfect_feature', 'other_feature'],
            secret_column='target',
            column_classifications=column_classifications,
            si=mock_score_interval,
            random_state=42
        )
        
        assert model_name == "OneToOnePredictor"
        assert baseline_predictor.otop is not None
        assert baseline_predictor.selected_model is None  # No ML model should be stored


if __name__ == "__main__":
    pytest.main([__file__])
