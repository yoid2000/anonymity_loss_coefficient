import pytest
import pandas as pd
from anonymity_loss_coefficient.alc.baseline_predictor import OneToOnePredictor


class TestOneToOnePredictor:
    """Test cases for OneToOnePredictor class."""
    
    def test_init_creates_mapping(self):
        """Test that initialization creates correct mapping."""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'C'],
            'target': [1, 2, 3]
        })
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        assert predictor.feature_name == 'feature'
        assert predictor.target_name == 'target'
        assert predictor.mapping == {'A': 1, 'B': 2, 'C': 3}
    
    def test_predict_valid_values(self):
        """Test prediction for valid feature values."""
        df = pd.DataFrame({
            'feature': ['X', 'Y', 'Z'],
            'target': ['apple', 'banana', 'cherry']
        })
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        assert predictor.predict('X') == 'apple'
        assert predictor.predict('Y') == 'banana'
        assert predictor.predict('Z') == 'cherry'
    
    def test_predict_invalid_value_raises_keyerror(self):
        """Test that predicting with invalid feature value raises KeyError."""
        df = pd.DataFrame({
            'feature': ['A', 'B'],
            'target': [10, 20]
        })
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        with pytest.raises(KeyError, match="Feature value 'C' not found in mapping"):
            predictor.predict('C')
    
    def test_numeric_features_and_targets(self):
        """Test with numeric feature and target values."""
        df = pd.DataFrame({
            'feature': [100, 200, 300],
            'target': [1.5, 2.5, 3.5]
        })
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        assert predictor.predict(100) == 1.5
        assert predictor.predict(200) == 2.5
        assert predictor.predict(300) == 3.5
    
    def test_mixed_data_types(self):
        """Test with mixed data types in features and targets."""
        df = pd.DataFrame({
            'feature': [1, 'text', 3.14],
            'target': ['one', 2, True]
        })
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        assert predictor.predict(1) == 'one'
        assert predictor.predict('text') == 2
        assert predictor.predict(3.14) == True
    
    def test_duplicate_rows_same_mapping(self):
        """Test handling of duplicate rows with same mapping."""
        df = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B'],
            'target': [1, 2, 1, 2]
        })
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        # Should still work correctly with duplicates
        assert predictor.predict('A') == 1
        assert predictor.predict('B') == 2
        assert len(predictor.mapping) == 2
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['feature', 'target'])
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        assert predictor.mapping == {}
        with pytest.raises(KeyError):
            predictor.predict('anything')
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'feature': ['only'],
            'target': ['value']
        })
        
        predictor = OneToOnePredictor(df, 'feature', 'target')
        
        assert predictor.predict('only') == 'value'
        assert len(predictor.mapping) == 1
    