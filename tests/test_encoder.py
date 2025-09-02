"""
Test suite for encoding methods in DataFiles class.

Tests _fit_encoders(), _transform_df(), and decode_value() methods
to ensure proper encoding/decoding functionality and compatibility.
"""
import pytest
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder

from anonymity_loss_coefficient.alc.data_files import DataFiles
from anonymity_loss_coefficient.alc.params import ALCParams


class TestEncoder:
    """Test class for encoding functionality in DataFiles."""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample test data with various column types."""
        return pd.DataFrame({
            'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A'],
            'boolean_col': [True, False, True, True, False, True],
            'string_col': ['x', 'y', 'x', 'z', 'y', 'x'],
            'numeric_col': [1, 2, 3, 4, 5, 6],  # This won't be encoded
            'mixed_col': ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
        })
    
    @pytest.fixture
    def anon_data(self) -> List[pd.DataFrame]:
        """Create anonymized datasets with some variations."""
        anon1 = pd.DataFrame({
            'categorical_col': ['A', 'C', 'B', 'A', 'C'],
            'boolean_col': [False, True, False, True, False],
            'string_col': ['y', 'z', 'x', 'y', 'z'],
            'numeric_col': [7, 8, 9, 10, 11],
            'mixed_col': ['bird', 'cat', 'dog', 'bird', 'cat']
        })
        
        anon2 = pd.DataFrame({
            'categorical_col': ['B', 'C', 'A', 'B'],
            'boolean_col': [True, True, False, True],
            'string_col': ['x', 'y', 'z', 'x'],
            'numeric_col': [12, 13, 14, 15],
            'mixed_col': ['dog', 'bird', 'cat', 'dog']
        })
        
        return [anon1, anon2]
    
    @pytest.fixture
    def datafiles_instance(self, sample_data: pd.DataFrame, anon_data: List[pd.DataFrame]) -> DataFiles:
        """Create a DataFiles instance for testing."""
        # Create logger
        logger = logging.getLogger('test_encoder')
        logger.setLevel(logging.DEBUG)
        
        # Create DataFiles instance manually without file I/O
        df = DataFiles.__new__(DataFiles)
        df.logger = logger
        df.orig_all = sample_data.copy()
        df.anon = anon_data
        df._encoders = {}
        
        return df
    
    def test_fit_encoders_basic(self, datafiles_instance: DataFiles):
        """Test basic functionality of _fit_encoders."""
        columns_to_encode = ['categorical_col', 'boolean_col', 'string_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        
        encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Check that encoders were created for all specified columns
        assert len(encoders) == len(columns_to_encode)
        for col in columns_to_encode:
            assert col in encoders
            assert isinstance(encoders[col], LabelEncoder)
    
    def test_fit_encoders_combined_classes(self, datafiles_instance: DataFiles):
        """Test that _fit_encoders uses combined unique values from all DataFrames."""
        columns_to_encode = ['categorical_col', 'string_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        
        encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Check categorical_col encoder has classes from all DataFrames
        expected_categorical = sorted(['A', 'B', 'C'])  # All unique values
        assert list(encoders['categorical_col'].classes_) == expected_categorical
        
        # Check string_col encoder has classes from all DataFrames
        expected_string = sorted(['x', 'y', 'z'])  # All unique values
        assert list(encoders['string_col'].classes_) == expected_string
    
    def test_fit_encoders_boolean_handling(self, datafiles_instance: DataFiles):
        """Test that boolean columns are handled correctly."""
        columns_to_encode = ['boolean_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        
        encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Boolean encoder should have both False and True
        expected_classes = [False, True]  # LabelEncoder sorts these
        assert list(encoders['boolean_col'].classes_) == expected_classes
    
    def test_fit_encoders_empty_columns(self, datafiles_instance: DataFiles):
        """Test _fit_encoders with empty column list."""
        encoders = datafiles_instance._fit_encoders([], [datafiles_instance.orig_all])
        assert len(encoders) == 0
    
    def test_fit_encoders_missing_column(self, datafiles_instance: DataFiles):
        """Test _fit_encoders with column that doesn't exist in some DataFrames."""
        # Add a column that only exists in orig_all
        datafiles_instance.orig_all['unique_col'] = ['val1', 'val2', 'val1', 'val2', 'val1', 'val2']
        
        columns_to_encode = ['unique_col', 'categorical_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        
        encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Should still create encoders for both columns
        assert 'unique_col' in encoders
        assert 'categorical_col' in encoders
        assert list(encoders['unique_col'].classes_) == ['val1', 'val2']
    
    def test_transform_df_basic(self, datafiles_instance: DataFiles, sample_data: pd.DataFrame):
        """Test basic functionality of _transform_df."""
        # First fit encoders
        columns_to_encode = ['categorical_col', 'boolean_col', 'string_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Transform the data
        transformed = datafiles_instance._transform_df(sample_data)
        
        # Check that the DataFrame structure is preserved
        assert transformed.shape == sample_data.shape
        assert set(transformed.columns) == set(sample_data.columns)
        
        # Check that encoded columns are now integers
        for col in columns_to_encode:
            assert transformed[col].dtype in ['Int64', 'int64']  # Nullable or regular int
        
        # Check that non-encoded columns are unchanged
        pd.testing.assert_series_equal(transformed['numeric_col'], sample_data['numeric_col'])
    
    def test_transform_df_encoding_correctness(self, datafiles_instance: DataFiles, sample_data: pd.DataFrame):
        """Test that _transform_df encodes values correctly."""
        columns_to_encode = ['categorical_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        transformed = datafiles_instance._transform_df(sample_data)
        
        # Check specific encodings
        encoder = datafiles_instance._encoders['categorical_col']
        
        # 'A' should map to 0, 'B' to 1, 'C' to 2 (alphabetical order)
        expected_A = encoder.transform(['A'])[0]
        expected_B = encoder.transform(['B'])[0]
        expected_C = encoder.transform(['C'])[0]
        
        # Check first few values
        assert transformed.loc[0, 'categorical_col'] == expected_A  # 'A'
        assert transformed.loc[1, 'categorical_col'] == expected_B  # 'B'
        assert transformed.loc[3, 'categorical_col'] == expected_C  # 'C'
    
    def test_transform_df_with_nan(self, datafiles_instance: DataFiles):
        """Test _transform_df handles NaN values correctly."""
        # Create data with NaN values
        data_with_nan = pd.DataFrame({
            'categorical_col': ['A', None, 'B', 'A'],
            'string_col': ['x', 'y', None, 'x']
        })
        
        columns_to_encode = ['categorical_col', 'string_col']
        dataframes = [data_with_nan]
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        transformed = datafiles_instance._transform_df(data_with_nan)
        
        # NaN values should be encoded as -1 (our sentinel value)
        assert transformed.loc[1, 'categorical_col'] == -1
        assert transformed.loc[2, 'string_col'] == -1
        
        # Non-NaN values should be encoded properly
        assert isinstance(transformed.loc[0, 'categorical_col'], (int, np.integer))
        assert isinstance(transformed.loc[0, 'string_col'], (int, np.integer))
        assert transformed.loc[0, 'categorical_col'] >= 0
        assert transformed.loc[0, 'string_col'] >= 0
    
    def test_transform_df_missing_column(self, datafiles_instance: DataFiles, sample_data: pd.DataFrame):
        """Test _transform_df when DataFrame is missing some encoded columns."""
        # Fit encoders for more columns than exist in test data
        columns_to_encode = ['categorical_col', 'nonexistent_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        
        # Add the nonexistent column to one DataFrame for fitting
        temp_df = sample_data.copy()
        temp_df['nonexistent_col'] = ['val1', 'val2', 'val1', 'val2', 'val1', 'val2']
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, [temp_df])
        
        # Transform original data (missing nonexistent_col)
        transformed = datafiles_instance._transform_df(sample_data)
        
        # Should transform available columns and skip missing ones
        assert 'categorical_col' in transformed.columns
        assert 'nonexistent_col' not in transformed.columns
        assert transformed['categorical_col'].dtype in ['Int64', 'int64']
    
    def test_decode_value_basic(self, datafiles_instance: DataFiles):
        """Test basic functionality of decode_value."""
        columns_to_encode = ['categorical_col', 'boolean_col', 'string_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Test decoding for categorical column
        encoder = datafiles_instance._encoders['categorical_col']
        for i, class_val in enumerate(encoder.classes_):
            decoded = datafiles_instance.decode_value('categorical_col', i)
            assert decoded == str(class_val)
    
    def test_decode_value_boolean(self, datafiles_instance: DataFiles):
        """Test decode_value with boolean columns."""
        columns_to_encode = ['boolean_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Test boolean decoding
        false_encoded = datafiles_instance._encoders['boolean_col'].transform([False])[0]
        true_encoded = datafiles_instance._encoders['boolean_col'].transform([True])[0]
        
        # decode_value returns the actual numpy boolean values, not strings
        assert datafiles_instance.decode_value('boolean_col', false_encoded) == False
        assert datafiles_instance.decode_value('boolean_col', true_encoded) == True
    
    def test_decode_value_invalid_column(self, datafiles_instance: DataFiles):
        """Test decode_value with invalid column name."""
        datafiles_instance._encoders = {}
        
        # decode_value returns the encoded value as-is for missing columns, doesn't raise KeyError
        result = datafiles_instance.decode_value('nonexistent_col', 42)
        assert result == 42
    
    def test_decode_value_invalid_code(self, datafiles_instance: DataFiles):
        """Test decode_value with invalid encoded value."""
        columns_to_encode = ['categorical_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Try to decode a value outside the valid range
        num_classes = len(datafiles_instance._encoders['categorical_col'].classes_)
        
        with pytest.raises(ValueError):
            datafiles_instance.decode_value('categorical_col', num_classes + 10)
    
    def test_round_trip_encoding_decoding(self, datafiles_instance: DataFiles, sample_data: pd.DataFrame):
        """Test complete round-trip: original -> encoded -> decoded."""
        columns_to_encode = ['categorical_col', 'boolean_col', 'string_col', 'mixed_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        
        # Fit encoders and transform data
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        transformed = datafiles_instance._transform_df(sample_data)
        
        # Test round-trip for each encoded column
        for col in columns_to_encode:
            original_values = sample_data[col].dropna()
            encoded_values = transformed[col].dropna()
            
            for orig_idx in original_values.index:
                original_val = sample_data.loc[orig_idx, col]
                encoded_val = transformed.loc[orig_idx, col]
                decoded_val = datafiles_instance.decode_value(col, int(encoded_val))
                
                # Compare the actual decoded value with the original value directly
                # For boolean columns, compare the boolean values, not string representations
                if col == 'boolean_col':
                    assert bool(decoded_val) == bool(original_val), \
                        f"Round-trip failed for {col}: {original_val} -> {encoded_val} -> {decoded_val}"
                else:
                    assert str(original_val) == str(decoded_val), \
                        f"Round-trip failed for {col}: {original_val} -> {encoded_val} -> {decoded_val}"
    
    def test_consistency_across_dataframes(self, datafiles_instance: DataFiles):
        """Test that encoding is consistent across multiple DataFrames."""
        columns_to_encode = ['categorical_col', 'string_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        
        # Fit encoders on all data
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Transform each DataFrame
        transformed_dfs = []
        for df in dataframes:
            transformed_dfs.append(datafiles_instance._transform_df(df))
        
        # Check that the same original values get the same encoded values across DataFrames
        for col in columns_to_encode:
            encoder = datafiles_instance._encoders[col]
            
            # Test a specific value that appears in multiple DataFrames
            test_value = 'A' if col == 'categorical_col' else 'x'
            expected_encoded = encoder.transform([test_value])[0]
            
            for transformed_df in transformed_dfs:
                if col in transformed_df.columns:
                    # Find rows with the test value in original data
                    for i, orig_df in enumerate(dataframes):
                        if col in orig_df.columns:
                            mask = orig_df[col] == test_value
                            if mask.any():
                                # Check that encoded values match
                                encoded_values = transformed_dfs[i].loc[mask, col]
                                assert all(encoded_values == expected_encoded), \
                                    f"Inconsistent encoding for {test_value} in {col}"
    
    def test_encoder_deterministic(self, datafiles_instance: DataFiles, sample_data: pd.DataFrame):
        """Test that encoder fitting is deterministic (same input produces same output)."""
        columns_to_encode = ['categorical_col', 'string_col']
        dataframes = [sample_data]
        
        # Fit encoders twice
        encoders1 = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        encoders2 = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Should produce identical encoders
        for col in columns_to_encode:
            assert list(encoders1[col].classes_) == list(encoders2[col].classes_)
            
            # Test that they produce the same transformations
            test_values = sample_data[col].unique()
            encoded1 = encoders1[col].transform(test_values)
            encoded2 = encoders2[col].transform(test_values)
            np.testing.assert_array_equal(encoded1, encoded2)
    
    def test_decode_value_with_different_types(self, datafiles_instance: DataFiles, sample_data: pd.DataFrame):
        """Test decode_value with different numeric types that might be produced by pandas/numpy."""
        columns_to_encode = ['categorical_col', 'boolean_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        # Test with different types of encoded values
        import numpy as np
        
        # Standard int
        result1 = datafiles_instance.decode_value('categorical_col', 0)
        assert str(result1) == 'A'
        
        # Numpy int64
        result2 = datafiles_instance.decode_value('categorical_col', np.int64(1))
        assert str(result2) == 'B'
        
        # Numpy int32
        result3 = datafiles_instance.decode_value('categorical_col', np.int32(2))
        assert str(result3) == 'C'
        
        # Test error cases
        with pytest.raises(ValueError):
            datafiles_instance.decode_value('categorical_col', 999)  # Out of range
        
        with pytest.raises(ValueError):
            datafiles_instance.decode_value('categorical_col', -1)   # Negative
    
    def test_transform_df_produces_standard_int(self, datafiles_instance: DataFiles, sample_data: pd.DataFrame):
        """Test that _transform_df produces standard int64 dtype for sklearn compatibility."""
        columns_to_encode = ['categorical_col', 'boolean_col', 'string_col']
        dataframes = [datafiles_instance.orig_all] + datafiles_instance.anon
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, dataframes)
        
        transformed = datafiles_instance._transform_df(sample_data)
        
        # Check that encoded columns use standard int64 (not nullable Int64)
        for col in columns_to_encode:
            assert transformed[col].dtype == 'int64', f"Column {col} should have int64 dtype, got {transformed[col].dtype}"
            
            # Verify the values are proper integers that sklearn can handle
            for val in transformed[col]:
                assert isinstance(val, (int, np.integer)), f"Value {val} should be integer type, got {type(val)}"
    
    def test_all_dataframes_properly_encoded(self, datafiles_instance: DataFiles):
        """Test that both orig_all and anon DataFrames are properly encoded when simulating DataFiles.__init__."""
        # Create test data with string values
        orig_data = pd.DataFrame({
            'test_col': ['A', 'B', 'C'],
            'numeric_col': [1, 2, 3]
        })
        
        anon_data = pd.DataFrame({
            'test_col': ['B', 'C', 'A'],
            'numeric_col': [4, 5, 6]
        })
        
        # Reset the instance
        datafiles_instance.orig_all = orig_data.copy()
        datafiles_instance.anon = [anon_data.copy()]
        datafiles_instance._encoders = {}
        
        # Simulate the encoding process from DataFiles.__init__
        columns_to_encode = ['test_col']
        datafiles_instance._encoders = datafiles_instance._fit_encoders(columns_to_encode, [datafiles_instance.orig_all] + datafiles_instance.anon)
        
        # Transform all DataFrames (testing the critical fix)
        datafiles_instance.orig_all = datafiles_instance._transform_df(datafiles_instance.orig_all)
        for i, df in enumerate(datafiles_instance.anon):
            datafiles_instance.anon[i] = datafiles_instance._transform_df(df)  # This was the bug - previously just assigned to df
        
        # Verify all DataFrames are properly encoded
        assert datafiles_instance.orig_all['test_col'].dtype == 'int64'
        assert datafiles_instance.anon[0]['test_col'].dtype == 'int64'
        
        # Verify all values are integers (not strings)
        for val in datafiles_instance.orig_all['test_col']:
            assert isinstance(val, (int, np.integer)), f"orig_all contains non-integer: {val} (type: {type(val)})"
        
        for val in datafiles_instance.anon[0]['test_col']:
            assert isinstance(val, (int, np.integer)), f"anon[0] contains non-integer: {val} (type: {type(val)})"
        
        # Verify decoding works with the encoded values
        for df_name, df in [("orig_all", datafiles_instance.orig_all), ("anon[0]", datafiles_instance.anon[0])]:
            for encoded_val in df['test_col'].unique():
                decoded = datafiles_instance.decode_value('test_col', encoded_val)
                assert decoded in ['A', 'B', 'C'], f"Unexpected decoded value: {decoded} from {encoded_val} in {df_name}"
