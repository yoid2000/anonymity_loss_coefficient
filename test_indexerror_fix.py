"""
Quick test to verify that the IndexError fix works with real data types
that might be encountered in production.
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anonymity_loss_coefficient'))

from anonymity_loss_coefficient.alc.data_files import DataFiles
import logging

def test_indexerror_fix():
    """Test that various pandas/numpy integer types work with decode_value."""
    print("Testing IndexError fix...")
    
    # Create test data
    test_data = pd.DataFrame({
        'test_col': ['A', 'B', 'C', 'A', 'B']
    })
    
    # Create DataFiles instance manually
    df = DataFiles.__new__(DataFiles)
    df.logger = logging.getLogger('test')
    df.orig_all = test_data.copy()
    df.anon = [test_data.copy()]
    df._encoders = {}
    
    # Fit encoders
    columns_to_encode = ['test_col']
    df._encoders = df._fit_encoders(columns_to_encode, [df.orig_all] + df.anon)
    
    # Transform data
    transformed = df._transform_df(test_data)
    print(f"Transformed data types: {transformed.dtypes}")
    print(f"Transformed values: {transformed['test_col'].tolist()}")
    
    # Test various encoded value types that might cause IndexError
    test_cases = [
        ("Python int", 0),
        ("numpy.int64", np.int64(1)),
        ("numpy.int32", np.int32(2)),
        ("pandas scalar from iloc", transformed.iloc[0, 0]),
        ("pandas scalar from loc", transformed.loc[0, 'test_col']),
    ]
    
    print("\nTesting decode_value with different types:")
    for desc, encoded_val in test_cases:
        try:
            decoded = df.decode_value('test_col', encoded_val)
            print(f"✓ {desc} ({type(encoded_val).__name__}): {encoded_val} -> {decoded}")
        except Exception as e:
            print(f"✗ {desc} ({type(encoded_val).__name__}): {encoded_val} -> ERROR: {e}")
    
    print("\nIndexError fix test completed!")

if __name__ == "__main__":
    test_indexerror_fix()
