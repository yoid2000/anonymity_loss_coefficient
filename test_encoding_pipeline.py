"""
Test script to validate the new encoding pipeline with decode_value() compatibility.
"""
import pandas as pd
import sys
import os
import logging

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anonymity_loss_coefficient'))

from anonymity_loss_coefficient.alc.data_files import DataFiles
from anonymity_loss_coefficient.alc.params import ALCParams

def test_encoding_pipeline():
    """Test that the new encoding pipeline works with decode_value()."""
    print("Testing encoding pipeline...")
    
    # Create test data with mixed column types including boolean
    test_data = pd.DataFrame({
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'boolean_col': [True, False, True, True, False],
        'numeric_col': [1, 2, 3, 4, 5],
        'string_col': ['x', 'y', 'x', 'z', 'y']
    })
    
    print("Original test data:")
    print(test_data)
    print(f"Data types:\n{test_data.dtypes}")
    print()
    
    # Create mock params for DataFiles
    params = ALCParams()
    params.datafile = "dummy"
    params.aux_datafile = None  # Use None for no auxiliary file
    params.cntl_size = 2
    
    # Create a simple logger
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Create DataFiles instance manually without going through file loading
    df = DataFiles.__new__(DataFiles)  # Create instance without calling __init__
    df.params = params
    df.logger = logger
    df.orig_all = test_data.copy()
    df.anon = [test_data.copy()]  # Single anonymized dataset
    df._encoders = {}
    
    # Define columns to encode (excluding numeric_col)
    columns_to_encode = ['categorical_col', 'boolean_col', 'string_col']
    
    # Fit encoders on the combined data
    print("Fitting encoders...")
    df._encoders = df._fit_encoders(columns_to_encode, [df.orig_all] + df.anon)
    
    print("Fitted encoders:")
    for col, encoder in df._encoders.items():
        print(f"  {col}: classes = {encoder.classes_}")
    print()
    
    # Transform the original data
    print("Transforming data...")
    transformed_data = df._transform_df(test_data)
    
    print("Transformed data:")
    print(transformed_data)
    print(f"Data types:\n{transformed_data.dtypes}")
    print()
    
    # Test decode_value() for each encoded column
    print("Testing decode_value()...")
    for col in columns_to_encode:
        print(f"\nTesting column '{col}':")
        unique_encoded = transformed_data[col].dropna().unique()
        
        for encoded_val in unique_encoded:
            try:
                decoded_val = df.decode_value(col, int(encoded_val))
                print(f"  Encoded {encoded_val} -> Decoded '{decoded_val}'")
                
                # Verify round-trip: original -> encoded -> decoded
                # Find the original value that corresponds to this encoded value
                mask = transformed_data[col] == encoded_val
                original_vals = test_data.loc[mask, col].drop_duplicates()
                if len(original_vals) == 1:
                    original_val = original_vals.iloc[0]
                    if str(decoded_val) == str(original_val):
                        print(f"    ✓ Round-trip successful: {original_val} -> {encoded_val} -> {decoded_val}")
                    else:
                        print(f"    ✗ Round-trip failed: {original_val} -> {encoded_val} -> {decoded_val}")
                        
            except Exception as e:
                print(f"    ✗ Error decoding {encoded_val}: {e}")
    
    print("\nEncoding pipeline test completed!")

if __name__ == "__main__":
    test_encoding_pipeline()
