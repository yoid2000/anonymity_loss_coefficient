"""
Test to verify that all DataFrames (orig_all and anon) are properly encoded.
"""
import pandas as pd
import sys
import os
import logging

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anonymity_loss_coefficient'))

from anonymity_loss_coefficient.alc.data_files import DataFiles
from anonymity_loss_coefficient.alc.params import ALCParams

def test_all_dataframes_encoded():
    """Test that both orig_all and anon DataFrames are properly encoded."""
    print("Testing that all DataFrames are properly encoded...")
    
    # Create test data with string values that need encoding
    orig_data = pd.DataFrame({
        'MakeModel': ['Economy', 'FamilySedan', 'Luxury', 'SportsCar'],
        'numeric_col': [1, 2, 3, 4]
    })
    
    anon_data1 = pd.DataFrame({
        'MakeModel': ['Luxury', 'Economy', 'SportsCar', 'FamilySedan'],
        'numeric_col': [5, 6, 7, 8]
    })
    
    anon_data2 = pd.DataFrame({
        'MakeModel': ['SportsCar', 'Luxury', 'Economy', 'SuperLuxury'],
        'numeric_col': [9, 10, 11, 12]
    })
    
    print("Original data:")
    print(f"orig_all:\n{orig_data}")
    print(f"anon[0]:\n{anon_data1}")
    print(f"anon[1]:\n{anon_data2}")
    
    # Create DataFiles instance manually to simulate the initialization process
    params = ALCParams()
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Create DataFiles instance manually without going through file loading
    df = DataFiles.__new__(DataFiles)
    df.logger = logger
    df.orig_all = orig_data.copy()
    df.anon = [anon_data1.copy(), anon_data2.copy()]
    df._encoders = {}
    df.discretize_in_place = True  # For simplicity
    df.columns_for_discretization = []
    df.column_classification = {}
    
    # Simulate the encoding process from DataFiles.__init__
    columns_to_encode = df.orig_all.select_dtypes(exclude=[pd.Int64Dtype(), pd.Int32Dtype(), int]).columns
    print(f"\nColumns to encode: {list(columns_to_encode)}")
    
    # Fit encoders on all DataFrames
    df._encoders = df._fit_encoders(columns_to_encode, [df.orig_all] + df.anon)
    
    print(f"\nFitted encoders:")
    for col, encoder in df._encoders.items():
        print(f"  {col}: classes = {encoder.classes_}")
    
    # Transform all DataFrames (this is the critical part we're testing)
    print(f"\nBefore transformation:")
    print(f"orig_all MakeModel types: {df.orig_all['MakeModel'].dtype}")
    print(f"anon[0] MakeModel types: {df.anon[0]['MakeModel'].dtype}")
    print(f"anon[1] MakeModel types: {df.anon[1]['MakeModel'].dtype}")
    
    df.orig_all = df._transform_df(df.orig_all)
    for i, anon_df in enumerate(df.anon):
        df.anon[i] = df._transform_df(anon_df)  # This is the fix
    
    print(f"\nAfter transformation:")
    print(f"orig_all:\n{df.orig_all}")
    print(f"anon[0]:\n{df.anon[0]}")
    print(f"anon[1]:\n{df.anon[1]}")
    
    # Verify all DataFrames are properly encoded
    print(f"\nVerification:")
    print(f"orig_all MakeModel types: {df.orig_all['MakeModel'].dtype}")
    print(f"anon[0] MakeModel types: {df.anon[0]['MakeModel'].dtype}")
    print(f"anon[1] MakeModel types: {df.anon[1]['MakeModel'].dtype}")
    
    # Check that all values are integers
    for name, data in [("orig_all", df.orig_all), ("anon[0]", df.anon[0]), ("anon[1]", df.anon[1])]:
        makemodel_values = data['MakeModel'].tolist()
        print(f"{name} MakeModel values: {makemodel_values}")
        
        all_integers = all(isinstance(val, (int, pd.Int64Dtype().type)) for val in makemodel_values)
        print(f"{name} all integers: {all_integers}")
        
        if not all_integers:
            print(f"  ❌ ERROR: {name} contains non-integer values!")
            for i, val in enumerate(makemodel_values):
                print(f"    Row {i}: {val} (type: {type(val)})")
        else:
            print(f"  ✅ {name} properly encoded")
    
    # Test decoding to verify it works
    print(f"\nDecoding test:")
    for val in df.orig_all['MakeModel'].unique():
        try:
            decoded = df.decode_value('MakeModel', val)
            print(f"  {val} -> {decoded}")
        except Exception as e:
            print(f"  {val} -> ERROR: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_all_dataframes_encoded()
