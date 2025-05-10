import pandas as pd
import numpy as np
from typing import Any, Tuple, Dict, List, Union

# Yes, that's right, we are banking on the real data not having these values  :(
num_filler = -123456789
str_filler = "no__match_xyz"

def find_best_matches(df_candidates: pd.DataFrame,
                      df_query: pd.DataFrame,
                      column_classifications: Dict[str, str],
                      columns: list = None,
                      debug_on: bool = False) -> Tuple[pd.Index, float]:
    """
    Optimized version of find_best_matches to compute Gower distances efficiently.

    Parameters:
    - df_candidates: pd.DataFrame, the DataFrame with multiple rows.
    - df_query: pd.DataFrame, the DataFrame with a single row (same columns as df_candidates).
    - column_classifications: dict, a dictionary mapping column names to "categorical" or "continuous".
    - columns: list, optional, the list of columns to use for the Gower distance calculation.
    - debug_on: bool, if True, writes a CSV file 'find_best_matches.csv' with debug information.

    Returns:
    - Tuple:
        - pd.Index: Indices of rows in df_candidates that share the minimum Gower distance.
        - float: The minimum Gower distance.
    """
    if len(df_query) != 1:
        raise ValueError("df_query must contain exactly one row.")

    if column_classifications is None:
        raise ValueError("column_classifications must be provided.")

    # If columns are specified, filter both DataFrames to include only those columns
    if columns is not None:
        df_candidates = df_candidates[columns]
        df_query = df_query[columns]

    # Ensure df_query is a single row
    query_row = df_query.iloc[0]

    # Separate continuous and categorical columns
    continuous_cols = [col for col, cls in column_classifications.items() if cls == "continuous"]
    continuous_cols = [col for col in continuous_cols if col in df_candidates.columns]
    categorical_cols = [col for col, cls in column_classifications.items() if cls == "categorical"]
    categorical_cols = [col for col in categorical_cols if col in df_candidates.columns]

    # Precompute min and max for continuous columns
    if continuous_cols:
        # We given NULL values the mean so that they kindof average out when computing the distance
        col_min = df_candidates[continuous_cols].min()
        col_max = df_candidates[continuous_cols].max()

        # Avoid division by zero
        range_values = (col_max - col_min).replace(0, 1)

        # Normalize continuous columns in df_candidates and df_query
        normalized_candidates = (df_candidates[continuous_cols] - col_min) / range_values
        normalized_query = (query_row[continuous_cols] - col_min) / range_values
        # A normalized query value can be outside the range [0, 1] if
        # it is outside the range of the candidates. So we clip it to [0, 1].
        normalized_query = normalized_query.clip(0, 1)

        # Compute absolute differences for continuous columns
        continuous_distances = np.abs(normalized_candidates - normalized_query)
        continuous_distances_sum = continuous_distances.sum(axis=1)
    else:
        continuous_distances = pd.DataFrame(np.zeros((len(df_candidates), 0)), index=df_candidates.index)
        continuous_distances_sum = np.zeros(len(df_candidates))

    # Compute binary distances for categorical columns
    if categorical_cols:
        categorical_distances = (df_candidates[categorical_cols] != query_row[categorical_cols]).astype(int)
        categorical_distances_sum = categorical_distances.sum(axis=1)
    else:
        categorical_distances = pd.DataFrame(np.zeros((len(df_candidates), 0)), index=df_candidates.index)
        categorical_distances_sum = np.zeros(len(df_candidates))

    # Combine distances and normalize by the number of features
    total_features = len(continuous_cols) + len(categorical_cols)
    distances = (continuous_distances_sum + categorical_distances_sum) / total_features

    # If debug_on is True, write debug information to a CSV file
    if debug_on:
        debug_data = pd.DataFrame({'Distance': distances})
        for col in df_candidates.columns:
            debug_data[f'{col}_candidate'] = df_candidates[col]
            debug_data[f'{col}_query'] = query_row[col]
            if col in continuous_cols:
                debug_data[f'{col}_distance'] = continuous_distances[col]
            elif col in categorical_cols:
                debug_data[f'{col}_distance'] = categorical_distances[col]
        debug_data.to_csv('find_best_matches.csv', index=False)

    # Find the minimum Gower distance
    min_distance = distances.min()

    # Find the indices of rows in df_candidates with the minimum Gower distance
    idx = df_candidates.index[distances == min_distance]

    return pd.Index(idx), round(float(min_distance), 3)

def modal_fraction(df_candidates: pd.DataFrame, idx: pd.Index, column: str) -> Tuple[Any, int]:
    """
    Determines the modal value, its count, and the fraction of rows in df_candidates[column]
    defined by idx that have the modal value.

    Parameters:
    - df_candidates: pd.DataFrame, the DataFrame containing the data.
    - idx: pd.Index, the indices of the rows to consider.
    - column: str, the column in which to find the modal value.

    Returns:
    - modal_value: The modal value in the specified column.
    - count: The count of the modal value (as a Python int).
    - fraction: The fraction of rows in idx that have the modal value (as a Python float).
    """
    # Filter the rows defined by idx
    subset = df_candidates.loc[idx, column]

    # Determine the modal value and its count
    modal_value = subset.mode().iloc[0]  # Get the first modal value (in case of ties)
    count = int((subset == modal_value).sum())  # Convert to Python int

    return modal_value, count

def best_match_confidence(gower_distance: float, modal_fraction: float, match_count: int) -> float:
    '''
    This function for computing the best match confidence penalizes the confidence score
    where there are many matches, because this suggests that the match can be for many
    different individuals, not just the target individual. It also penalizes the score
    if the fraction of individuals with the modal value is low, because it is less likely
    that the value is indeed for the target. 
    '''
    if match_count == 0:
        raise ValueError("Error: match_count cannot be zero.")
    if not (0 <= gower_distance <= 1):
        raise ValueError(f"Error: gower_distance ({gower_distance}) must be between 0 and 1.")
    if not (0 <= modal_fraction <= 1):
        raise ValueError("Error: modal_fraction must be between 0 and 1.")
    return round((1 - gower_distance) * modal_fraction * (1/match_count), 3)

if __name__ == "__main__":
    import pandas as pd

    # Test for find_best_matches
    print("Testing find_best_matches...")
    df_candidates = pd.DataFrame({
        'Age': [25, 35, 45, 25, 35],
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female'],
        'Income': [50000, 60000, 55000, 50000, 60000]
    })

    df_query = pd.DataFrame({
        'Age': [30],
        'Gender': ['Male'],
        'Income': [52000]
    })

    # Column classifications
    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }

    # Test 1: Find best matches using all columns
    idx_all, min_distance_all = find_best_matches(df_candidates, df_query, column_classifications=column_classifications)
    expected_idx_all = pd.Index([0, 3])  # Expected indices
    expected_min_distance_all = 0.15  # Expected minimum Gower distance
    print("Rows corresponding to idx_all:")
    print(df_candidates.loc[idx_all])  # Print rows corresponding to idx_all
    if idx_all.equals(expected_idx_all) and min_distance_all == expected_min_distance_all:
        print("Test 1 Passed: Indices and minimum Gower distance (all columns) are correct.")
    else:
        print("Test 1 Failed: Expected", (expected_idx_all, expected_min_distance_all),
              "but got", (idx_all, min_distance_all))

    # Test 2: Find best matches using only 'Age' and 'Income'
    idx_subset, min_distance_subset = find_best_matches(df_candidates, df_query, column_classifications=column_classifications, columns=['Age', 'Income'])
    expected_idx_subset = pd.Index([0, 3])  # Expected indices
    expected_min_distance_subset = 0.225  # Expected minimum Gower distance
    print("Rows corresponding to idx_subset:")
    print(df_candidates.loc[idx_subset])  # Print rows corresponding to idx_subset
    if idx_subset.equals(expected_idx_subset) and min_distance_subset == expected_min_distance_subset:
        print("Test 2 Passed: Indices and minimum Gower distance (subset of columns) are correct.")
    else:
        print("Test 2 Failed: Expected", (expected_idx_subset, expected_min_distance_subset),
              "but got", (idx_subset, min_distance_subset))

    # Test for modal_fraction
    print("\nTesting modal_fraction...")
    idx = pd.Index([0, 1, 3])

    # Test 3: Modal value and count for 'Gender'
    modal_value, count = modal_fraction(df_candidates, idx, 'Gender')
    expected_modal_value = 'Male'
    expected_count = 2
    print("Rows corresponding to idx:")
    print(df_candidates.loc[idx])  # Print rows corresponding to idx
    if modal_value == expected_modal_value and count == expected_count:
        print("Test 3 Passed: Modal value and count for 'Gender' are correct.")
    else:
        print("Test 3 Failed: Expected", (expected_modal_value, expected_count),
              "but got", (modal_value, count))

    # Test 4: Modal value and count for 'Age'
    modal_value, count = modal_fraction(df_candidates, idx, 'Age')
    expected_modal_value = 25
    expected_count = 2
    print("Rows corresponding to idx:")
    print(df_candidates.loc[idx])  # Print rows corresponding to idx
    if modal_value == expected_modal_value and count == expected_count:
        print("Test 4 Passed: Modal value and count for 'Age' are correct.")
    else:
        print("Test 4 Failed: Expected", (expected_modal_value, expected_count),
              "but got", (modal_value, count))

    # Test for best_match_confidence
    print("\nTesting best_match_confidence...")
    gower_distance = 0.15
    modal_fraction_value = 0.667
    match_count = 3
    confidence = best_match_confidence(gower_distance, modal_fraction_value, match_count)
    expected_confidence = round((1 - gower_distance) * modal_fraction_value * (1 / match_count), 3)
    if confidence == expected_confidence:
        print("Test 5 Passed: Best match confidence is correct.")
    else:
        print("Test 5 Failed: Expected", expected_confidence, "but got", confidence)

def create_full_anon(anon: Union[pd.DataFrame, List[pd.DataFrame]],
                     column_classifications: Dict[str, str],
                    ) -> pd.DataFrame:
    """
    Processes `anon` to create a single DataFrame `df_anon_full` with columns defined by `original_columns`.
    Missing columns in the DataFrames are filled with NULL values.

    Args:
        anon: A DataFrame or a list of DataFrames.
        original_columns: A list of column names to ensure in the final DataFrame.

    Returns:
        pd.DataFrame: The concatenated DataFrame with all `original_columns`.
    """
    all_anon_columns = set()
    for df in anon:
        all_anon_columns.update(df.columns.tolist())

    if isinstance(anon, pd.DataFrame):
        # If `anon` is a single DataFrame, ensure it has all `original_columns`
        df_anon_full = anon.reindex(columns=list(all_anon_columns))
    elif isinstance(anon, list):
        # If `anon` is a list of DataFrames, reindex each and concatenate
        df_anon_full = pd.concat(
            [df.reindex(columns=list(all_anon_columns)) for df in anon],
            ignore_index=True
        )
    else:
        raise ValueError("`anon` must be a DataFrame or a list of DataFrames.")
    
    for column, column_class in column_classifications.items():
        if column not in df_anon_full.columns:
            continue
        if column_class == "categorical":
            if pd.api.types.is_numeric_dtype(df_anon_full[column]):
                # Fill NaN with -1 for numeric columns
                df_anon_full[column] = df_anon_full[column].fillna(num_filler)
            else:
                # Handle other dtypes (e.g., categorical, datetime) if needed
                df_anon_full[column] = df_anon_full[column].fillna(str_filler)  # Default to "missing" for non-numeric types
        elif column_class == "continuous":
            # Fill in the NaN values with the mean of the column
            df_anon_full[column] = df_anon_full[column].fillna(df_anon_full[column].mean())
    
    return df_anon_full

def remove_rows_with_filled_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(df[column]):
        df = df[df[column] != num_filler]
    else:
        df = df[df[column] != str_filler]
    return df.reset_index(drop=True)
