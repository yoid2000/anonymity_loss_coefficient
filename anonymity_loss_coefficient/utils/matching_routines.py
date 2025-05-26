import pandas as pd
import numpy as np
from collections import Counter
from typing import Any, Tuple, Dict, List, Union

# Yes, that's right, we are banking on the real data not having these values  :(
num_filler = -123456789
str_filler = "no__match_xyz"

def find_best_matches(
    anon: list[pd.DataFrame],
    df_query: pd.DataFrame,
    secret_column: str,
    column_classifications: Dict[str, str],
) -> Tuple[float, list[Any]]:
    """
    Finds the best matches in a list of DataFrames based on Gower distance.
    Returns the minimum adjusted Gower distance and the values of the secret_column for the best matches.
    """
    best_match_values = []
    min_gower_distance = float("inf")
    known_columns = list(df_query.columns)

    for df_candidates in anon:
        # Check if df_candidates has the secret column and at least one known column
        if secret_column not in df_candidates.columns:
            continue
        if not any(col in df_candidates.columns for col in known_columns):
            continue

        # Only use columns present in both df_candidates and df_query for matching
        matching_columns = [col for col in known_columns if col in df_candidates.columns and col in df_query.columns]
        if not matching_columns:
            continue

        # Prepare min_max for continuous columns
        continuous_cols = [col for col in matching_columns if column_classifications.get(col) == "continuous"]
        min_max = _get_min_max([df_candidates, df_query], continuous_cols)

        # Prepare column_classifications for only the columns present
        filtered_column_classifications = {col: column_classifications[col] for col in matching_columns if col in column_classifications}

        # Subset dataframes to matching columns
        df_candidates_sub = df_candidates[matching_columns]
        df_query_sub = df_query[matching_columns]

        idx, gower_distance = _find_best_matches_one(
            df_candidates_sub,
            df_query_sub,
            filtered_column_classifications,
            min_max
        )

        # Count mismatched columns
        num_mismatched_columns = len([col for col in df_query.columns if col not in df_candidates.columns])
        # Treat each of the columns not in df_query as gower distance of 1
        adjusted_gower_distance = (gower_distance + num_mismatched_columns) / len(df_query.columns)

        # Get the secret values for the best matches
        secret_values = df_candidates.loc[idx, secret_column].tolist()

        if adjusted_gower_distance < min_gower_distance:
            min_gower_distance = adjusted_gower_distance
            best_match_values = secret_values
        elif adjusted_gower_distance == min_gower_distance:
            best_match_values.extend(secret_values)

    return min_gower_distance, best_match_values

def find_best_matches_one(df_candidates: pd.DataFrame,
                      df_query: pd.DataFrame,
                      column_classifications: Dict[str, str],
                      ) -> Tuple[pd.Index, float]:
    continuous_cols = [col for col, cls in column_classifications.items() if cls == "continuous"]
    min_max = _get_min_max([df_candidates, df_query], continuous_cols)
    return _find_best_matches_one(
        df_candidates, df_query, column_classifications, min_max 
    )

def _find_best_matches_one(df_candidates: pd.DataFrame,
                      df_query: pd.DataFrame,
                      column_classifications: Dict[str, str],
                      min_max: Dict[str, Tuple[float, float]],
                      ) -> Tuple[pd.Index, float]:
    """
    Compute Gower distance

    Parameters:
    - df_candidates: pd.DataFrame, the DataFrame with multiple rows.
    - df_query: pd.DataFrame, the DataFrame with a single row (same columns as df_candidates).
    - column_classifications: dict, a dictionary mapping column names to "categorical" or "continuous".
    - columns: list, optional, the list of columns to use for the Gower distance calculation.

    Returns:
    - Tuple:
        - pd.Index: Indices of rows in df_candidates that share the minimum Gower distance.
        - float: The minimum Gower distance.
    """
    if len(df_query) != 1:
        raise ValueError("df_query must contain exactly one row.")

    if column_classifications is None:
        raise ValueError("column_classifications must be provided.")

    df_candidates = df_candidates[list(df_query.columns)]

    # Ensure df_query is a single row
    query_row = df_query.iloc[0]

    # Separate continuous and categorical columns
    continuous_cols = [col for col, cls in column_classifications.items() if cls == "continuous"]
    continuous_cols = [col for col in continuous_cols if col in df_candidates.columns]
    categorical_cols = [col for col, cls in column_classifications.items() if cls == "categorical"]
    categorical_cols = [col for col in categorical_cols if col in df_candidates.columns]

    # Precompute min and max for continuous columns
    if continuous_cols:
        col_min = pd.Series({col: min_max[col][0] for col in continuous_cols}, index=continuous_cols)
        col_max = pd.Series({col: min_max[col][1] for col in continuous_cols}, index=continuous_cols)

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

    # Find the minimum Gower distance
    min_distance = distances.min()

    # Find the indices of rows in df_candidates with the minimum Gower distance
    idx = df_candidates.index[distances == min_distance]

    return pd.Index(idx), round(float(min_distance), 3)

def modal_fraction(values: list) -> Tuple[Any, int]:
    """
    Determines the modal value and its count

    Parameters:
    - values: list, the list of values to consider.

    Returns:
    - modal_value: The modal value in the specified column.
    - count: The count of the modal value (as a Python int).
    """
    counter = Counter(values)
    modal_value, count = counter.most_common(1)[0]
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
    return round((1 - gower_distance) * modal_fraction, 3)

def _get_min_max(df_list: list, columns: list) -> dict:
    min_max = {}
    for col in columns:
        min_val, max_val = None, None
        for df in df_list:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                min_val = col_min if min_val is None else min(min_val, col_min)
                max_val = col_max if max_val is None else max(max_val, col_max)
        min_max[col] = (min_val, max_val)
    return min_max

