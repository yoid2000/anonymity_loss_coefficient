import pandas as pd
import numpy as np
from collections import Counter
from typing import Any, Tuple, Dict, List, Union

def find_best_matches(
    anon: list[pd.DataFrame],
    df_query: pd.DataFrame,
    secret_column: str,
    column_classifications: Dict[str, str],
    match_method: str = "gower",
    max_num_anon_datasets: int = 1,
) -> Tuple[float, list[Any]]:
    """
    Finds the best matches in a list of DataFrames based on Gower distance.
    Returns the minimum adjusted Gower distance and the values of the secret_column for the best matches.

    Currently supports two match methods: "gower" and "mod_gower". The latter takes into account the
    frequency of the categorical value in the dataset. Our experiments suggest that "mod_gower" less
    effective that "gower", so we use "gower" by default.

    When there are multiple DataFrames in anon, especially cases where not all of the multiple dataframes
    have the same columns, we may want to use the matches from multiple dataframes. The parameter
    max_num_anon_datasets controls how many DataFrames in anon we will use to find the best matches.

    Parameters:
    - anon: list of pd.DataFrame, the DataFrames to search for matches.
    - df_query: pd.DataFrame, the DataFrame with a single row to match against.
    - secret_column: str, the name of the column containing the secret values. 
    - column_classifications: dict, a dictionary mapping column names to "categorical" or "continuous".
    - match_method: str, the method to use for matching ("gower" or "mod_gower").
    - max_num_anon_datasets: int, the maximum number of DataFrames in anon to include in the set of matches.
    """
    results = []
    known_columns = list(df_query.columns)

    # Make a first pass through anon to prepare min_max values for each continuous column
    min_max = {}
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
        min_max = _get_min_max([df_candidates, df_query], continuous_cols, min_max)

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

        # Prepare column_classifications for only the columns present
        filtered_column_classifications = {col: column_classifications[col] for col in matching_columns if col in column_classifications}

        df_query_sub = df_query[matching_columns]

        idx, gower_distance = _find_best_matches_one(
            df_candidates,
            df_query_sub,
            filtered_column_classifications,
            min_max,
            match_method=match_method,
        )

        # Adjust for missing columns
        all_categorical_cols = [col for col in df_query.columns if column_classifications.get(col) == "categorical"]
        all_continuous_cols = [col for col in df_query.columns if column_classifications.get(col) == "continuous"]
        mismatched_columns = [col for col in df_query.columns if col not in matching_columns]
        mismatched_cat_columns = [col for col in mismatched_columns if col in all_categorical_cols]
        mismatched_con_columns = [col for col in mismatched_columns if col in all_continuous_cols]
        mismatched_con_weight = len(mismatched_con_columns) * 0.5
        mismatched_cat_weight = len(mismatched_cat_columns) * 0.5
        if match_method == "gower":
            mismatched_con_weight = len(mismatched_con_columns) * 0.5
            mismatched_cat_weight = len(mismatched_cat_columns) * 0.5
        total_mismatched_weight = mismatched_con_weight + mismatched_cat_weight
        num_matched_columns = len(matching_columns)
        adjusted_gower_distance = ((gower_distance * num_matched_columns) + total_mismatched_weight) / len(df_query.columns)

        # Get the secret values for the best matches
        secret_values = df_candidates.loc[idx, secret_column].tolist()

        results.append((adjusted_gower_distance, secret_values))

    if not results:
        return 1.0, []

    # Sort by adjusted_gower_distance
    results.sort(key=lambda x: x[0])
    min_distance = results[0][0]

    # Collect all sets with the same min_distance
    min_sets = [r for r in results if r[0] == min_distance]

    if len(min_sets) >= max_num_anon_datasets:
        selected = min_sets
    else:
        # Add additional sets (with higher distances) to reach max_num_anon_datasets
        selected = min_sets
        for r in results[len(min_sets):]:
            selected.append(r)
            if len(selected) >= max_num_anon_datasets:
                break

    avg_adjusted_gower_distance = sum(r[0] for r in selected) / len(selected)
    all_secret_values = []
    for _, secrets in selected:
        all_secret_values.extend(secrets)

    return avg_adjusted_gower_distance, all_secret_values

def find_best_matches_one(df_candidates: pd.DataFrame,
                      df_query: pd.DataFrame,
                      column_classifications: Dict[str, str],
                      ) -> Tuple[pd.Index, float]:
    continuous_cols = [col for col, cls in column_classifications.items() if cls == "continuous"]
    min_max = _get_min_max([df_candidates, df_query], continuous_cols, {})
    return _find_best_matches_one(
        df_candidates, df_query, column_classifications, min_max 
    )

def _find_best_matches_one(df_candidates: pd.DataFrame,
                      df_query: pd.DataFrame,
                      column_classifications: Dict[str, str],
                      min_max: Dict[str, Tuple[float, float]],
                      match_method: str = "gower",
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
        continuous_distances = pd.DataFrame(np.abs(normalized_candidates - normalized_query))
    else:
        continuous_distances = pd.DataFrame(np.zeros((len(df_candidates), 0)), index=df_candidates.index)

    if match_method == "gower":
        return _gower_distance(df_candidates, query_row, continuous_cols, categorical_cols, continuous_distances)
    elif match_method == "mod_gower":
        return _mod_gower_distance(df_candidates, query_row, continuous_cols, categorical_cols, continuous_distances)

def _mod_gower_distance(df_candidates: pd.DataFrame,
                 query_row: pd.Series,
                 continuous_cols: list[str],
                 categorical_cols: list[str],
                 continuous_distances: pd.DataFrame,
                ) -> Tuple[pd.Index, float]:
    # Continuous distances: same as Gower
    continuous_distances_sum = continuous_distances.sum(axis=1)

    # Categorical distances: sensitive to the frequency of the value in the query
    if categorical_cols:
        categorical_distances = pd.DataFrame(index=df_candidates.index, columns=categorical_cols, dtype=float)
        for col in categorical_cols:
            query_val = query_row[col]
            freq = (df_candidates[col] == query_val).sum()
            n = len(df_candidates)
            match_dist = freq / (2 * n) if n > 0 else 0.0
            mismatch_dist = 1 - ((n - freq) / (2 * n)) if n > 0 else 1.0
            categorical_distances[col] = np.where(
                df_candidates[col] == query_val,
                match_dist,
                mismatch_dist
            )
        categorical_distances_sum = categorical_distances.sum(axis=1)
    else:
        categorical_distances_sum = pd.Series(0, index=df_candidates.index)

    total_features = len(continuous_cols) + len(categorical_cols)
    distances = (continuous_distances_sum + categorical_distances_sum) / total_features

    min_distance = distances.min()
    idx = df_candidates.index[distances == min_distance]
    return pd.Index(idx), round(float(min_distance), 3)

def _gower_distance(df_candidates: pd.DataFrame,
                    query_row: pd.Series,
                    continuous_cols: list[str],
                    categorical_cols: list[str],
                    continuous_distances: pd.DataFrame,
                   ) -> Tuple[pd.Index, float]:
        continuous_distances_sum = continuous_distances.sum(axis=1)

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

def _get_min_max(df_list: list, columns: list, min_max: dict) -> dict:
    for col in columns:
        if col in min_max:
            min_val = min_max[col][0]
            max_val = min_max[col][1]
        else:
            min_val, max_val = None, None
        for df in df_list:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                min_val = col_min if min_val is None else min(min_val, col_min)
                max_val = col_max if max_val is None else max(max_val, col_max)
        min_max[col] = (min_val, max_val)
    return min_max

