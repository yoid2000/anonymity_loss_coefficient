import pandas as pd
import pytest
from anonymity_loss_coefficient.utils.matching_routines import (
    find_best_matches_one,
    modal_fraction,
    best_match_confidence,
    find_best_matches
)

@pytest.fixture
def df_candidates():
    return pd.DataFrame({
        'Age': [25, 35, 45, 25, 35],
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female'],
        'Income': [50000, 60000, 55000, 50000, 60000]
    })

@pytest.fixture
def df_query():
    return pd.DataFrame({
        'Age': [30],
        'Gender': ['Male'],
        'Income': [52000]
    })

@pytest.fixture
def column_classifications():
    return {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }

def test_find_best_matches_one_all_columns(df_candidates, df_query, column_classifications):
    idx_all, min_distance_all = find_best_matches_one(
        df_candidates, df_query, column_classifications=column_classifications
    )
    expected_idx_all = pd.Index([0, 3])
    expected_min_distance_all = 0.15
    assert idx_all.equals(expected_idx_all), f"Expected indices {expected_idx_all}, got {idx_all}"
    assert min_distance_all == expected_min_distance_all, f"Expected min distance {expected_min_distance_all}, got {min_distance_all}"

def test_find_best_matches_one_subset_columns(df_candidates, df_query, column_classifications):
    idx_subset, min_distance_subset = find_best_matches_one(
        df_candidates, df_query[['Age', 'Income']], column_classifications=column_classifications,
    )
    expected_idx_subset = pd.Index([0, 3])
    expected_min_distance_subset = 0.225
    assert idx_subset.equals(expected_idx_subset), f"Expected indices {expected_idx_subset}, got {idx_subset}"
    assert min_distance_subset == expected_min_distance_subset, f"Expected min distance {expected_min_distance_subset}, got {min_distance_subset}"

def test_best_match_confidence():
    gower_distance = 0.15
    modal_fraction_value = 0.667
    match_count = 3
    confidence = best_match_confidence(gower_distance, modal_fraction_value, match_count)
    expected_confidence = round((1 - gower_distance) * modal_fraction_value, 3)
    assert confidence == expected_confidence, f"Expected confidence {expected_confidence}, got {confidence}"

def test_get_min_max():
    from anonymity_loss_coefficient.utils.matching_routines import _get_min_max

    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})
    df2 = pd.DataFrame({'A': [4, 5], 'C': [100, 200]})
    df3 = pd.DataFrame({'B': [5, 15], 'C': [300, 400]})

    df_list = [df1, df2, df3]
    columns = ['A', 'B', 'C', 'D']

    result = _get_min_max(df_list, columns, {})

    assert result['A'] == (1, 5)
    assert result['B'] == (5, 30)
    assert result['C'] == (100, 400)
    assert result['D'] == (None, None)

def test_find_best_matches_basic():

    # Create two candidate DataFrames
    df1 = pd.DataFrame({
        'Age': [25, 35, 45],
        'Gender': ['Male', 'Female', 'Male'],
        'Income': [50000, 60000, 55000],
        'Secret': ['A', 'B', 'C']
    })
    df2 = pd.DataFrame({
        'Age': [25, 35],
        'Gender': ['Male', 'Female'],
        'Income': [50000, 60000],
        'Secret': ['D', 'E']
    })

    anon = [df1, df2]

    df_query = pd.DataFrame({
        'Age': [25.1],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_gower_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications
    )

    # approx assertion for floating point comparison
    assert min_gower_distance == pytest.approx(0.002, 0.0001)
    assert set(best_match_values) == {'A', 'D'}

def test_find_best_matches_with_missing_columns():

    # df1 is missing 'Income', df2 is missing 'Gender'
    df1 = pd.DataFrame({
        'Age': [25, 35],
        'Gender': ['Male', 'Female'],
        'Secret': ['A', 'B']
    })
    df2 = pd.DataFrame({
        'Age': [25, 35],
        'Income': [50000, 60000],
        'Secret': ['C', 'D']
    })

    anon = [df1, df2]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_gower_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications
    )

    assert min_gower_distance == pytest.approx(1/6, 0.01)
    assert set(best_match_values) == {'A', 'C'}

def test_find_best_matches_no_match():

    df1 = pd.DataFrame({
        'Age': [35, 45],
        'Gender': ['Female', 'Male'],
        'Income': [60000, 55000],
        'Secret': ['A', 'B']
    })

    anon = [df1]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_gower_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications
    )

    # The closest match is not exact, but should still return the closest row(s)
    assert min_gower_distance > 0
    assert isinstance(best_match_values, list)
    assert len(best_match_values) > 0

def test_find_best_matches_anon_missing_and_extra_columns():

    # df1 is missing 'Income', has extra 'Location'
    # df1 returns gower of 1/3, secret 'A'
    df1 = pd.DataFrame({
        'Age': [25, 35],
        'Gender': ['Male', 'Female'],
        'Location': ['NY', 'CA'],
        'Secret': ['A', 'B']
    })
    # df2 is missing 'Gender', has extra 'Occupation'
    # returns gower of 1/3, secret 'C'
    df2 = pd.DataFrame({
        'Age': [25, 35],
        'Income': [50000, 60000],
        'Occupation': ['Engineer', 'Artist'],
        'Secret': ['C', 'D']
    })
    # df3 has only 'Income' and 'Secret'
    # returns gower of 2/3, secret 'E' (and so is ignored)
    df3 = pd.DataFrame({
        'Income': [50000, 70000],
        'Secret': ['E', 'F']
    })

    anon = [df1, df2, df3]

    # df_query has all three columns
    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_gower_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications
    )

    assert min_gower_distance == pytest.approx(1/6, 0.01)
    assert set(best_match_values) == {'A', 'C'}


def test_modal_fraction_simple():
    values = ['A', 'B', 'A', 'C', 'A', 'B']
    modal, count = modal_fraction(values)
    assert modal == 'A'
    assert count == 3

def test_modal_fraction_numeric():
    values = [1, 2, 2, 3, 2, 1]
    modal, count = modal_fraction(values)
    assert modal == 2
    assert count == 3

def test_modal_fraction_tie():
    # Counter.most_common returns the first encountered in case of tie
    values = ['X', 'Y', 'X', 'Y']
    modal, count = modal_fraction(values)
    assert modal in ('X', 'Y')
    assert count == 2

def test_modal_fraction_single_value():
    values = ['Z']
    modal, count = modal_fraction(values)
    assert modal == 'Z'
    assert count == 1

def test_modal_fraction_empty():
    with pytest.raises(IndexError):
        modal_fraction([])

def test_find_best_matches_mod_gower_basic():

    df1 = pd.DataFrame({
        'Age': [25, 35, 45],
        'Gender': ['Male', 'Female', 'Male'],
        'Income': [50000, 60000, 55000],
        'Secret': ['A', 'B', 'C']
    })
    df2 = pd.DataFrame({
        'Age': [25, 35],
        'Gender': ['Male', 'Female'],
        'Income': [50000, 60000],
        'Secret': ['D', 'E']
    })

    anon = [df1, df2]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications, match_method='mod_gower'
    )

    assert min_distance == pytest.approx(0.0833, 0.01)
    assert set(best_match_values) == {'D'}

def test_find_best_matches_mod_gower_with_missing_columns():

    # df1 is missing 'Income', df2 is missing 'Gender'
    df1 = pd.DataFrame({
        'Age': [25, 35],
        'Gender': ['Male', 'Female'],
        'Secret': ['A', 'B']
    })
    df2 = pd.DataFrame({
        'Age': [25, 35],
        'Income': [50000, 60000],
        'Secret': ['C', 'D']
    })

    anon = [df1, df2]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications, match_method='mod_gower'
    )

    assert min_distance == pytest.approx(1/6, 0.01)
    assert set(best_match_values) == {'C'}

def test_find_best_matches_mod_gower_no_match():

    df1 = pd.DataFrame({
        'Age': [35, 45],
        'Gender': ['Female', 'Male'],
        'Income': [60000, 55000],
        'Secret': ['A', 'B']
    })

    anon = [df1]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications, match_method='mod_gower'
    )

    # No exact matches, but should still return the closest row(s)
    assert min_distance > 0
    assert isinstance(best_match_values, list)
    assert len(best_match_values) > 0

def test_find_best_matches_mod_gower_anon_missing_and_extra_columns():

    # df1 is missing 'Income', has extra 'Location'
    df1 = pd.DataFrame({
        'Age': [25, 35],
        'Gender': ['Male', 'Female'],
        'Location': ['NY', 'CA'],
        'Secret': ['A', 'B']
    })
    # df2 is missing 'Gender', has extra 'Occupation'
    df2 = pd.DataFrame({
        'Age': [25, 35],
        'Income': [50000, 60000],
        'Occupation': ['Engineer', 'Artist'],
        'Secret': ['C', 'D']
    })
    # df3 has only 'Income' and 'Secret'
    df3 = pd.DataFrame({
        'Income': [50000, 70000],
        'Secret': ['E', 'F']
    })

    anon = [df1, df2, df3]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Income': [50000]
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical',
        'Income': 'continuous'
    }
    secret_column = 'Secret'

    min_distance, best_match_values = find_best_matches(
        anon, df_query, secret_column, column_classifications, match_method='mod_gower'
    )

    assert min_distance == pytest.approx(1/6, 0.01)
    assert set(best_match_values) == {'C'}

def test_find_best_matches_multiple_anon_datasets_min_sets():

    df1 = pd.DataFrame({
        'Age': [25, 35],
        'Gender': ['Male', 'Female'],
        'Secret': ['A', 'B']
    })
    df2 = pd.DataFrame({
        'Age': [25, 45],
        'Gender': ['Male', 'Male'],
        'Secret': ['C', 'D']
    })
    anon = [df1, df2]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male']
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical'
    }
    secret_column = 'Secret'

    avg_dist, all_secrets = find_best_matches(
        anon, df_query, secret_column, column_classifications, max_num_anon_datasets=2
    )

    assert all_secrets == ['A', 'C']
    assert avg_dist == 0.0

def test_find_best_matches_multiple_anon_datasets_more_than_max():

    df1 = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Secret': ['A']
    })
    df2 = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Secret': ['B']
    })
    df3 = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Secret': ['C']
    })
    anon = [df1, df2, df3]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male']
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical'
    }
    secret_column = 'Secret'

    avg_dist, all_secrets = find_best_matches(
        anon, df_query, secret_column, column_classifications, max_num_anon_datasets=2
    )

    # All three have the same min distance: Gender freq=1, n=1, match_dist=1/(2*1)=0.5, Age is exact match (0)
    # Distance: (0+0.5)/2=0.25 for all, so all are included
    assert all_secrets == ['A', 'B', 'C']
    assert avg_dist == 0.0

def test_find_best_matches_multiple_anon_datasets_fill_to_max():
    from anonymity_loss_coefficient.utils.matching_routines import find_best_matches

    df1 = pd.DataFrame({
        'Age': [26],
        'Gender': ['Male'],
        'Secret': ['A']
    })
    df2 = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male'],
        'Secret': ['B']
    })
    df3 = pd.DataFrame({
        'Age': [35],
        'Gender': ['Female'],
        'Secret': ['C']
    })
    anon = [df1, df2, df3]

    df_query = pd.DataFrame({
        'Age': [26],
        'Gender': ['Male']
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical'
    }
    secret_column = 'Secret'

    avg_dist, all_secrets = find_best_matches(
        anon, df_query, secret_column, column_classifications, max_num_anon_datasets=2
    )

    assert all_secrets == ['A', 'B']
    assert avg_dist == pytest.approx(0.025, 0.01)

def test_find_best_matches_multiple_anon_datasets_no_match():
    from anonymity_loss_coefficient.utils.matching_routines import find_best_matches

    df1 = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male']
    })
    df2 = pd.DataFrame({
        'Age': [30],
        'Gender': ['Female']
    })
    anon = [df1, df2]

    df_query = pd.DataFrame({
        'Age': [25],
        'Gender': ['Male']
    })

    column_classifications = {
        'Age': 'continuous',
        'Gender': 'categorical'
    }
    secret_column = 'Secret'

    avg_dist, all_secrets = find_best_matches(
        anon, df_query, secret_column, column_classifications, max_num_anon_datasets=2
    )

    assert all_secrets == []
    assert avg_dist == 1.0