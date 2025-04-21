import pandas as pd
import random
from typing import List
from itertools import combinations

def get_good_known_column_sets(df: pd.DataFrame,
                            known_columns: List[str],
                            max_sets: int = 1000,
                            unique_rows_threshold: float = 0.45) -> list:
    '''
    Makes a list of known column sets, where at least unique_rows_threshold of the rows
    containing the known columns are unique. Starts with the fewest possible known columns
    and works its way up.
    '''
    num_unique_rows = df.drop_duplicates().shape[0]
    column_sets = []
    stats = {}
    for col in known_columns:
        stats[col] = 0
    for r in range(1, len(known_columns) + 1):
        inactive = 0
        col_combs = list(combinations(known_columns, r))
        random.shuffle(col_combs)
        for cols in col_combs:
            inactive += 1
            if inactive > 50:
                # Don't continue working on overly small column sets
                break
            num_distinct = df[list(cols)].drop_duplicates().shape[0]
            # We want a given known columns set to have a large number of uniques
            if num_distinct >= (num_unique_rows * unique_rows_threshold):
                inactive = 0
                column_sets.append(cols)
                for col in cols:
                    stats[col] += 1
                if len(column_sets) >= max_sets:
                    return column_sets
    return column_sets