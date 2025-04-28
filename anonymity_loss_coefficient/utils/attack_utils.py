import pandas as pd
import random
import math
from typing import List
from itertools import combinations

def get_good_known_column_sets(df: pd.DataFrame,
                            known_columns: List[str],
                            max_sets: int = 1000,
                            unique_rows_threshold: float = 0.45) -> List[List[str]]:
    '''
    Makes a list of known column sets, where at least unique_rows_threshold of the rows
    containing the known columns are unique. Starts with the fewest possible known columns
    and works its way up.
    '''
    # make a copy of df with no more than 10000 rows. Otherwise, the drop_duplicates()
    # operation will be too slow.
    if df.shape[0] > 10000:
        df = df.sample(10000, random_state=1)

    num_unique_rows = df.drop_duplicates().shape[0]
    column_sets = []
    stats = {}
    for col in known_columns:
        stats[col] = 0
    for r in range(1, len(known_columns) + 1):
        inactive = 0
        col_combs = _get_column_combinations(known_columns, r, max_sets)
        for cols in col_combs:
            inactive += 1
            if inactive > 50:
                # Don't continue working on overly small column sets
                break
            num_distinct = df[cols].drop_duplicates().shape[0]
            # We want a given known columns set to have a large number of uniques
            if num_distinct >= (num_unique_rows * unique_rows_threshold):
                inactive = 0
                column_sets.append(cols)
                for col in cols:
                    stats[col] += 1
                if len(column_sets) >= max_sets:
                    return column_sets
    return column_sets

def _get_column_combinations(known_columns: List[str],
                             r: int, max_sets: int) -> List[List[str]]:
    '''
    The idea here is that we will have very few duplicate combinations if we
    make max_sets*10 random combinations of something where there are more than
    100k combinations. 
    '''
    col_combs = []
    return_num = max_sets * 10
    max_combinations = max([100000, return_num])
    num_combs = math.comb(len(known_columns), r)
    if num_combs > max_combinations:
        for _ in range(return_num):
            col_combs.append(random.sample(known_columns, r))
    else:
        # There aren't too many combinations, so just return them all
        col_combs = list(combinations(known_columns, r))
        col_combs = [list(comb) for comb in col_combs]
    random.shuffle(col_combs)
    return col_combs[:return_num]

if __name__ == "__main__":
    # Make a list of characters from a to z
    # and a list of numbers from 0 to 9
    chars = [chr(i) for i in range(97, 123)]
    combs = _get_column_combinations(chars, 6, 1000)
    print(len(combs))
    # count the number of unique combinations in combs
    unique_combs = set()
    for comb in combs:
        unique_combs.add(tuple(comb))
    print(len(unique_combs))