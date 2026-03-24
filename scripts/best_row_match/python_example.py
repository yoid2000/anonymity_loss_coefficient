import pprint

import numpy as np
import pandas as pd

from anonymity_loss_coefficient import brm_attack_simple, BrmAttack, prediction_results, results, make_text_summary


def make_original_df(num_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    print(f"Making random original dataframe with {num_rows} rows and 4 columns (1 text, 1 int, and 2 float)")
    rng = np.random.default_rng(random_state)
    text_values = [f"txt_{i}" for i in range(10)]
    int_values = list(range(10))
    return pd.DataFrame(
        {
            "text_col": rng.choice(text_values, size=num_rows),
            "int_col": rng.choice(int_values, size=num_rows),
            "float_col_1": rng.uniform(0.0, 1.0, size=num_rows),
            "float_col_2": rng.uniform(0.0, 1.0, size=num_rows),
        }
    )


def swap_random_values_per_column(
    df: pd.DataFrame, swaps_per_column: int = 250, random_state: int = 99
) -> pd.DataFrame:
    print(f"Making anon dataframe by swapping {swaps_per_column} random values per column in dataframe with {len(df)} rows")
    rng = np.random.default_rng(random_state)
    out = df.copy()
    n_rows = len(out)
    if swaps_per_column <= 1 or swaps_per_column > n_rows:
        raise ValueError(
            f"swaps_per_column must be in [2, {n_rows}], got {swaps_per_column}"
        )

    # For each column, pick independent rows and randomly permute values among them.
    for col in out.columns:
        indices = rng.choice(n_rows, size=swaps_per_column, replace=False)
        shuffled = rng.permutation(out.loc[indices, col].to_numpy())
        out.loc[indices, col] = shuffled
    return out


def main() -> None:
    print("Examples of running the best row match attack with Python-only interface (i.e. without saving results into files).")
    original = make_original_df(num_rows=1000, random_state=42)
    anon = swap_random_values_per_column(
        original, swaps_per_column=250, random_state=99
    )

    print("\n######## Attacks with brm_attack_simple() #########\n")

    secret_column = "float_col_1"
    print(f"Running best row match attack using brm_attack_simple() with secret column '{secret_column}'")
    results = brm_attack_simple(original, anon, secret_column)

    pp = pprint.PrettyPrinter(indent=4)
    print("Miscellaneous results:")
    pp.pprint(results['misc'])
    print("Baseline results:")
    pp.pprint(results['baseline'])
    print("Attack results:")
    pp.pprint(results['attack'])
    print(f"ALC: {results['alc']:.4f}")

    print("\n######## Attacks with BrmAttack class #########\n")
    brm = BrmAttack(df_original=original, anon=anon)
    secret_column1 = "float_col_1"
    known_columns1 = ["text_col", "int_col"]
    print(f"First run one attack on secret column {secret_column1} with known columns {known_columns1}")
    brm.run_one_attack(secret_column=secret_column1, known_columns=known_columns1)
    secret_column2 = "float_col_1"
    known_columns2 = ["text_col", "float_col_2"]
    print(f"Then run one attack on secret column {secret_column2} with known columns {known_columns2}")
    brm.run_one_attack(secret_column=secret_column2, known_columns=known_columns2)
    known_columns3 = ["text_col", "float_col_1"]
    secret_column3 = "float_col_2"
    print(f"Finally, run one attack on secret column {secret_column3} with known columns {known_columns3}")
    brm.run_one_attack(secret_column=secret_column3, known_columns=known_columns3)
    print(f"We can examine the individual attacks.")
    df_attacks = brm.alcm.prediction_results()
    print(f"There are {len(df_attacks)} attack predictions recorded.")
    print("Here is an example individual attack:")
    for column in df_attacks.columns:
        print(f"    {column}: {df_attacks.iloc[0][column]}")
    print("We can examine all the attack measures:")
    df_attack_measures = brm.alcm.results()
    print(f"There are {len(df_attack_measures)} attack measures (one per attack)")
    print("Here are the most important attack measures for the first attack:")
    for column in ["secret_column", "known_columns", "alc", "base_prec", "base_recall", "base_prc", "attack_prec", "attack_recall", "attack_prc"]:
        print(f"    {column}: {df_attack_measures.iloc[0][column]}")

    print("Finally, we can obtain a summary report")
    summary_report = brm.alcm.make_text_summary()
    print(summary_report)

if __name__ == "__main__":
    main()
