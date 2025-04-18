# Best Row Match attack

This attack is designed to work with microdata (aka synthetic data). The attack is an attribute inference attack: the attacker knows from information about a person in the anonymized data, finds a row that "best matches" that person, and the infers unknown attributes.

We use the Anonymity Loss Coefficient to measure anonymity. The ALC compares the attack inference with a baseline inference that uses an ML model to predict the unknown attribute given the known attributes.

A unique feature of our attack is that it can handle the case where multiple synthetic datasets have been released. Essentially, it runs the attack on all of the synthetic datasets, and selects the most common answer as the best answer.

## Scripts

See `scripts/best_row_match/README.md` for details on running the attack on your data.

## Limitations

We currently don't recognize datetime columns as datetime columns.

We only input csv files. Should add parquet to this in the future.

## Interface

`brm_attack.py` contains the `BrmAttack` class:

```
from anonymity_loss_coefficient.attacks import BrmAttack

brm = BrmAttack(df_original=df_original,
                df_synthetic=syn_dfs,
                results_path=results_path,
                )
```

The required parameters are:

* `df_original` is the original dataset.
* `df_synthetic` is a list of one or more anonymized datasets. 
* `results_path` is a path to the directory where the results of the attack should be stored.


`BrmAttack` exposes three methods:

* `brm_attack(secret_col: str, known_columns: List[str]) -> None:` where `secret_col` is the unknown attribute being predicted, and `known_columns` are the columns known to the attacker. `brm_attack()` runs until a statistically significant number of attacks are completed.
* `run_all_columns_attack()` runs the `brm_attack()` for every secret column in the dataset, assuming that all other columns are known.
* `run_auto_attack()` finds sets of known columns for each secret_col that likely to produce a large number of unique rows, and runs attacks over those known columns.
