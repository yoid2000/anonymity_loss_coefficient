import pandas as pd
import numpy as np
from typing import List
from anonymity_loss_coefficient import ALCManager
import pprint
import warnings
#warnings.filterwarnings('error')

def make_data(num_rows: int) -> pd.DataFrame:
    t1_values = np.random.choice(['a', 'b', 'c', 'd'], size=num_rows)
    t1_to_i1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    # i1 and t1 are perfectly correlated
    i1_values = np.vectorize(t1_to_i1.get)(t1_values)
    df = pd.DataFrame({
        't1': t1_values,
        'i1': i1_values,
        'i2': np.random.randint(100, 200, size=num_rows),
        'f1': np.random.normal(loc=10, scale=2, size=num_rows)
    })
    return df


def attack_prediction(alcm: ALCManager,
                      df_atk: pd.DataFrame,
                      row: pd.DataFrame,
                      secret_column: str,
                      known_columns: List[str]) -> None:
    true_value = row[secret_column].iloc[0]
    # Find all rows in df_atk that match the known_columns in row
    single_row = row.iloc[0]
    matching_rows = df_atk[df_atk[known_columns].eq(single_row[known_columns].values).all(axis=1)]
    if matching_rows.empty:
        predicted_value = None
        fraction = 0.0
    else:
        mode_result = matching_rows[secret_column].mode()
        if not mode_result.empty:
            predicted_value = mode_result.iloc[0]
            mode_count = (matching_rows[secret_column] == predicted_value).sum()
        else:
            predicted_value = None
            mode_count = 0
        fraction = mode_count / len(matching_rows)
        predicted_value = alcm.decode_value(secret_column, predicted_value)

    alcm.add_attack_result(known_columns = known_columns,
                                secret_col = secret_column,
                                predicted_value = predicted_value,
                                true_value = alcm.decode_value(secret_column, true_value),
                                attack_confidence = fraction,
                                )


def anonymize_data(df: pd.DataFrame) -> pd.DataFrame:
    df_anonymized = df.copy()
    num_rows = len(df)
    num_swaps = int(num_rows * 0.05)
    
    for column in df.columns:
        if num_swaps > 0:
            # Randomly select indices to swap
            swap_indices = np.random.choice(num_rows, size=num_swaps * 2, replace=False)
            swap_indices_1 = swap_indices[:num_swaps]
            swap_indices_2 = swap_indices[num_swaps:]
            
            # Swap the values
            df_anonymized.loc[swap_indices_1, column], df_anonymized.loc[swap_indices_2, column] = \
            df_anonymized.loc[swap_indices_2, column].values, df_anonymized.loc[swap_indices_1, column].values
    
    return df_anonymized

def base_prediction(alcm: ALCManager,
                    row: pd.DataFrame,
                    secret_col: str,
                    known_columns: List[str]) -> None:
    # get the prediction for the row
    predicted_value, proba = alcm.predict(row[known_columns])
    true_value = row[secret_col].iloc[0]
    # We decode the encoded values for later inspection of the attack results
    alcm.add_base_result(known_columns = known_columns,
                             secret_col = secret_col,
                             predicted_value = alcm.decode_value(secret_col, predicted_value),
                             true_value = alcm.decode_value(secret_col, true_value),
                             base_confidence = proba,
                            )

def run_predictions_loop(alcm: ALCManager,
                         secret_col: str, 
                         known_columns: List[str],
                         df_atk_in: pd.DataFrame) -> None:
    # Shuffle df_cntl and df_atk to avoid any bias over multiple attacks
    df_cntl = alcm.df.cntl.sample(frac=1).reset_index(drop=True)
    df_atk = df_atk_in.sample(frac=1).reset_index(drop=True)
    for i in range(len(df_cntl)):
        # Get one base and attack measure at a time, and continue until we have
        # enough confidence in the results
        atk_row_df = df_cntl.iloc[[i]]
        base_prediction(alcm, atk_row_df, secret_col, known_columns)

        attack_prediction(alcm, df_atk, atk_row_df, secret_col, known_columns)

        halt_ok, info, reason = alcm.ok_to_halt()
        if halt_ok:
            print(f'\nOk to halt after {i} attacks with ALC {info['alc']:.2f} and reason: "{reason}"')
            return

def cb() -> str:
    print("```")

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(42)

print("Assume that `df_original` is the original raw data that we want to measure")
df_original = make_data(10000)
print(f"The original data has {len(df_original)} rows. Here are the first 5 rows:")
print("\n`df_original.head()`")
cb()
print(df_original.head())
cb()
print(f"and here are some summary statistics:")
print("\n`df_original.describe()`")
cb()
print(df_original.describe())
cb()

print('Assume that 4 anonymized datasets have been generated from `df_original`. Of course, this also works with only a single anonymized dataset. (The "anonymization" here is nothing more than swapping a small fraction of the values.)')
syn_data = [anonymize_data(df_original) for _ in range(4)]

print("\nAt this point, we have prepared the dataframes needed for the ALC measures.")
print("\nThe ALCManager class is used for all operations. It prepares the data, runs the baseline model, holds the various predictions, computes the ALC measures, and writes the results to files.\n" \
"\nTo prepare the data, it removes NaN rows, discretizes continuous variables, and encodes non-integer columns as integers. Note in particular that, unless the optional parameter `discertize_in_place` is set to True, it creates a new column for each discretized column, given the name `colname__discretized`. The original column is also kept. The discretized column should be used for the secret column, while the original column should be used for the known column.")
print('''\n`alcm = ALCManager(df_original, syn_data)`''')
alcm = ALCManager(df_original, syn_data)
print("\nWe see for instance that the text column 't1' has been encoded as integers, and two discretized columns have been created from the continuous columns:")
print("\n`alcm.df.orig_all.head()`")
cb()
print(alcm.df.orig_all.head())
cb()

print("\nNote in particular that the df_original and syn_data dataframes are not used once the ALCManager object has been created. All subsequent operations are made on the processed dataframes in the ALCManager object (`alcm.df.orig`, `alcm.df.cntl`, and `alcm.df.syn_list`).")


print("\nNow lets run the attacks. An attack consists of a set of predictions on the value of a categorical column (the 'secret' column), assuming knowledge of the value of one or more other columns (the 'known columns'). We make two kinds of predictions, attack predictions and baseline predictions. An attack prediction is made on a row taken from the control data over the anonymized data. A baseline prediction is made from a row taken from the control data over the original data. Note that, since the control row is not part of the original data, the baseline prediction is privacy neutral.")

print("\nTo keep this example simple, the attack itself is also naively simple. We combine the preprocessed anonymized dataframes into a single dataframe. We find the rows in the combined anonymized dataset whose values match the known columns of the attack row, if any, and predict that value that is most common among these rows. We then select the majority value of this set of values as our attack prediction. We also compute a 'confidence' associated with the prediction. In this case, our confidence will be the fraction of rows of the matching rows that contain the predicted value.")

# df_anon is our combined anonymized dataset
df_anon = pd.concat(alcm.df.syn_list, ignore_index=True)

print("\nIn a first attack, let's assume that the attacker knows the values of column 'i2' and 'f1', and wants to predict the value of column 't1'.")
print('''
```
known_columns = ['i2', 'f1']
secret_column = alcm.get_discretized_column('t1')
```
''')
known_columns = ['i2', 'f1']
secret_column = alcm.get_discretized_column('t1')

print("\nNote the use of the `get_discretized_column` method.  This produces the column name of the discretized column, if any. If none, it returns the original column name (which is the case here).")

print("\nFor the baseline predictions, we need to make a model from the original data.")

print("\n`alcm.init_cntl_and_build_model(known_columns, secret_column)`")
alcm.init_cntl_and_build_model(known_columns, secret_column)

print("\nRun the predictions loop. There is the question of how many predictions to make in order to get a statistically significant result. To avoid doing more work than necessary, the ALCManager can calculate confidence intervals over the set of predictions made so far.")

run_predictions_loop(alcm, secret_column, known_columns, df_anon)

print("\nAfter the predictions loop, we can get a dataframe listing every prediction. Here is an example of a row for an attack prediction:")
print('''
```
df_results = alcm.get_results_df()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
```
''')
df_results = alcm.get_results_df()
cb()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
cb()

print("\nNote that attack_confidence is 1.0. For this particular type of attack, this means that all of the rows that matched the known columns agreed on the predicted value. Different predictions, however, may have different confidence levels. In this case, we see that, among the predictions, there are 5 unique confidence levels:")

print("\n`df_results['attack_confidence'].unique())`")
cb()
print(df_results['attack_confidence'].unique())
cb()

print("\nThe PredictionResults class can compute the precision, recall, and ALC for each combination of known columns and secret column. When there are multiple confidence levels, the PredictionResults class computes the ALC for different recall values starting with only the highest confidence predictions (low recall), and working through lower confidence predictions.")
print("\n`df_per_comb_results = alcm.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)`")
df_per_comb_results = alcm.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)

print("\nIn total, here are the columns produced by the alc_per_secret_and_known_df method:")
print("\n`df_per_comb_results.columns`")
cb()
print(df_per_comb_results.columns)
cb()

print("\nLet's look at the precision, recall, and ALC scores:")
print("\n`df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']]`")
cb()
print(df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']])
cb()

print("\nAs it so happens, there is no correlation between 't1' and 'i2' or 'f1'. As a result, the baseline precision is always quite low, and different recall values don't really help. By contrast, because our anonymity is weak, attack precision is high, and improves as recall is lower. This leads to a high ALC score, the highest at recall=0.75, showing that anonymity is indeed weak.")

print("\nLet's run a second attack, here assuming that the attacker knows the value of column 'i1' and wants to predict the value of column 't1'.")
known_columns = ['i1']
secret_column = alcm.get_discretized_column('t1')
alcm.init_cntl_and_build_model(known_columns, secret_column)
run_predictions_loop(alcm, secret_column, known_columns, df_anon)

print("\nLet's look at the precision, recall, and ALC scores for the second attack:")
df_per_comb_results = alcm.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)
cb()
print(df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']])
cb()

print("\nHere we see quite a different story. Since 'i1' and 't1' are perfectly correlated, the baseline precision is always 1.0. Because of the anonymization, however, the attack precision, while pretty high (around 0.9), is not perfect. Because the attack precision is always less than the baseline precision, the ALC score is always negative, which translates to no anonymity loss whatsoever.")

print("\nFinally, we can get a summary of the ALC scores for all attacks. This is placed in a directory with the name originally conveyed to the PredictionResults class `results_path` variable. In our case, 'example'.")

print('''\n`alcm.summarize_results(results_path = "example", attack_name = "Example Attacks")`''')
alcm.summarize_results(results_path = "example", attack_name = "Example Attacks")

print('''
This produces the following files:
* summary_raw.csv: All of the predictions
* summary_secret.csv: The precision, recall, and ALC scores for predictions grouped by secret column
* summary_secret_known.csv: The precision, recall, and ALC scores for predictions grouped by secret column and known columns
* summary.txt: A descriptive summary of the results

Note finally that, if there are enough attacks to warrant it, `alcm.summarize_results()` generates several plots as well.
''')