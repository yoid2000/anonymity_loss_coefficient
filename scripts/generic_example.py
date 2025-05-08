import pandas as pd
import numpy as np
from typing import List, Any, Tuple
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
                      known_columns: List[str]) -> Tuple[Any, float, bool]:
    # Find all rows in df_atk that match the known_columns in row
    single_row = row.iloc[0]
    matching_rows = df_atk[df_atk[known_columns].eq(single_row[known_columns].values).all(axis=1)]
    if matching_rows.empty:
        encoded_predicted_value = None
        fraction = 0.0
    else:
        mode_result = matching_rows[secret_column].mode()
        if not mode_result.empty:
            encoded_predicted_value = mode_result.iloc[0]
            mode_count = (matching_rows[secret_column] == encoded_predicted_value).sum()
        else:
            encoded_predicted_value = None
            mode_count = 0
        fraction = mode_count / len(matching_rows)
    abstain = False
    if fraction < 0.0001:
        abstain = True
    return encoded_predicted_value, fraction, abstain


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


def cb() -> str:
    print("```")

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(42)

print("## Example of using the ALCManager class to build attacks.\n")

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
print("\nThe `ALCManager` class is used for all operations. It prepares the data, runs the baseline model, holds the various predictions, computes the ALC measures, and writes the results to files.\n" \
"\nTo prepare the data, it removes NaN rows, discretizes continuous variables, and encodes non-integer columns as integers. Note in particular that, unless the optional parameter `discertize_in_place` is set to True, it creates a new column for each discretized column, given the name `colname__discretized`. The original column is also kept. The discretized column should be used for the secret column, while the original column should be used for the known column. The `flush` parameter (default `False`) tells the `ALCManager` to remove all previously recorded attacks. If set to `False`, the `ALCManager` will not repeat any attacks already run.")
print('''\n`alcm = ALCManager(df_original, syn_data, results_path = "generic_example_files", attack_name = "Example Attacks", flush = True)`''')
alcm = ALCManager(df_original, syn_data, results_path = "generic_example_files", attack_name = "Example Attacks", flush = True, random_state=42)
print("\nWe see for instance that the text column 't1' has been encoded as integers, and two discretized columns have been created from the continuous columns:")
print("\n`alcm.df.orig_all.head()`")
cb()
print(alcm.df.orig_all.head())
cb()

print("\nNote in particular that the df_original and syn_data dataframes are not used once the ALCManager object has been created. All subsequent operations are made on the processed dataframes in the ALCManager object (`alcm.df.orig`, `alcm.df.cntl`, and `alcm.df.anon`).")


print("\nNow lets run the attacks. An attack consists of a set of predictions on the value of a categorical column (the 'secret' column), assuming knowledge of the value of one or more other columns (the 'known columns'). We make two kinds of predictions, attack predictions and baseline predictions. An attack prediction is made on a row taken from the control data over the anonymized data. A baseline prediction is made from a row taken from the control data over the original data. Note that, since the control row is not part of the original data, the baseline prediction is privacy neutral.")

print("\nTo keep this example simple, the attack itself is also naively simple. We combine the preprocessed anonymized dataframes into a single dataframe. We find the rows in the combined anonymized dataset whose values match the known columns of the attack row, if any, and predict that value that is most common among these rows. We then select the majority value of this set of values as our attack prediction. We also compute a 'confidence' associated with the prediction. In this case, our confidence will be the fraction of rows of the matching rows that contain the predicted value.")

# df_anon is our combined anonymized dataset (encoded)
df_anon = pd.concat(alcm.df.anon, ignore_index=True)

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

print("\nRun the predictions loop. There is the question of how many predictions to make in order to get a statistically significant result. To avoid doing more work than necessary, the ALCManager calculates confidence intervals over the set of predictions made so far, and stops when confidence intervals are high enough.")

print("\nThe predictions loop runs from the ALCManager `predictor()` class. This is a generator that feeds attack rows one by one. Inside the loop, we make our prediction (here `attack_prediction()`) and then call either the ACLManager `prediction()` or `abstention()` method to register our prediction. The `abstention()` method is used when the quality of the prediction is likely quite poor. Under the hood, it causes the baseline prediction to be used as the attack prediction. The `predictor()` generator does the rest, including computing the baseline precision and deciding when to quit.")

print('''
```
for atk_row, _, _ in alcm.predictor(known_columns, secret_column):
    encoded_predicted_value, confidence, abstain = attack_prediction(alcm, df_anon, atk_row, secret_column, known_columns)
    if abstain:
        alcm.abstention()
    else:
        alcm.prediction(encoded_predicted_value, confidence)
```

Note that we are ignoring two values returned by the predictor(). The two parameters are `encoded_true_value` and `decoded_true_value`. These are the values used internally by the ACLManager class to determine if the prediction is True of False. They are provided as a convenience, but otherwise aren't needed by the attacking code.
''')

for atk_row, _, _ in alcm.predictor(known_columns, secret_column):
    encoded_predicted_value, confidence, abstain = attack_prediction(alcm, df_anon, atk_row, secret_column, known_columns)
    if abstain:
        alcm.abstention()
    else:
        alcm.prediction(encoded_predicted_value, confidence)

print("\nThat's really all there is to it! There are a few ways in which we can now look at the results of the attack.")

print("\nAfter the predictions loop, we can get a dataframe listing every prediction using the `prediction_results()` method. Here is an example of a row for an individual attack prediction:")
print('''
```
df_results = alcm.prediction_results()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
```
''')
df_results = alcm.prediction_results()
cb()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
cb()

unique_levels = df_results[df_results['predict_type'] == 'attack']['attack_confidence'].unique()

print(f"\nNote that attack_confidence is 1.0. For this particular type of attack, this means that all of the rows that matched the known columns agreed on the predicted value. Different predictions, however, may have different confidence levels. In this case, we see that, among the predictions, there are {len(unique_levels)} unique confidence levels:")

print("\n`df_results[df_results['predict_type'] == 'attack']['attack_confidence'].unique()`")
cb()
print(df_results[df_results['predict_type'] == 'attack']['attack_confidence'].unique())
cb()

print("\nThe method `results()` groups the individual attack predictions by secret column and known columns, and computes a variety of scores including precision, recall, and ALC. When there are multiple confidence levels, the PredictionResults class computes the ALC for different recall values starting with only the highest confidence predictions (low recall), and working through lower confidence predictions.")
print("\n`df_per_comb_results = alcm.results(known_columns=known_columns, secret_column=secret_column)`")
df_per_comb_results = alcm.results(known_columns=known_columns, secret_column=secret_column)

print("\nLet's look at the precision, recall, and ALC scores:")
print("\n`df_per_comb_results[['paired', 'base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']]`")
cb()
print(df_per_comb_results[['paired', 'base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']])
cb()

print("\nAs it so happens, there is no correlation between 't1' and 'i2' or 'f1'. As a result, the baseline precision is always quite low. By contrast, because our anonymity is weak, attack precision is high uniformly high. The fact that attack precision is greater than baseline precision leads to high ALC scores, showing that anonymity is indeed weak.")

print("\nThe `paired` column indicates whether the ALC score is generated from a pair of closely-matched recall values for attack and baseline. If `False`, then the ALC score is generated from the best attack Privacy-Recall Coefficient (PRC) and the best baseline PRC regardless of recall. This represents the most appropriate ALC score (though in general not the highest ALC score).")

print("\nLet's run a second attack, here assuming that the attacker knows the value of column 'i1' and wants to predict the value of column 't1'.")
known_columns = ['i1']
secret_column = alcm.get_discretized_column('t1')
for atk_row, _, _ in alcm.predictor(known_columns, secret_column):
    encoded_predicted_value, confidence, abstain = attack_prediction(alcm, df_anon, atk_row, secret_column, known_columns)
    if abstain:
        alcm.abstention()
    else:
        alcm.prediction(encoded_predicted_value, confidence)

print("\nLet's look at the precision, recall, and ALC scores for the second attack:")
df_per_comb_results = alcm.results(known_columns=known_columns, secret_column=secret_column)
cb()
print(df_per_comb_results[['paired', 'base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']])
cb()

print("\nHere we see quite a different story. Since 'i1' and 't1' are perfectly correlated, all baseline predictions are correct. (The reason `base_prec` is not perfect is because of how we compute precision: as the midpoint of the confidence interval rather than the actual predictions. The actual sampled precision, however, is also computed and can be viewed.) As it so happens, all attack predictions are also correct. (In this case, `attack_prec` is lower for lower recall values only because the confidence bounds are larger.). Because the attack precision is no better than the base precision, the ALC is at best 0.0, meaning no loss of anonymity.")

print("\nBesides being able to obtain the results as dataframes, the method `summarize_results()` writes the results to CSV files and can generate plots as well:")

print('''\n`alcm.summarize_results()`''')
alcm.summarize_results()

print('''
This produces the following files (which can be viewed in the `generic_example_files` directory):
* summary_raw.parquet: All of the predictions
* summary_secret_known.csv: The precision, recall, and ALC scores for predictions grouped by secret column and known columns
* summary.txt: A descriptive summary of the results

''')