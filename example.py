import pandas as pd
import numpy as np
from scipy import stats
from typing import List
from anonymity_loss_coefficient.anonymity_loss_coefficient import AnonymityLossCoefficient, DataFiles, BaselinePredictor, PredictionResults
import pprint
import warnings
warnings.filterwarnings('error')

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


def attack_prediction(adf: DataFiles, pred_res: PredictionResults,
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
        mode_result = stats.mode(matching_rows[secret_column])
        predicted_value = mode_result.mode
        mode_count = mode_result.count
        fraction = mode_count / len(matching_rows)
        predicted_value = adf.decode_value(secret_column, predicted_value)

    pred_res.add_attack_result(known_columns = known_columns,
                                target_col = secret_column,
                                predicted_value = predicted_value,
                                true_value = adf.decode_value(secret_column, true_value),
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

def base_prediction(adf: DataFiles, base_pred: BaselinePredictor, pred_res: PredictionResults,
                    row: pd.DataFrame,
                    secret_col: str,
                    known_columns: List[str]) -> None:
    # get the prediction for the row
    predicted_value, proba = base_pred.predict(row[known_columns])
    true_value = row[secret_col].iloc[0]
    # We decode the encoded values for later inspection of the attack results
    pred_res.add_base_result(known_columns = known_columns,
                             target_col = secret_col,
                             predicted_value = adf.decode_value(secret_col, predicted_value),
                             true_value = adf.decode_value(secret_col, true_value),
                             base_confidence = proba,
                            )

def run_predictions_loop(adf: DataFiles,
                         base_pred: BaselinePredictor,
                         pred_res: PredictionResults,
                         secret_col: str, known_columns: List[str],
                         df_base_in: pd.DataFrame, df_atk_in: pd.DataFrame) -> None:
    # Shuffle df_base and df_atk to avoid any bias over multiple attacks
    df_base = df_base_in.sample(frac=1).reset_index(drop=True)
    df_atk = df_atk_in.sample(frac=1).reset_index(drop=True)
    confidence_interval_tolerance = 0.1
    for i in range(min(len(df_base), len(df_atk))):
        # Get one base and attack measure at a time, and continue until we have
        # enough confidence in the results
        base_row_df = df_base.iloc[[i]]
        base_prediction(adf, base_pred, pred_res, base_row_df, secret_col, known_columns)

        atk_row = df_atk.iloc[[i]]
        attack_prediction(adf, pred_res, df_atk, atk_row, secret_col, known_columns)

        if i >= 50 and i % 10 == 0:
            # Check for confidence after every 50th attack prediction, then every 10 predictions
            ci_info = pred_res.get_ci()
            cii = ci_info['base']
            if cii['ci_high'] - cii['ci_low'] <= confidence_interval_tolerance:
                print(f"Base confidence interval ({round(cii['ci_low'],2)}, {round(cii['ci_high'],2)}) is within tolerance after {i+1} attacks on precision {round(cii['prec'],2)}")
                return
            cii = ci_info['attack']
            pos_pred_count = round(cii['n'] * cii['prec'])
            if cii['ci_high'] - cii['ci_low'] <= confidence_interval_tolerance:
                print(f"Attack confidence interval ({round(cii['ci_low'],2)}, {round(cii['ci_high'],2)}) is within tolerance after {i+1} attacks on precision {round(cii['prec'],2)}")
                return


pp = pprint.PrettyPrinter(indent=4)
np.random.seed(42)

print("Assume that df_initial is the initial raw data that we want to measure")
df_initial = make_data(10000)
print(f"The initial data has {len(df_initial)} rows. Here are the first 5 rows:")
print("`df_initial.head()`")
print(df_initial.head())
print(f"and here are some summary statistics:")
print("`df_initial.describe()`")
print(df_initial.describe())

print('Assume that 4 anonymized datasets have been generated from df_initial. Of course, this also works with only a single anonymized dataset. (The "anonymization" here is nothing more than swapping a small fraction of the values.)')
syn_data = [anonymize_data(df_initial) for _ in range(4)]

print("\nTo make baseline predictions, we need to separate out a control dataset from the initial raw data. Typically 1000 control rows is enough, though if the initial data is small, then the control dataset should be smaller (say 10% or 20% or the data).")
print('''
```
df_control = df_initial.sample(1000)
df_original = df_initial.drop(df_control.index)
```
''')
df_control = df_initial.sample(1000)
df_original = df_initial.drop(df_control.index)
print(f"df_control has {len(df_control)} randomly sampled rows from df_initial.")
print(f"df_original has the remaining {len(df_original)} rows.")

print("\nAt this point, we have prepared the dataframes needed for the ALC measures.")
print("\nThe DataFiles class is used primarily to preprocess the data. It removes NaN rows, and encodes categorical columns as integers.")
print("`adf = DataFiles(df_original, df_control, syn_data)`")
adf = DataFiles(df_original, df_control, syn_data)
print("We see for instance that the text column 't1' has been encoded as integers:")
print("`adf.orig.head()`")
print(adf.orig.head())

print("\n The Datafiles class also labels each column as categorical or continuous, if this labeling was not supplied to the DataFiles class:")
print("`adf.col_types`")
pp.pprint(adf.col_types)

print("\nThe BaselinePredictor class is used to make baseline predictions on categorical columns. Internally it uses df_original and col_types from the DataFiles class.")
print("`base_pred = BaselinePredictor(adf)`")
base_pred = BaselinePredictor(adf)

print("\nThe PredictionResults class is a helper class that stores the results of the predictions, and provides methods that summarize the precision, recall, and ALC scores.")
print('''`pred_res = PredictionResults(attack_name = "Example Attacks")`''')
pred_res = PredictionResults(attack_name = "Example Attacks")

print("\nNow lets run the attacks. An attack consists of a set of predictions on the value of a categorical column (the 'secret' column), assuming knowledge of the value of one or more other columns (the 'known columns'). We make two kinds of predictions, attack predictions and baseline predictions. An attack prediction is made on a row taken from the initial data over the anonymized data. A baseline prediction is made from a row taken from the control data over the original data. Note that, since the control row is not part of the original data, the prediction is privacy neutral.")

print("\nWe want to make the same number of attack and baseline predictions. Let's set aside a some original rows, equal to the number of control rows, for the attack predictions. Note that we need to work only with the data that has been preprocessed by the DataFiles class. The preprocessed original, control, and synthetic dataframes can be accessed as adf.orig, adf.cntl, and adf.syn_list[i] respectively.")
print('''
```
num_control_rows = len(adf.cntl)
df_atk = adf.orig.sample(num_control_rows)
```
''')
num_control_rows = len(adf.cntl)
df_atk = adf.orig.sample(num_control_rows)

print("\nTo keep this example simple, the attack itself is also naively simple. We combine the preprocessed anonymized dataframes into a single dataframe. We find the rows in the combined anonymized dataset whose values match the known columns of the attack row, if any, and predict that value that is most common among these rows. We then select the majority value of this set of values as our attack prediction. We also compute a 'confidence' associated with the prediction. In this case, our confidence will be the fraction of rows of the matching rows that contain the predicted value.")

# df_anon is our combined anonymized dataset
df_anon = pd.concat(adf.syn_list, ignore_index=True)

print("\nIn a first attack, let's assume that the attacker knows the values of column 'i2' and 'f1', and wants to predict the value of column 't1'.")
print('''
```
known_columns = ['i2', 'f1']
secret_column = 't1'
```
''')
known_columns = ['i2', 'f1']
secret_column = 't1'

print("\nFor the baseline predictions, we need to make a model from the original data.")
print("`base_pred.build_model(known_columns, secret_column)`")
base_pred.build_model(known_columns, secret_column)

print("\nRun the predictions loop. There is the question of how many predictions to make in order to get a statistically significant result. To avoid doing more work than necessary, pred_res can calculate confidence intervals over the set of predictions made so far.")

run_predictions_loop(adf, base_pred, pred_res, secret_column,
                     known_columns, adf.cntl, df_anon)

print("\nAfter the predictions loop, we can get a dataframe listing every prediction. Here is an example of a row for an attack prediction:")
print('''
```
df_results = pred_res.get_results_df()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
```
''')
df_results = pred_res.get_results_df()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])

print("\nNote that attack_confidence is 1.0. For this particular type of attack, this means that all of the rows that matched the known columns agreed on the predicted value. Different predictions, however, may have different confidence levels. In this case, we see that, among the predictions, there are 5 unique confidence levels:")

print("`df_results['attack_confidence'].unique())`")
print(df_results['attack_confidence'].unique())

print("\nThe PredictionResults class can compute the precision, recall, and ALC for each combination of known columns and secret column. When there are multiple confidence levels, the PredictionResults class computes the ALC for different recall values starting with only the highest confidence predictions (low recall), and working through lower confidence predictions.")
print("`df_per_comb_results = pred_res.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)`")
df_per_comb_results = pred_res.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)

print("\nIn total, here are the columns produced by the alc_per_secret_and_known_df method:")
print("`df_per_comb_results.columns`")
print(df_per_comb_results.columns)

print("\nLet's look at the precision, recall, and ALC scores:")
print("`df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']]`")
print(df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']])

print("\nAs it so happens, there is no correlation between 't1' and 'i2' or 'f1'. As a result, the baseline precision is always quite low, and different recall values don't really help. By contrast, because our anonymity is weak, attack precision is high, and improves as recall is lower. This leads to a high ALC score, the highest at recall=0.75, showing that anonymity is indeed weak.")

print("\nLet's run a second attack, here assuming that the attacker knows the value of column 'i1' and wants to predict the value of column 't1'.")
known_columns = ['i1']
secret_column = 't1'
base_pred.build_model(known_columns, secret_column)
run_predictions_loop(adf, base_pred, pred_res, secret_column,
                     known_columns, adf.cntl, df_anon)

print("\nLet's look at the precision, recall, and ALC scores for the second attack:")
df_per_comb_results = pred_res.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)
print(df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']])

print("\nHere we see quite a different story. Since 'i1' and 't1' are perfectly correlated, the baseline precision is always 1.0. Because of the anonymization, however, the attack precision, while pretty high (around 0.9), is not perfect. Because the attack precision is always less than the baseline precision, the ALC score is always negative, which translates to no anonymity loss whatsoever.")