Assume that `df_initial` is the initial raw data that we want to measure
The initial data has 10000 rows. Here are the first 5 rows:

`df_initial.head()`
```
  t1  i1   i2         f1
0  c   3  150   9.333691
1  d   4  170   7.313547
2  a   1  195  11.171824
3  c   3  147  12.849438
4  c   3  118   7.755031
```
and here are some summary statistics:

`df_initial.describe()`
```
                i1            i2            f1
count  10000.00000  10000.000000  10000.000000
mean       2.49430    149.634600     10.013697
std        1.12138     28.771338      1.996967
min        1.00000    100.000000      1.870794
25%        1.00000    125.000000      8.655646
50%        2.00000    150.000000     10.028124
75%        4.00000    175.000000     11.383861
max        4.00000    199.000000     17.554980
```
Assume that 4 anonymized datasets have been generated from `df_initial`. Of course, this also works with only a single anonymized dataset. (The "anonymization" here is nothing more than swapping a small fraction of the values.)

To make baseline predictions, we need to separate out a control dataset from the initial raw data. Typically 1000 control rows is enough, though if the initial data is small, then the control dataset should be smaller (say 10% or 20% or the data).

```
df_control = df_initial.sample(1000)
df_original = df_initial.drop(df_control.index)
```

`df_control` has 1000 randomly sampled rows from `df_initial`.
`df_original` has the remaining 9000 rows.

At this point, we have prepared the dataframes needed for the ALC measures.

The DataFiles class is used primarily to preprocess the data. It removes NaN rows, discretizes continuous variables, and encodes non-integer columns as integers. Note in particular that, unless the optional parameter `discertize_in_place` is set to True, the DataFiles class creates a new column for each discretized column, given the name `colname__discretized`. The original column is also kept. The discretized column should be used for the secret column, while the original column should be used for the known column.

`adf = DataFiles(df_original, df_control, syn_data)`

We see for instance that the text column 't1' has been encoded as integers, and two discretized columns have been created from the continuous columns:

`adf.orig.head()`
```
   t1  i1   i2    f1  i2__discretized  f1__discretized
1   3   4  170   881               14               15
2   0   1  195  7165               19                1
3   2   3  147  9251                9                3
4   2   3  118  1317                3               16
6   0   1  183  2686               16               17
```

The `BaselinePredictor` class is used to make baseline predictions on categorical columns. Internally it uses `df_original` from the `DataFiles` class.

`base_pred = BaselinePredictor(adf)`

The PredictionResults class is a helper class that stores the results of the predictions, and provides methods that summarize the precision, recall, and ALC scores.

```
pred_res = PredictionResults(results_path = "example",
                             attack_name = "Example Attacks")
```


Now lets run the attacks. An attack consists of a set of predictions on the value of a categorical column (the 'secret' column), assuming knowledge of the value of one or more other columns (the 'known columns'). We make two kinds of predictions, attack predictions and baseline predictions. An attack prediction is made on a row taken from the initial data over the anonymized data. A baseline prediction is made from a row taken from the control data over the original data. Note that, since the control row is not part of the original data, the prediction is privacy neutral.

We want to make the same number of attack and baseline predictions. Let's set aside a some original rows, equal to the number of control rows, for the attack predictions. Note that we need to work only with the data that has been preprocessed by the DataFiles class. The preprocessed original, control, and synthetic dataframes can be accessed as adf.orig, adf.cntl, and adf.syn_list[i] respectively.

```
num_control_rows = len(adf.cntl)
df_atk = adf.orig.sample(num_control_rows)
```


To keep this example simple, the attack itself is also naively simple. We combine the preprocessed anonymized dataframes into a single dataframe. We find the rows in the combined anonymized dataset whose values match the known columns of the attack row, if any, and predict that value that is most common among these rows. We then select the majority value of this set of values as our attack prediction. We also compute a 'confidence' associated with the prediction. In this case, our confidence will be the fraction of rows of the matching rows that contain the predicted value.

In a first attack, let's assume that the attacker knows the values of column 'i2' and 'f1', and wants to predict the value of column 't1'.

```
known_columns = ['i2', 'f1']
secret_column = adf.get_discretized_column('t1')
```


Note the use of the `get_discretized_column` method.  This produces the column name of the discretized column, if any. If none, it returns the original column name (which is the case here).

For the baseline predictions, we need to make a model from the original data.

`base_pred.build_model(known_columns, secret_column)`

Run the predictions loop. There is the question of how many predictions to make in order to get a statistically significant result. To avoid doing more work than necessary, pred_res can calculate confidence intervals over the set of predictions made so far.
Attack confidence interval (0.91, 1.0) is within tolerance after 51 attacks on precision 1.0

After the predictions loop, we can get a dataframe listing every prediction. Here is an example of a row for an attack prediction:

```
df_results = pred_res.get_results_df()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
```

```
predict_type               attack
known_columns        ["f1", "i2"]
num_known_columns               2
secret_column                  t1
predicted_value                 a
true_value                      a
prediction                   True
base_confidence               NaN
attack_confidence             1.0
Name: 1, dtype: object
```

Note that attack_confidence is 1.0. For this particular type of attack, this means that all of the rows that matched the known columns agreed on the predicted value. Different predictions, however, may have different confidence levels. In this case, we see that, among the predictions, there are 5 unique confidence levels:

`df_results['attack_confidence'].unique())`
```
[       nan 1.         0.75       0.33333333 0.5        0.66666667]
```

The PredictionResults class can compute the precision, recall, and ALC for each combination of known columns and secret column. When there are multiple confidence levels, the PredictionResults class computes the ALC for different recall values starting with only the highest confidence predictions (low recall), and working through lower confidence predictions.

`df_per_comb_results = pred_res.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)`

In total, here are the columns produced by the alc_per_secret_and_known_df method:

`df_per_comb_results.columns`
```
Index(['secret_column', 'known_columns', 'num_known_columns', 'base_prec',
       'base_recall', 'attack_prec', 'attack_recall', 'alc', 'base_count',
       'attack_count', 'base_ci', 'base_ci_low', 'base_ci_high', 'base_n',
       'attack_ci', 'attack_ci_low', 'attack_ci_high', 'attack_n'],
      dtype='object')
```

Let's look at the precision, recall, and ALC scores:

`df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']]`
```
   base_prec  base_recall  attack_prec  attack_recall       alc
0   0.131579     0.745098     1.000000       0.745098  0.934172
1   0.170213     0.921569     0.957447       0.921569  0.867970
2   0.163265     0.960784     0.959184       0.960784  0.873563
3   0.180000     0.980392     0.940000       0.980392  0.843409
4   0.176471     1.000000     0.921569       1.000000  0.824924
```

As it so happens, there is no correlation between 't1' and 'i2' or 'f1'. As a result, the baseline precision is always quite low, and different recall values don't really help. By contrast, because our anonymity is weak, attack precision is high, and improves as recall is lower. This leads to a high ALC score, the highest at recall=0.75, showing that anonymity is indeed weak.

Let's run a second attack, here assuming that the attacker knows the value of column 'i1' and wants to predict the value of column 't1'.
Base confidence interval (0.9, 1.0) is within tolerance after 131 attacks on precision 1.0

Let's look at the precision, recall, and ALC scores for the second attack:
```
   base_prec  base_recall  attack_prec  attack_recall          alc
5        1.0     0.274809     0.888889       0.274809   -20.068768
6        1.0     0.519084     0.911765       0.519084  -118.980444
7        1.0     0.770992     0.900990       0.770992 -1522.527493
8        1.0     1.000000     0.893130       1.000000 -5343.564885
```

Here we see quite a different story. Since 'i1' and 't1' are perfectly correlated, the baseline precision is always 1.0. Because of the anonymization, however, the attack precision, while pretty high (around 0.9), is not perfect. Because the attack precision is always less than the baseline precision, the ALC score is always negative, which translates to no anonymity loss whatsoever.

Finally, we can get a summary of the ALC scores for all attacks. This is placed in a directory with the name originally conveyed to the PredictionResults class `results_path` variable. In our case, 'example'.

`pred_res.summarize_results()`

This produces the following files:
* summary_raw.csv: All of the predictions
* summary_secret.csv: The precision, recall, and ALC scores for predictions grouped by secret column
* summary_secret_known.csv: The precision, recall, and ALC scores for predictions grouped by secret column and known columns
* summary.txt: A descriptive summary of the results

Note finally that, if there are enough attacks to warrant it, `pred_res.summarize_results()` generates several plots as well.

