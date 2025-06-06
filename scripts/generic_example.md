## Example of using the ALCManager class to build attacks.

Assume that `df_original` is the original raw data that we want to measure
The original data has 10000 rows. Here are the first 5 rows:

`df_original.head()`
```
  t1  i1   i2         f1
0  c   3  150   9.333691
1  d   4  170   7.313547
2  a   1  195  11.171824
3  c   3  147  12.849438
4  c   3  118   7.755031
```
and here are some summary statistics:

`df_original.describe()`
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
Assume that 4 anonymized datasets have been generated from `df_original`. Of course, this also works with only a single anonymized dataset. (The "anonymization" here is nothing more than swapping a small fraction of the values.)

At this point, we have prepared the dataframes needed for the ALC measures.

The `ALCManager` class is used for all operations. It prepares the data, runs the baseline model, holds the various predictions, computes the ALC measures, and writes the results to files.

To prepare the data, it removes NaN rows, discretizes continuous variables, and encodes non-integer columns as integers. Note in particular that, unless the optional parameter `discertize_in_place` is set to True, it creates a new column for each discretized column, given the name `colname__discretized`. The original column is also kept. The discretized column should be used for the secret column, while the original column should be used for the known column.
The `ALCManager` passes the `attack_tags` to result files for later housekeeping.
The `flush` parameter (default `False`) tells the `ALCManager` to remove all previously recorded attacks. If set to `False`, the `ALCManager` will not repeat any attacks already run.

`alcm = ALCManager(df_original, syn_data, results_path = "generic_example_files", attack_name = "Example Attacks", attack_tags = {'foo': 1, 'bar': 'simple'}, flush = True)`

We see for instance that the text column 't1' has been encoded as integers, and two discretized columns have been created from the continuous columns:

`alcm.df.orig_all.head()`
```
  t1  i1   i2         f1  i2__discretized  f1__discretized
0  2   3  150   9.333691               10                9
1  3   4  170   7.313547               14                6
2  0   1  195  11.171824               19               11
3  2   3  147  12.849438                9               13
4  2   3  118   7.755031                3                7
```

Note in particular that the df_original and syn_data dataframes are not used once the ALCManager object has been created. All subsequent operations are made on the processed dataframes in the ALCManager object (`alcm.df.orig`, `alcm.df.cntl`, and `alcm.df.anon`).

Now lets run the attacks. An attack consists of a set of predictions on the value of a categorical column (the 'secret' column), assuming knowledge of the value of one or more other columns (the 'known columns'). We make two kinds of predictions, attack predictions and baseline predictions. An attack prediction is made on a row taken from the control data over the anonymized data. A baseline prediction is made from a row taken from the control data over the original data. Note that, since the control row is not part of the original data, the baseline prediction is privacy neutral.

To keep this example simple, the attack itself is also naively simple. We combine the preprocessed anonymized dataframes into a single dataframe. We find the rows in the combined anonymized dataset whose values match the known columns of the attack row, if any, and predict that value that is most common among these rows. We then select the majority value of this set of values as our attack prediction. We also compute a 'confidence' associated with the prediction. In this case, our confidence will be the fraction of rows of the matching rows that contain the predicted value.

In a first attack, let's assume that the attacker knows the values of column 'i2' and 'f1', and wants to predict the value of column 't1'.

```
known_columns = ['i2', 'f1']
secret_column = alcm.get_discretized_column('t1')
```


Note the use of the `get_discretized_column` method.  This produces the column name of the discretized column, if any. If none, it returns the original column name (which is the case here).

Run the predictions loop. There is the question of how many predictions to make in order to get a statistically significant result. To avoid doing more work than necessary, the ALCManager calculates confidence intervals over the set of predictions made so far, and stops when confidence intervals are high enough.

The predictions loop runs from the ALCManager `predictor()` class. This is a generator that feeds attack rows one by one. Inside the loop, we make our prediction (here `attack_prediction()`) and then call either the ACLManager `prediction()` or `abstention()` method to register our prediction. The `abstention()` method is used when the quality of the prediction is likely quite poor. Under the hood, it causes the baseline prediction to be used as the attack prediction. The `predictor()` generator does the rest, including computing the baseline precision and deciding when to quit.

```
for atk_row, _, _ in alcm.predictor(known_columns, secret_column):
    encoded_predicted_value, confidence, abstain = attack_prediction(alcm, df_anon, atk_row, secret_column, known_columns)
    if abstain:
        alcm.abstention()
    else:
        alcm.prediction(encoded_predicted_value, confidence)
```

Note that we are ignoring two values returned by the predictor(). The two parameters are `encoded_true_value` and `decoded_true_value`. These are the values used internally by the ACLManager class to determine if the prediction is True of False. They are provided as a convenience, but otherwise aren't needed by the attacking code.


That's really all there is to it! There are a few ways in which we can now look at the results of the attack.

After the predictions loop, we can get a dataframe listing every prediction using the `prediction_results()` method. Here is an example of a row for an individual attack prediction:

```
df_results = alcm.prediction_results()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
```

```
predict_type                     attack
known_columns              ["f1", "i2"]
num_known_columns                     2
secret_column                        t1
predicted_value                       c
true_value                            c
encoded_predicted_value               2
encoded_true_value                    2
prediction                         True
base_confidence                     NaN
attack_confidence                   1.0
Name: 1, dtype: object
```

Note that attack_confidence is 1.0. For this particular type of attack, this means that all of the rows that matched the known columns agreed on the predicted value. Different predictions, however, may have different confidence levels. In this case, we see that, among the predictions, there are 5 unique confidence levels:

`df_results[df_results['predict_type'] == 'attack']['attack_confidence'].unique()`
```
[1.         0.75       0.66666667 0.5        0.33333333]
```

The method `results()` groups the individual attack predictions by secret column and known columns, and computes a variety of scores including precision, recall, and ALC. When there are multiple confidence levels, the PredictionResults class computes the ALC for different recall values starting with only the highest confidence predictions (low recall), and working through lower confidence predictions.

`df_per_comb_results = alcm.results(known_columns=known_columns, secret_column=secret_column)`

Let's look at the precision, recall, and ALC scores:

`df_per_comb_results[['paired', 'base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc', 'atk_bar']]`
```
   paired  base_prec  base_recall  attack_prec  attack_recall     alc atk_bar
0    True     0.2856       0.7567       0.9873         0.7567  0.9823  simple
1    True     0.2627       0.9633       0.9900         0.9633  0.9865  simple
2    True     0.2630       1.0000       0.9871         1.0000  0.9825  simple
3   False     0.2630       1.0000       0.9900         0.9633  0.9865  simple
```

As it so happens, there is no correlation between 't1' and 'i2' or 'f1'. As a result, the baseline precision is always quite low. By contrast, because our anonymity is weak, attack precision is uniformly high. The fact that attack precision is greater than baseline precision leads to high ALC scores, showing that anonymity is indeed weak.

The `paired` column indicates whether the ALC score is generated from a pair of closely-matched recall values for attack and baseline. If `False`, then the ALC score is generated from the best attack Privacy-Recall Coefficient (PRC) and the best baseline PRC regardless of recall. This represents the most appropriate ALC score (though not necessarily the highest ALC score).

Note that the attack tags are placed in this output as 'atk_key' columns names.

Let's run a second attack, here assuming that the attacker knows the value of column 'i1' and wants to predict the value of column 't1'.

Let's look at the precision, recall, and ALC scores for the second attack:
```
   paired  base_prec  base_recall  attack_prec  attack_recall     alc
4    True     0.9906          1.0       0.9668          0.270 -2.8189
5    True     0.9906          1.0       0.9881          0.785 -0.2692
6    True     0.9906          1.0       0.9906          1.000  0.0000
7   False     0.9906          1.0       0.9906          1.000  0.0000
```

Here we see quite a different story. Since 'i1' and 't1' are perfectly correlated, all baseline predictions are correct. (The reason `base_prec` is not perfect is because of how we compute precision: as the midpoint of the confidence interval rather than the actual predictions. The actual sampled precision, however, is also computed and can be viewed.) As it so happens, all attack predictions are also correct. (In this case, `attack_prec` is lower for lower recall values only because the confidence bounds are larger.). Because the attack precision is no better than the base precision, the ALC is at best 0.0, meaning no loss of anonymity.

Besides being able to obtain the results as dataframes, the method `summarize_results()` writes the results to CSV files and can generate plots as well:

`alcm.summarize_results()`

This produces the following files (which can be viewed in the `generic_example_files` directory):
* summary_raw.parquet: All of the predictions
* summary_secret_known.csv: The precision, recall, and ALC scores for predictions grouped by secret column and known columns
* summary.txt: A descriptive summary of the results


