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

The ALCManager class is used for all operations. It prepares the data, runs the baseline model, holds the various predictions, computes the ALC measures, and writes the results to files.

To prepare the data, it removes NaN rows, discretizes continuous variables, and encodes non-integer columns as integers. Note in particular that, unless the optional parameter `discertize_in_place` is set to True, it creates a new column for each discretized column, given the name `colname__discretized`. The original column is also kept. The discretized column should be used for the secret column, while the original column should be used for the known column.

`alcm = ALCManager(df_original, syn_data)`

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

Note in particular that the df_original and syn_data dataframes are not used once the ALCManager object has been created. All subsequent operations are made on the processed dataframes in the ALCManager object (`alcm.df.orig`, `alcm.df.cntl`, and `alcm.df.syn_list`).

Now lets run the attacks. An attack consists of a set of predictions on the value of a categorical column (the 'secret' column), assuming knowledge of the value of one or more other columns (the 'known columns'). We make two kinds of predictions, attack predictions and baseline predictions. An attack prediction is made on a row taken from the control data over the anonymized data. A baseline prediction is made from a row taken from the control data over the original data. Note that, since the control row is not part of the original data, the baseline prediction is privacy neutral.

To keep this example simple, the attack itself is also naively simple. We combine the preprocessed anonymized dataframes into a single dataframe. We find the rows in the combined anonymized dataset whose values match the known columns of the attack row, if any, and predict that value that is most common among these rows. We then select the majority value of this set of values as our attack prediction. We also compute a 'confidence' associated with the prediction. In this case, our confidence will be the fraction of rows of the matching rows that contain the predicted value.

In a first attack, let's assume that the attacker knows the values of column 'i2' and 'f1', and wants to predict the value of column 't1'.

```
known_columns = ['i2', 'f1']
secret_column = alcm.get_discretized_column('t1')
```


Note the use of the `get_discretized_column` method.  This produces the column name of the discretized column, if any. If none, it returns the original column name (which is the case here).

Run the predictions loop. There is the question of how many predictions to make in order to get a statistically significant result. To avoid doing more work than necessary, the ALCManager calculates confidence intervals over the set of predictions made so far, and stops when confidence intervals are high enough.

The predictions loop runs from the ALCManager `predictor()` class. This is a generator that feeds attack rows one by one. Inside the loop, we make our prediction and then call the ACLManager `prediction()` method to register our prediction. The `predictor()` generator does the rest, including computing the baseline precision and deciding when to quit.

```
for atk_row, _, _ in alcm.predictor(known_columns, secret_column):
    encoded_predicted_value, prediction_confidence = attack_prediction(alcm, df_anon, atk_row, secret_column, known_columns)
    alcm.prediction(encoded_predicted_value, prediction_confidence)
```

Note that we are ignoring two values returned by the predictor(). The two parameters are `encoded_true_value` and `decoded_true_value`. These are the values used internally by the ACLManager class to determine if the prediction is True of False. They are provided as a convenience, but otherwise aren't needed by the attacking code.


That's really all there is to it! There are a few ways in which we can now look at the results of the attack.

After the predictions loop, we can get a dataframe listing every prediction using the `get_results_df()` method. Here is an example of a row for an individual attack prediction:

```
df_results = alcm.get_results_df()
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

Note that attack_confidence is 1.0. For this particular type of attack, this means that all of the rows that matched the known columns agreed on the predicted value. Different predictions, however, may have different confidence levels. In this case, we see that, among the predictions, there are 3 unique confidence levels:

`df_results['attack_confidence'].unique())`
```
[1.         0.75       0.66666667]
```

The method `alc_per_secret_and_known_df()` groups the individual attack predictions by secret column and known columns, and computes a variety of scores including precision, recall, and ALC. When there are multiple confidence levels, the PredictionResults class computes the ALC for different recall values starting with only the highest confidence predictions (low recall), and working through lower confidence predictions.

`df_per_comb_results = alcm.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)`

Let's look at the precision, recall, and ALC scores:

`df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']]`
```
   base_prec  base_recall  attack_prec  attack_recall       alc
0   0.326394     0.786885     0.963651       0.803279  0.946006
1   0.287565     0.901639     0.967358       0.901639  0.954167
2   0.291800     1.000000     0.970378       1.000000  0.958159
```

As it so happens, there is no correlation between 't1' and 'i2' or 'f1'. As a result, the baseline precision is always quite low, and different recall values don't really help. By contrast, because our anonymity is weak, attack precision is high uniformly high, so as it happens recall doesn't really matter here either. The fact that attack precision is greater than baseline precision leads to high ALC scores, showing that anonymity is indeed weak.

Let's run a second attack, here assuming that the attacker knows the value of column 'i1' and wants to predict the value of column 't1'.

Let's look at the precision, recall, and ALC scores for the second attack:
```
   base_prec  base_recall  attack_prec  attack_recall       alc
3    0.95055          1.0     0.878753       0.342857 -1.479484
4    0.95055          1.0     0.907841       0.485714 -0.872335
5    0.95055          1.0     0.935636       0.742857 -0.302169
6    0.95055          1.0     0.950550       1.000000  0.000000
```

Here we see quite a different story. Since 'i1' and 't1' are perfectly correlated, all baseline predictions are correct. (The reason `base_prec` is not perfect is because of how we compute precision: as the midpoint of the confidence interval rather than the actual predictions. The actual sampled precision, however, is also computed and can be viewed.) As it so happens, all attack predictions are also correct. (In this case, `attack_prec` is lower for lower recall values only because the confidence bounds are larger.). Because the attack precision is no better than the base precision, the ALC is at best 0.0, meaning no loss of anonymity.

Besides being able to obtain the results as dataframes, the method `summarize_results()` writes the results to CSV files and can generate plots as well:

`alcm.summarize_results(results_path = "generic_example_files", attack_name = "Example Attacks")`

This produces the following files (which can be viewed in the `generic_example_files` directory):
* summary_raw.csv: All of the predictions
* summary_secret.csv: The precision, recall, and ALC scores for predictions grouped by secret column
* summary_secret_known.csv: The precision, recall, and ALC scores for predictions grouped by secret column and known columns
* summary.txt: A descriptive summary of the results


