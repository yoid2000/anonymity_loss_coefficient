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

The DataFiles class is used primarily to preprocess the data. It removes NaN rows, discretizes continuous variables, and encodes non-integer columns as integers.

`adf = DataFiles(df_original, df_control, syn_data)`

We see for instance that the text column 't1' has been encoded as integers:

`adf.orig.head()`
```
   t1  i1  i2  f1
1   3   4  14  15
2   0   1  19   1
3   2   3   9   3
4   2   3   3  16
6   0   1  16  17
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
secret_column = 't1'
```


For the baseline predictions, we need to make a model from the original data.

`base_pred.build_model(known_columns, secret_column)`

Run the predictions loop. There is the question of how many predictions to make in order to get a statistically significant result. To avoid doing more work than necessary, pred_res can calculate confidence intervals over the set of predictions made so far.

After the predictions loop, we can get a dataframe listing every prediction. Here is an example of a row for an attack prediction:

```
df_results = pred_res.get_results_df()
print(df_results[df_results['predict_type'] == 'attack'].iloc[0])
```

```
predict_type               attack
known_columns        ["f1", "i2"]
num_known_columns               2
target_column                  t1
predicted_value                 c
true_value                      a
prediction                  False
base_confidence               NaN
attack_confidence        0.323887
Name: 1, dtype: object
```

Note that attack_confidence is 1.0. For this particular type of attack, this means that all of the rows that matched the known columns agreed on the predicted value. Different predictions, however, may have different confidence levels. In this case, we see that, among the predictions, there are 5 unique confidence levels:

`df_results['attack_confidence'].unique())`
```
[       nan 0.32388664 0.33571429 0.30434783 0.27667984 0.26804124
 0.40952381 0.27304965 0.2971246  0.37931034 0.28909953 0.27046263
 0.2740113  0.27407407 0.28708134 0.35714286 0.35433071 0.39726027
 0.30916031 0.27760252 0.27848101 0.40384615 0.28901734 0.38461538
 0.29787234 0.36216216 0.3        0.28776978 0.3046595  0.30363036
 0.26845638 0.29454545 0.35625    0.28108108 0.3539823  0.35869565
 0.30232558 0.32919255 0.31538462 0.43157895 0.28627451 0.63636364
 0.28275862 0.31007752 0.30877193 0.32163743 0.27310924 0.35294118
 0.28279883 0.27380952 0.31962025 0.32653061 0.34782609 0.31333333
 0.30718954 0.34317343 0.32984293 0.3373494  0.30405405 0.34883721
 0.32467532 0.46666667 0.34375    0.41860465 0.27952756 0.27272727
 0.3003663  0.30716724 0.28358209 0.28       0.32967033 0.28286853
 0.29483283 0.33928571 0.32098765 0.34358974 0.30851064 0.30687831
 0.31698113 0.26038781 0.30821918 0.28476821 0.31313131 0.40983607
 0.29878049 0.30534351 0.28846154 0.30841121 0.29677419 0.27027027
 0.39252336 0.296875   0.38235294 0.28706625 0.29607251 0.33976834
 0.28027682 0.26785714 0.32051282 0.36526946 0.42307692 0.25590551
 0.27941176 0.27777778 0.28643216 0.34302326 0.29126214 0.28327645
 0.33727811 0.38888889 0.35897436 0.3164557  0.29182879 0.32170543
 0.31399317 0.27388535 0.29885057 0.2994012  0.26739927 0.28807947
 0.3625     0.28174603 0.29714286 0.39393939 0.29936306 0.8
 0.31192661 0.27868852 0.33333333 0.34391534 0.34482759 0.31120332
 0.42857143 0.33557047 0.32663317 0.3537415  0.28506787 0.34146341
 0.29850746 0.2754717  0.29861111 0.31683168 0.40909091 0.34163701
 0.28947368 0.4        0.37241379 0.27329193 0.83333333 0.28571429
 0.33474576 0.61538462 0.37837838 0.38636364 0.29665072 0.29139073
 0.29378531 0.28235294 0.35251799 0.27631579 0.35106383 0.32075472
 0.31578947 0.27586207 0.34334764 0.33663366 0.34666667 0.37878788
 0.41818182 0.29496403 0.30344828 0.27532468 0.52941176 0.37179487
 0.29613734 0.2748538  0.88888889 0.61111111 0.26690391 1.
 0.36507937 0.35114504 0.40853659 0.29323308 0.28645833 0.47058824
 0.32894737 0.28787879 0.66666667 0.70833333 0.32333333 0.36170213
 0.31481481 0.5        0.34285714 0.36764706 0.33766234 0.29411765
 0.44444444 0.39130435 0.375     ]
```

The PredictionResults class can compute the precision, recall, and ALC for each combination of known columns and secret column. When there are multiple confidence levels, the PredictionResults class computes the ALC for different recall values starting with only the highest confidence predictions (low recall), and working through lower confidence predictions.

`df_per_comb_results = pred_res.alc_per_secret_and_known_df(known_columns=known_columns, secret_column=secret_column)`

In total, here are the columns produced by the alc_per_secret_and_known_df method:

`df_per_comb_results.columns`
```
Index(['target_column', 'known_columns', 'num_known_columns', 'base_prec',
       'base_recall', 'attack_prec', 'attack_recall', 'alc', 'base_count',
       'attack_count', 'base_ci', 'base_ci_low', 'base_ci_high', 'base_n',
       'attack_ci', 'attack_ci_low', 'attack_ci_high', 'attack_n'],
      dtype='object')
```

Let's look at the precision, recall, and ALC scores:

`df_per_comb_results[['base_prec', 'base_recall', 'attack_prec', 'attack_recall', 'alc']]`
```
   base_prec  base_recall  attack_prec  attack_recall       alc
0   1.000000        0.001     1.000000          0.001  0.000000
1   0.194444        0.036     0.527778          0.036  0.353785
2   0.238636        0.088     0.465909          0.088  0.257219
3   0.204545        0.176     0.397727          0.176  0.216347
4   0.232704        0.318     0.327044          0.318  0.108400
5   0.242222        0.450     0.311111          0.450  0.079837
6   0.236181        0.597     0.298157          0.597  0.071543
7   0.234795        0.707     0.298444          0.707  0.073409
8   0.244392        0.847     0.299882          0.847  0.064463
9   0.253000        1.000     0.302000          1.000  0.057297
```

As it so happens, there is no correlation between 't1' and 'i2' or 'f1'. As a result, the baseline precision is always quite low, and different recall values don't really help. By contrast, because our anonymity is weak, attack precision is high, and improves as recall is lower. This leads to a high ALC score, the highest at recall=0.75, showing that anonymity is indeed weak.

Let's run a second attack, here assuming that the attacker knows the value of column 'i1' and wants to predict the value of column 't1'.
Base confidence interval (0.9, 1.0) is within tolerance after 131 attacks on precision 1.0

Let's look at the precision, recall, and ALC scores for the second attack:
```
    base_prec  base_recall  attack_prec  attack_recall          alc
10        1.0     0.274809     0.888889       0.274809   -20.068768
11        1.0     0.519084     0.911765       0.519084  -118.980444
12        1.0     0.770992     0.900990       0.770992 -1522.527493
13        1.0     1.000000     0.893130       1.000000 -5343.564885
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

