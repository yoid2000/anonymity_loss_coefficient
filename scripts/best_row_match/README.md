# Best run_brm_attack.py

This code example runs the Best Row Match attack found in `anonymity_loss_coefficient/attacks/best_row_match/`.

It can be used either as an example for how to write your own code to run the Best Row Match attack, or it can be used as a complete attack itself.

## To run

Pip install `anonymity_loss_coefficient`.

* Setup the original and anonymous files as described below.
* Run `python run_brm_attack.py --data /path/to/attack_directory`, where `/path/to/attack_directory` is the location of the files you setup.

`run_brm_attack.py` has the following options:
* -d or --data: <path to attack directory>
* -s or --secret: the secret columns that you wish to attack, supplied as a list of column names separated by spaces. If not supplied, assumes that all columns are secret.
* -k or --known: the known columns that you wish to attack, supplied as a list of column names separated by spaces. If not supplied, assumes that all columns are known columns.
* -1 or --one: if set, then run exactly one attack on the supplied secret column and (if present) the supplied known columns. If not set, then a variety of known column combinations, taken from the columns listed in --known (if any), will be attacked.

Examples:
* `python run_brm_attack.py --data /path/to/attack_directory`: Assigns every column as a secret column in turn, and then runs attacks using a variety of known column sets (starting with all known columns, and then finding the most promising smaller known column sets)
* `python run_brm_attack.py --data /path/to/attack_directory --secret secret_col`: Runs attacks against secret column `secret_col` using a variety of known column sets (starting with all known columns, and then finding the most promising smaller known column sets)
* `python run_brm_attack.py --data /path/to/attack_directory --secret secret_col --known col1 col2 col3 col4 col5 col6`: Runs attacks against secret column `secret_col` using a variety of known column sets taken only from the specified columns (col1 through col6)
* `python run_brm_attack.py --data /path/to/attack_directory --secret secret_col --known col1 col2 col3 col4 col5 col6 --one`: Runs exactly one attack against secret column `secret_col` using the specified known columns (`col1` through `col6`)
* `python run_brm_attack.py --data /path/to/attack_directory --secret scol1 scol2`: Runs attacks against secret columns `scol1` and `scol2` using a variety of known column sets (starting with all known columns, and then finding the most promising smaller known column sets)


Note that `run_brm_attack.py` updates its results after every run. If --one is not supplied, then there can be a large number of known columns and secret column combinations (hundreds), and `run_brm_attack.py` will simply continue to generate more results until you stop it.

For quick testing, use the setup directories already setup in the `files` directory. In this case, copy `files/attack_files_anon` or `files/attack_files_raw` to your `/path/to/attack_directory` and run as above.

## Setup

`run_brm_attack.py` expects to find everything it needs in `attack_directory/inputs`. Specifically, the following should be placed there:

* `your_file_name.csv` or `your_file_name.parquet`: Contains the original data.

* `synthetic_files`: This is a directory containing one or more synthetic datasets generated from the original data. Each of these files may have a subset of the columns from the original data. These can be `.csv` or `.parquet` files.

Note that the larger the synthetic datasets are, the longer it'll take to run the tests. Therefore it might be a good idea to limit the synthetic datasets (and likewise the original data from which they were derived) to 10k-20k or so rows.

See the directories under the `files` directory for examples of the setup.


## Results

`run_brm_attack.py` creates a directory `results` under `attack_diretory`. `results` contains these files:

* `summary_raw.csv`: This contains the results of every individual prediction, both baseline and attack.
* `summary_secret.csv`: This contains data every secret (unknown) attribute being predicted. It gives the precision, recall, ALC score, and confidence intervals for a variety of different recall values.
* `summary_secret_known.csv`: This contains a row for every combination of secret and known attributes. It gives the precision, recall, ALC score, and confidence intervals for a variety of different recall values.
* `summary.txt`: Gives a text summary of the results, included an anonymity grade ranging from VERY STRONG to VERY POOR.
* `alc_plot.png`: A plot summarizing the ALC scores for each set of secret attributes.
* `alc_plot_best.png`: A plot summarizing the ALC scores for the highest ALC score among a given set of secret attribute and known attributes.
* `alc_plot_prec.png`: A scatterplot showing the ALC score and attack precision for each set of secret and known attributes.
* `alc_plot_prec_best.png`: A scatterplot showing the highest ALC score and attack precision for each set of secret and known attributes.

ALC scores of ALC=0.5 or less can be regarded as having very strong anonymity.


## Operation

`run_brm_attack.py` does the following:

It identifies the categorical columns to be used as unknown (secret) attributes.

For each secret:

* It runs an attack assuming that all other columns are known attributes.
* It discovers smaller sets of known attributes that uniquely define each row of the data.
* It selects a set of secret values where each value constitutes fewer than 60% of the rows, but more than 0.05% of the rows. We avoid very common values because they tend not to be sensitive. We avoid very rare values because most predictions tend to be False, and an occasional random correct prediction can skew the results.
* After shuffling the target rows, it steps through the rows for both baseline and synthetic data attacks, making a prediction for each row. It quits when either the confidence bounds are with 10% of the precision for either the baseline or the attacks, and when at least 5 True predictions have been made for both baseline and attack.
* When making a prediction, if there are multiple synthetic data files, it first makes a prediction for each data file, and then uses the most common prediction as the prediction. It also assigns a confidence value to the prediction, depending on how many of the individual predictions constitute the most common prediction.
* It updates the three results files. In this fashion, the results files continuously receive more results data as `run_brm_attack.py` runs.

Note that `run_brm_attack.py` can take a long time to run (many hours), so it is good to just let it go while the results files build up.