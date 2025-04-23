import os
import sys
import argparse
import pandas as pd
from typing import List
from anonymity_loss_coefficient.attacks import BrmAttack

def launch_attack(data: str,
                  name: str,
                  secret: List[str] = None,
                  known: List[str] = None,
                  run_once: bool = False,
                  verbose: verbose,
                  ) -> None:
    if not os.path.isdir(data):
        print(f"Error: {data} is not a directory")
        sys.exit(1)
    inputs_path = os.path.join(data, 'inputs')
    if not os.path.exists(inputs_path) or not os.path.isdir(inputs_path):
        print(f"Error: {inputs_path} does not exist or is not a directory")
        print(f"Your test files should be in {inputs_path}")
        sys.exit(1)
    original_data_path = os.path.join(inputs_path, 'original.csv')
    try:
        df_original = pd.read_csv(original_data_path)
    except Exception as e:
        print(f"Error reading {original_data_path}")
        print(f"Error: {e}")
        sys.exit(1)
    if not os.path.exists(original_data_path):
        print(f"Error: {original_data_path} does not exist")
        sys.exit(1)
    synthetic_path = os.path.join(inputs_path, 'synthetic_files')
    if not os.path.exists(synthetic_path) or not os.path.isdir(synthetic_path):
        print(f"Error: {synthetic_path} does not exist or is not a directory")
        sys.exit(1)
    syn_dfs = []
    for file in os.listdir(synthetic_path):
        if file.endswith('.csv'):
            syn_dfs.append(pd.read_csv(os.path.join(synthetic_path, file)))
    results_path = os.path.join(data, 'results')
    brm = BrmAttack(df_original=df_original,
                    df_synthetic=syn_dfs,
                    results_path=results_path,
                    attack_name = name,
                    verbose = verbose,
                    )
    if run_once:
        brm.run_one_attack(secret_col=secret[0], known_columns=known)
        return
    if known is None:
        brm.run_all_columns_attack(secret_cols=secret)
    brm.run_auto_attack(secret_cols=secret, known_columns=known)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process command-line options.")

    # Add arguments
    parser.add_argument("-d", "--data", type=str, required=True, help="The path to the datasets.")
    parser.add_argument("-n", "--name", type=str, default=None, required=False, help="The name you'd like to give the attack.")
    parser.add_argument("-k", "--known", type=str, nargs='+', required=False, help="One or more known columns (separate by space). If not provided, all columns will be used.")
    parser.add_argument("-s", "--secret", type=str, nargs='+', required=False, help="One or more secret columns (separate by space). If not provided, all columns will be used.")
    parser.add_argument("-1", "--one", action="store_true", help="Run the attack once. If not included, runs multiple times.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Logging at debug level. (Does not effect sysout.)")

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    data = args.data
    # if data has trailing '/', then strip it
    if data.endswith('/'):
        data = data[:-1]
    if data.endswith('\\'):
        data = data[:-1]
    name = args.name
    if name is None:
        name = os.path.split(data)[-1]
    secret = args.secret
    known = args.known
    run_once = args.one
    if run_once:
        if secret is None or len(secret) != 1:
            print("Error: When running once, you must provide exactly one secret column.")
            sys.exit(1)
    verbose = args.verbose

    # Print the parsed arguments
    print(f"Data: {data}")
    print(f"Attack name: {name}")
    print(f"Secret: {secret}")
    print(f"Known columns: {known}")
    print(f"Run once: {run_once}")
    print(f"Verbose: {verbose}")

    launch_attack(data=data, name=name, secret=secret, known=known, run_once=run_once, verbose=verbose)

if __name__ == "__main__":
    main()