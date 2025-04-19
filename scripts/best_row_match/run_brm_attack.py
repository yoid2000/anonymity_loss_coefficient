import argparse
import os
import pandas as pd
import sys
from anonymity_loss_coefficient.attacks import BrmAttack
import pprint

pp = pprint.PrettyPrinter(indent=4)

test_params = False


def run_attacks(attack_files_path):
    #split attack_files_path into the path and the attack_files directory
    attack_dir_name = os.path.split(attack_files_path)[-1]
    # Check that there is indeed a directory at attack_files_path
    if not os.path.isdir(attack_files_path):
        print(f"Error: {attack_files_path} is not a directory")
        sys.exit(1)
    inputs_path = os.path.join(attack_files_path, 'inputs')
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
    results_path = os.path.join(attack_files_path, 'results')
    if test_params:
        brm = BrmAttack(df_original=df_original,
                        df_synthetic=syn_dfs,
                        results_path=results_path,
                        max_known_col_sets=100,
                        num_per_secret_attacks=2,
                        max_rows_per_attack=10,
                        min_rows_per_attack=5,
                        attack_name = attack_dir_name,
                        )
    else:
        brm = BrmAttack(df_original=df_original,
                        df_synthetic=syn_dfs,
                        results_path=results_path,
                        attack_name = attack_dir_name,
                        )
    brm.run_auto_attack()


def attack_stats():
    pass

def do_plots():
    pass

def main():
    parser = argparse.ArgumentParser(description="Run different anonymeter_plus commands.")
    parser.add_argument("command", help="'attack' to run make_config(), 'stats' to run attack_stats(), or 'plots' to run do_plots()")
    parser.add_argument("attack_files_path", nargs='?', help="Optional path for attack files, used with the 'attack' command")

    args = parser.parse_args()

    if args.command == 'attack':
        attack_files_path = args.attack_files_path
        if attack_files_path:
            print(f"Using attack files path: {attack_files_path}")
        else:
            attack_files_path = 'attack_files'
        run_attacks(attack_files_path)
    elif args.command == 'stats':
        attack_stats()
    elif args.command == 'plots':
        do_plots()
    else:
        raise ValueError(f"Unrecognized command: {args.command}")

    # Print the attack_files_path to verify
    print(f"attack_files_path: {attack_files_path}")

if __name__ == "__main__":
    main()