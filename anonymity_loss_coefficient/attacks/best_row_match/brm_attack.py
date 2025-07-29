import os
import pandas as pd
import random
from typing import List, Union, Any, Tuple, Optional
from itertools import combinations
import logging
from anonymity_loss_coefficient.alc.alc_manager import ALCManager
from anonymity_loss_coefficient.utils import get_good_known_column_sets, setup_logging, find_best_matches, modal_fraction, best_match_confidence
import pprint

pp = pprint.PrettyPrinter(indent=4)

class BrmAttack:
    def __init__(self,
                 df_original: pd.DataFrame,
                 anon: Union[pd.DataFrame, List[pd.DataFrame]],
                 results_path: str = None,
                 max_known_col_sets: int = 1000,
                 known_cols_sets_unique_threshold: float = 0.45,
                 num_per_secret_attacks: int = 100,
                 max_num_anon_datasets: int = 1,
                 attack_name: str = '',
                 additional_tags: dict = {},
                 verbose: bool = False,
                 no_counter: bool = True,
                 flush: bool = False,
                 prior_experiment_swap_fraction: float = -1.0,
                 match_method: str = 'gower',
                 ) -> None:
        # up to work with ML modeling

        self.max_num_anon_datasets = max_num_anon_datasets
        self.prior_experiment_swap_fraction = prior_experiment_swap_fraction
        self.flush = flush
        self.results_path = results_path
        self.match_method = match_method
        self.max_known_col_sets = max_known_col_sets
        self.known_cols_sets_unique_threshold = known_cols_sets_unique_threshold
        self.num_per_secret_attacks = num_per_secret_attacks
        self.attack_name = attack_name
        self.original_columns = df_original.columns.tolist()
        logger_path = os.path.join(results_path, 'brm_attack.log')
        file_level = logging.INFO
        if verbose:
            file_level = logging.DEBUG
        self.no_counter = no_counter
        self.logger = setup_logging(log_file_path=logger_path, file_level=file_level)
        self.logger.info(f"Original DataFrame shape: {df_original.shape}")
        self.logger.info(f"Original columns: {self.original_columns}")

        attack_tags = {'type': 'brm_attack',
                       'max_known_col_sets': max_known_col_sets,
                       'known_cols_sets_unique_threshold': known_cols_sets_unique_threshold,
                       'num_per_secret_attacks': num_per_secret_attacks,
                       'max_num_anon_datasets': max_num_anon_datasets,
        }
        if not isinstance(additional_tags, dict):
            raise TypeError("additional_tags must be a dictionary composed of key:value pairs.")
        overlap = set(attack_tags) & set(additional_tags)
        if overlap:
            raise KeyError(f"Duplicate keys found in additional_tags: {overlap}. Please do not use these keys.")
        attack_tags.update(additional_tags)

        self.alcm = ALCManager(df_original,
                               anon,
                               results_path = self.results_path,
                               attack_name = self.attack_name,
                               attack_tags=attack_tags,
                               logger=self.logger,
                               prior_experiment_swap_fraction=self.prior_experiment_swap_fraction,
                               flush=self.flush)
        # The known columns are the pre-discretized continuous columns and categorical
        # columns (i.e. all original columns). The secret columns are the discretized
        # continuous columns and categorical columns.
        self.all_known_columns = self.original_columns
        self.all_secret_columns = [self.alcm.get_discretized_column(col) for col in self.original_columns]
        # Used in any given attack loop
        self.logger.info(f"There are {len(self.all_known_columns)} potential known columns:")
        self.logger.info(self.all_known_columns)
        self.logger.info(f"There are {len(self.all_secret_columns)} potential secret columns:")
        self.logger.info(self.all_secret_columns)
        self.logger.info(f"Columns in first 5 of {len(self.alcm.df.anon)} anonymized dataframes:")
        for df in self.alcm.df.anon[:5]:
            self.logger.info(f"    {df.columns.tolist()}")
        self.logger.info("Columns are classified as:")
        self.logger.info(pp.pformat(self.alcm.get_column_classification_dict()))

    def run_all_columns_attack(self, secret_columns: List[str] = None) -> None:
        '''
        Runs attacks assuming all columns except secret are known
        '''
        if secret_columns is None:
            secret_columns = self.all_secret_columns
        for secret_column in secret_columns:
            known_columns = [col for col in self.all_known_columns if col != self.alcm.get_pre_discretized_column(secret_column)]
            self.run_one_attack(secret_column, known_columns)

    def run_one_attack(self, secret_column: str, known_columns: List[str] = None) -> None:
        # We do this again in case this is being called from the user
        secret_column = self.alcm.get_discretized_column(secret_column)
        if known_columns is None:
            # select a set of original rows to use for the attack
            known_columns = self.all_known_columns
        self.logger.info(f"\nAttack secret column {secret_column}\n    assuming {len(known_columns)} known columns {known_columns}")
        counter = 1
        last_alc = None
        last_reason = None
        for atk_row, _, _ in self.alcm.predictor(known_columns, secret_column):
            # Note that atk_row contains only the known_columns
            encoded_predicted_value, prediction_confidence = self._best_row_attack(atk_row, secret_column)
            if encoded_predicted_value is None:
                # Can happen if there are no candidates for the secret column
                self.alcm.abstention()
            else:
                self.alcm.prediction(encoded_predicted_value, prediction_confidence)
            counter += 1
            if self.no_counter is False:
                if 'alc' in self.alcm.halt_info:
                    last_alc = self.alcm.halt_info['alc']
                if last_reason is None:
                    last_reason = self.alcm.halt_info['reason']
                else:
                    if last_reason != self.alcm.halt_info['reason']:
                        last_reason = self.alcm.halt_info['reason']
                        print('')
                if last_alc is not None:
                    print(f"\r{counter} alc:{last_alc} {self.alcm.halt_info['reason']}", end="")
                else:
                    print(f"\r{counter} {self.alcm.halt_info['reason']}", end="")
        if self.no_counter is False:
            print("\r", end="")
        self.logger.info(f'''\n   Finished after {self.alcm.halt_info['num_attacks']} attacks with ALC {self.alcm.halt_info['alc'] if 'alc' in self.alcm.halt_info else 'unknown'} for reason "{self.alcm.halt_info['reason']}"''')


        self.alcm.summarize_results(with_plot=True)

    def run_auto_attack(self, secret_columns: List[str] = None, known_columns: List[str] = None) -> None:
        '''
        Runs attacks against all secret columns for a variety of known columns
        '''
        if self.results_path is None:
            # raise a value error
            raise ValueError("results_path must be set")
        if known_columns is None:
            # select a set of original rows to use for the attack
            known_columns = self.all_known_columns
        if secret_columns is None:
            secret_columns = self.all_secret_columns
        known_column_sets = get_good_known_column_sets(self.alcm.df.orig_all, known_columns, max_sets = self.max_known_col_sets, unique_rows_threshold = self.known_cols_sets_unique_threshold)
        self.logger.info(f"Found {len(known_column_sets)} unique known column sets ")
        min_set_size = min([len(col_set) for col_set in known_column_sets])
        max_set_size = max([len(col_set) for col_set in known_column_sets])
        self.logger.info(f"Minimum set size: {min_set_size}, Maximum set size: {max_set_size}")
        per_secret_column_sets = {}
        max_col_set_size = 0
        for secret_column in secret_columns:
            valid_known_column_sets = [col_set for col_set in known_column_sets if self.alcm.get_pre_discretized_column(secret_column) not in col_set]
            self.logger.info(f"For secret_column {secret_column}, found {len(valid_known_column_sets)} valid known column sets")
            sampled_known_column_sets = random.sample(valid_known_column_sets,
                                              min(self.num_per_secret_attacks, len(valid_known_column_sets)))
            self.logger.info(f"Selected {len(sampled_known_column_sets)} sampled known column sets")
            max_col_set_size = max(max_col_set_size, len(sampled_known_column_sets))
            per_secret_column_sets[secret_column] = {'known_column_sets': sampled_known_column_sets}
        for i in range(max_col_set_size):
            for secret_column, info in per_secret_column_sets.items():
                if i >= len(info['known_column_sets']):
                    continue
                known_columns = list(info['known_column_sets'][i])
                self.run_one_attack(secret_column, known_columns)


    def _best_row_attack(self, row: pd.DataFrame,
                          secret_column: str) -> Tuple[Any, float]:
        min_gower_distance, secret_values = find_best_matches(anon=self.alcm.df.anon,
                                                    df_query=row,
                                                    secret_column=secret_column,
                                                    column_classifications=self.alcm.get_column_classification_dict(),
                                                    match_method=self.match_method,
                                                    max_num_anon_datasets=self.max_num_anon_datasets,
                                                    )
        number_of_min_gower_distance_matches = len(secret_values)
        if number_of_min_gower_distance_matches == 0:
            return None, 0.0
        pred_value, modal_count = modal_fraction(secret_values)
        modal_frac = modal_count / number_of_min_gower_distance_matches
        confidence = best_match_confidence(
                                        gower_distance=min_gower_distance,
                                        modal_fraction=modal_frac,
                                        match_count=number_of_min_gower_distance_matches)
        return pred_value, confidence

