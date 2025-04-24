import os
import pandas as pd
import random
from typing import List, Union, Any, Tuple
from itertools import combinations
import logging
from anonymity_loss_coefficient.alc.alc_manager import ALCManager
from .matching_routines import find_best_matches, modal_fraction, best_match_confidence
from anonymity_loss_coefficient.utils import get_good_known_column_sets, setup_logging
import pprint

pp = pprint.PrettyPrinter(indent=4)

class BrmAttack:
    def __init__(self,
                 df_original: pd.DataFrame,
                 df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                 results_path: str = None,
                 max_known_col_sets: int = 1000,
                 known_cols_sets_unique_threshold: float = 0.45,
                 num_per_secret_attacks: int = 100,
                 max_rows_per_attack: int = 500,
                 min_positive_predictions: int = 5,
                 confidence_interval_tolerance: float = 0.1,
                 confidence_level: float = 0.95,
                 attack_name: str = '',
                 verbose: bool = False,
                 no_counter: bool = True,
                 ) -> None:
        # up to work with ML modeling
        self.results_path = results_path
        self.max_known_col_sets = max_known_col_sets
        self.known_cols_sets_unique_threshold = known_cols_sets_unique_threshold
        self.num_per_secret_attacks = num_per_secret_attacks
        self.max_rows_per_attack = max_rows_per_attack
        self.min_positive_predictions = min_positive_predictions
        self.confidence_interval_tolerance = confidence_interval_tolerance
        self.confidence_level = confidence_level
        self.attack_name = attack_name
        self.original_columns = df_original.columns.tolist()
        logger_path = os.path.join(results_path, 'brm_attack.log')
        file_level = logging.INFO
        if verbose:
            file_level = logging.DEBUG
        self.no_counter = no_counter
        self.logger = setup_logging(log_file_path=logger_path, file_level=file_level)
        self.logger.info(f"Original columns: {self.original_columns}")
        self.alcm = ALCManager(df_original, df_synthetic, logger=self.logger,)
        # The known columns are the pre-discretized continuous columns and categorical
        # columns (i.e. all original columns). The secret columns are the discretized
        # continuous columns and categorical columns.
        self.all_known_columns = self.original_columns
        self.all_secret_cols = [self.alcm.get_discretized_column(col) for col in self.original_columns]
        self.logger.info(f"There are {len(self.all_known_columns)} potential known columns:")
        self.logger.info(self.all_known_columns)
        self.logger.info(f"There are {len(self.all_secret_cols)} potential secret columns:")
        self.logger.info(self.all_secret_cols)
        self.logger.info("Columns are classified as:")
        self.logger.info(pp.pformat(self.alcm.get_column_classification_dict()))

    def run_all_columns_attack(self, secret_cols: List[str] = None) -> None:
        '''
        Runs attacks assuming all columns except secret are known
        '''
        if secret_cols is None:
            secret_cols = self.all_secret_cols
        for secret_col in secret_cols:
            known_columns = [col for col in self.all_known_columns if col != self.alcm.get_pre_discretized_column(secret_col)]
            self.run_one_attack(secret_col, known_columns)

    def run_one_attack(self, secret_col: str, known_columns: List[str] = None) -> None:
        if known_columns is None:
            # select a set of original rows to use for the attack
            known_columns = self.all_known_columns
        self.logger.info(f"\nAttack secret column {secret_col}\n    assuming {len(known_columns)} known columns {known_columns}")
        counter = 1
        for atk_row, _, _ in self.alcm.predictor(known_columns, secret_col):
            encoded_predicted_value, prediction_confidence = self._best_row_attack(atk_row, secret_col, known_columns)
            self.alcm.prediction(encoded_predicted_value, prediction_confidence)
            counter += 1
            if counter % 100 == 0:
                self.alcm.summarize_results(results_path = self.results_path,
                                        attack_name = self.attack_name, with_plot=False, with_text=False)
            if self.no_counter is False:
                if 'alc' in self.alcm.halt_info:
                    print(f"\r{counter} alc:{self.alcm.halt_info['alc']} {self.alcm.halt_info['reason']}", end="")
                else:
                    print(f"\r{counter} {self.alcm.halt_info['reason']}", end="")
        if self.no_counter is False:
            print("\r", end="")
        self.logger.info(f'''   Finished after {self.alcm.halt_info['num_attacks']} attacks with ALC {self.alcm.halt_info['alc'] if 'alc' in self.alcm.halt_info else 'unknown'} for reason "{self.alcm.halt_info['reason']}"''')


        self.alcm.summarize_results(results_path = self.results_path,
                                        attack_name = self.attack_name, with_plot=True)

    def run_auto_attack(self, secret_cols: List[str] = None, known_columns: List[str] = None) -> None:
        '''
        Runs attacks against all secret columns for a variety of known columns
        '''
        if self.results_path is None:
            # raise a value error
            raise ValueError("results_path must be set")
        if known_columns is None:
            # select a set of original rows to use for the attack
            known_columns = self.all_known_columns
        if secret_cols is None:
            secret_cols = self.all_secret_cols
        known_column_sets = get_good_known_column_sets(self.alcm.df.orig_all, known_columns, max_sets = self.max_known_col_sets, unique_rows_threshold = self.known_cols_sets_unique_threshold)
        self.logger.info(f"Found {len(known_column_sets)} unique known column sets ")
        min_set_size = min([len(col_set) for col_set in known_column_sets])
        max_set_size = max([len(col_set) for col_set in known_column_sets])
        self.logger.info(f"Minimum set size: {min_set_size}, Maximum set size: {max_set_size}")
        per_secret_column_sets = {}
        max_col_set_size = 0
        for secret_col in secret_cols:
            valid_known_column_sets = [col_set for col_set in known_column_sets if self.alcm.get_pre_discretized_column(secret_col) not in col_set]
            self.logger.info(f"For secret_col {secret_col}, found {len(valid_known_column_sets)} valid known column sets")
            sampled_known_column_sets = random.sample(valid_known_column_sets,
                                              min(self.num_per_secret_attacks, len(valid_known_column_sets)))
            self.logger.info(f"Selected {len(sampled_known_column_sets)} sampled known column sets")
            max_col_set_size = max(max_col_set_size, len(sampled_known_column_sets))
            per_secret_column_sets[secret_col] = {'known_column_sets': sampled_known_column_sets}
        for i in range(max_col_set_size):
            for secret_col, info in per_secret_column_sets.items():
                if i >= len(info['known_column_sets']):
                    continue
                known_columns = list(info['known_column_sets'][i])
                self.run_one_attack(secret_col, known_columns)


    def _best_row_attack(self, row: pd.DataFrame,
                          secret_col: str,
                          known_columns: List[str]) -> Tuple[Any, float]:
        best_confidence = -1
        best_pred_value = None
        for df_syn in self.alcm.df.syn_list:
            # Check if secret_col is in df_syn
            if secret_col not in df_syn.columns:
                continue
            # Make sure there is at least one known column in df_syn
            shared_known_columns = list(set(known_columns) & set(df_syn.columns))
            if len(shared_known_columns) == 0:
                continue
            df_query = row[shared_known_columns]
            idx, min_gower_distance = find_best_matches(df_query=df_query,
                                                        df_candidates=df_syn,
                                                        column_classifications=self.alcm.get_column_classification_dict(),
                                                        columns=shared_known_columns,
                                                        debug_on=False)
            number_of_min_gower_distance_matches = len(idx)
            this_pred_value, modal_count = modal_fraction(df_candidates=df_syn,
                                                     idx=idx, column=secret_col)
            this_modal_fraction = modal_count / number_of_min_gower_distance_matches
            this_confidence = best_match_confidence(
                                          gower_distance=min_gower_distance,
                                          modal_fraction=this_modal_fraction,
                                          match_count=number_of_min_gower_distance_matches)
            if this_confidence > best_confidence:
                best_confidence = this_confidence
                best_pred_value = this_pred_value
        return best_pred_value, best_confidence

