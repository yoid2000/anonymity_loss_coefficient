import os
from typing import Optional, Dict, List, Union, Any, Tuple, Generator
import numpy as np
import pandas as pd
import logging
import random
from .reporting import *
from .data_files import DataFiles
from .baseline_predictor import BaselinePredictor
from .score_interval import ScoreInterval
from .anonymity_loss_coefficient import AnonymityLossCoefficient
from .reporting import Reporter
from .defaults import defaults
from anonymity_loss_coefficient.utils import setup_logging
import pprint
pp = pprint.PrettyPrinter(indent=4)


class ALCManager:
    def __init__(self, df_original: pd.DataFrame,
                       df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                       results_path: str,
                       attack_name: str = '',
                       logger: logging.Logger = None,
                       flush: bool = defaults['flush'],
                       disc_max: int = defaults['disc_max'],
                       disc_bins: int = defaults['disc_bins'],
                       discretize_in_place: bool = defaults['discretize_in_place'],
                       si_type: str = defaults['si_type'],
                       si_confidence: float = defaults['si_confidence'],
                       max_score_interval: float = defaults['max_score_interval'],
                       halt_thresh_low = defaults['halt_thresh_low'],
                       halt_thresh_high = defaults['halt_thresh_high'],
                       halt_interval_thresh = defaults['halt_interval_thresh'],
                       halt_min_significant_attack_prcs = defaults['halt_min_significant_attack_prcs'],
                       halt_min_prc_improvement = defaults['halt_min_prc_improvement'],
                       halt_check_count = defaults['halt_check_count'],
                       random_state: Optional[int] = None
                       ) -> None:
        self.df = DataFiles(
                 df_original=df_original,
                 df_synthetic=df_synthetic,
                 disc_max=disc_max,
                 disc_bins=disc_bins,
                 discretize_in_place=discretize_in_place,
                 random_state=random_state,
        )
        self.logger = logger
        if self.logger is None:
            logger_path = os.path.join(results_path, 'alc_manager.log')
            self.logger = setup_logging(log_file_path=logger_path)
        self.base_pred = BaselinePredictor()
        self.alc = AnonymityLossCoefficient()
        self.random_state = random_state
        self.max_score_interval = max_score_interval
        self.halt_thresh_low = halt_thresh_low
        self.halt_thresh_high = halt_thresh_high
        self.halt_interval_thresh = halt_interval_thresh
        self.halt_min_significant_attack_prcs = halt_min_significant_attack_prcs
        self.halt_min_prc_improvement = halt_min_prc_improvement
        self.halt_check_count = halt_check_count
        self.si_confidence = si_confidence
        self.si_type = si_type
        self.si_secret = ''
        self.si_known_columns = []
        self.attack_in_progress = False

        self.rep = Reporter(results_path=results_path,
                            attack_name=attack_name,
                            logger=self.logger,
                            flush=flush,)

        # These three are used by the predictor to obtain the attacker's prediction
        # and confidence
        self.prediction_made = False
        self.encoded_predicted_value = None
        self.prediction_confidence = None
        # This is the true value that predicted_value is compared against
        self.encoded_true_value = None
        # This is the True/False result of the prediction
        self.prediction_result = None
        # These contain information that the attacker can use to determine
        # why an attack loop halted
        self.halt_info = None

    def predictor(self,
                  known_columns: List[str],
                  secret_col: str,
                  ) -> Generator[Tuple[pd.DataFrame, Any, Any, bool], None, None]:
        """
        This is the main method of the ALCManager class. It is a generator that yields
        the rows to use in an attack. It makes one baseline prediction per loop. The
        caller must make an attack prediction in each loop. This method determines
        when enough predictions have been made to produce a good ALC score.
        """
        # First check if we have already run this attack.
        if self.rep.already_attacked(secret_col, known_columns):
            self.logger.info(f"Already ran attack on {secret_col} with known columns {known_columns}. Skipping.")
            self.halt_info = {'halted': True, 'reason': 'attack already run. skipping.', 'num_attacks': 0}
            return

        # Establish the targets to ignore, if any, and make a ScoreInterval object
        # for the halting decision.
        ignore_value, ignore_fraction = self._get_target_to_ignore_for_halting(secret_col)
        if ignore_value is not None:
            decoded_ignore_value = self.decode_value(secret_col, ignore_value)
            self.logger.info(f"The value {decoded_ignore_value} constitutes {round((100*(1-ignore_fraction)),2)} % of column {secret_col}, so we'll ignore a proportional fraction of those values in the attacks so that our results are better balanced.")
        si_halt = ScoreInterval(si_type=self.si_type, si_confidence=self.si_confidence, max_score_interval=self.max_score_interval, logger=self.logger)

        # Initialize the first set of control rows
        self._init_cntl_and_build_model(known_columns, secret_col)

        num_attacks = 0
        self.halt_info = {'halted': False, 'reason': 'halt not yet checked', 'num_attacks': 1}
        self.attack_in_progress = True
        while True:
            for i in range(len(self.df.cntl)):
                # Get one base and attack measure at a time, and continue until we have
                # enough confidence in the results
                atk_row = self.df.cntl.iloc[[i]]

                # For the purpose of determining if the attack prediction is True or False,
                # we determine the true value. The decoded_true_value is the true value 
                # after decoding from the encoding done in pre-processing
                self.encoded_true_value = atk_row[secret_col].iloc[0]
                decoded_true_value = self.decode_value(secret_col, self.encoded_true_value)

                # Determine if the row should be ignored or not for the purpose of
                # the halting criteria.
                if (ignore_value is not None and
                    self.encoded_true_value == ignore_value and
                    random.random() > ignore_fraction):
                    continue
                num_attacks += 1
                self._model_prediction(atk_row, secret_col, known_columns, si_halt)

                # The attacker only needs to know the values of the known_columns for the
                # purpose of running the attack, so we separate them out.
                df_query = atk_row[known_columns]
                self.prediction_made = False
                yield df_query, self.encoded_true_value, decoded_true_value

                if self.prediction_made is False:
                    raise ValueError("Error: No prediction was made in the predictor method. The caller must call the ALCManager prediction() method in each loop of the predictor.")
                self.prediction_made = False
                if self.encoded_predicted_value is not None:
                    decoded_predicted_value = self.decode_value(secret_col, self.encoded_predicted_value)
                else:
                    decoded_predicted_value = None
                self._add_result(predict_type='attack',
                                known_columns=known_columns,
                                secret_col=secret_col,
                                decoded_predicted_value=decoded_predicted_value,
                                decoded_true_value=decoded_true_value,
                                encoded_predicted_value=self.encoded_predicted_value,
                                encoded_true_value=self.encoded_true_value,
                                attack_confidence=self.prediction_confidence,
                                si_halt=si_halt)
                if num_attacks % self.halt_check_count == 0:
                    # This check is a bit expensive, so by default don't do it every time
                    self.halt_info = self._ok_to_halt(si_halt)
                self.halt_info.update({'num_attacks': num_attacks})
                self.logger.debug(pp.pformat(self.halt_info))
                if self.halt_info['halted'] is True:
                    self.attack_in_progress = False
                    self.rep.consolidate_results(si_halt.get_alc_scores())
                    return
            is_assigned = self._next_cntl_and_build_model()
            if is_assigned is False:
                self.halt_info = {'halted': False, 'reason': 'exhausted all rows',  'num_attacks': num_attacks}
                self.attack_in_progress = False
                self.rep.consolidate_results(si_halt.get_alc_scores())
                return

    def prediction(self, encoded_predicted_value: Any, prediction_confidence: float) -> bool:
        self.prediction_made = True
        self.encoded_predicted_value = encoded_predicted_value
        self.prediction_confidence = prediction_confidence
        if self.encoded_predicted_value == self.encoded_true_value:
            self.prediction_result = True
        else:
            self.prediction_result = False
        # We return the result of the prediction to the caller as a convenience.
        # This allows the caller to ensure that it agrees with the result.
        return self.prediction_result

    def _model_prediction(self, row: pd.DataFrame,
                     secret_col: str,
                     known_columns: List[str],
                     si_halt: Optional[ScoreInterval]) -> None:
        # get the prediction for the row
        df_row = row[known_columns]  # This is already a DataFrame
        encoded_predicted_value, proba = self.predict(df_row)
        encoded_true_value = row[secret_col].iloc[0]
        decoded_predicted_value = self.decode_value(secret_col, encoded_predicted_value)
        decoded_true_value = self.decode_value(secret_col, encoded_true_value)
        self._add_result(predict_type='base',
                         known_columns=known_columns,
                         secret_col=secret_col,
                         decoded_predicted_value=decoded_predicted_value,
                         decoded_true_value=decoded_true_value,
                         encoded_predicted_value=encoded_predicted_value,
                         encoded_true_value=encoded_true_value,
                         base_confidence=proba,
                         si_halt=si_halt)

    def _get_target_to_ignore_for_halting(self, column: str) -> Tuple[Optional[Any], float]:
        """
        With respect to the halting decision, we want to ignore column values that are
        too common because then we won't get an adequate sampling of other column values.
        We put the threshold above 0.5 because we don't want to ignore a value in a
        well-balanced binary column.

        This version computes the normalized count of the largest normalized count.
        If that count is > 0.6, it returns the value and (1 - count).
        """
        # Compute normalized value counts
        value_counts = self.df.orig_all[column].value_counts(normalize=True)

        # Find the value with the largest normalized count
        max_value = value_counts.idxmax()
        max_count = value_counts.max()

        # Check if the largest normalized count exceeds the threshold
        if max_count > 0.6:
            return max_value, 1 - max_count

        # If no value exceeds the threshold, return None
        return None, 0.0
    
    
    def summarize_results(self,
                          strong_thresh: float = 0.5,
                          risk_thresh: float = 0.7,
                          with_text: bool = True,
                          with_plot: bool = True,
                          ) -> None:
        if self.attack_in_progress:
            self.logger.warning("Warning: Attack is still in progress. Summarize aborted.")
            return False
        return self.rep.summarize_results(strong_thresh=strong_thresh,
                                          risk_thresh=risk_thresh,
                                          with_text=with_text,
                                          with_plot=with_plot,
                                          )
    def _add_result(self,
                   predict_type: str,
                   known_columns: List[str],
                   secret_col: str,
                   decoded_predicted_value: Any,
                   decoded_true_value: Any,
                   encoded_predicted_value: Any,
                   encoded_true_value: Any,
                   base_confidence: Optional[float] = None,
                   attack_confidence: Optional[float] = None,
                   si_halt: Optional[ScoreInterval] = None,
                   ) -> None:

        if base_confidence is not None and base_confidence == 0:
            base_confidence = None
        if attack_confidence is not None and attack_confidence == 0:
            attack_confidence = None
        self.rep.add_known_columns(known_columns)
        self.rep.add_secret_column(secret_col)
        # sort known_columns
        known_columns = sorted(known_columns)
        # Check if predicted_value is a numpy type
        if isinstance(decoded_predicted_value, np.generic):
            decoded_predicted_value = decoded_predicted_value.item()
        if decoded_predicted_value == decoded_true_value:
            prediction = True
        else:
            prediction = False

        # Create a new row as a dictionary. The str() typecasting is to deal with
        # saving as parquet later, which requires consistent types
        row = {
            'predict_type': predict_type,
            'known_columns': self.rep._make_known_columns_str(known_columns),
            'num_known_columns': len(known_columns),
            'secret_column': secret_col,
            'predicted_value': str(decoded_predicted_value),
            'true_value': str(decoded_true_value),
            'encoded_predicted_value': str(encoded_predicted_value),
            'encoded_true_value': str(encoded_true_value),
            'prediction': prediction,
            'base_confidence': base_confidence,
            'attack_confidence': attack_confidence,
        }

        self.rep.add_result(row)

        confidence = base_confidence
        if predict_type == 'attack':
            confidence = attack_confidence
        si_halt.add_prediction(prediction, confidence, predict_type)

    def _get_significant_attack_prcs(self, alc_scores: List[Dict[str, Any]]) -> List[float]:
        attack_prcs = []
        alc_scores = sorted(alc_scores, key=lambda x: x['attack_recall'], reverse=True)
        for score in alc_scores:
            if score['attack_si'] > self.halt_interval_thresh:
                return attack_prcs
            attack_prcs.append(score['attack_prc'])
        return attack_prcs

    def _ok_to_halt(self, si_halt: ScoreInterval) -> Dict[str, Any]:
        if len(si_halt.df_base) < 10 or len(si_halt.df_attack) < 10:
            return {'halted': False, 'reason': f'not enough samples {len(si_halt.df_base)} base, {len(si_halt.df_attack)} attack'}
        alc_scores = si_halt.get_alc_scores()
        # put the values of alc_scores['paired'] into a list called paired
        if len(alc_scores) == 0:
            return {'halted': False, 'reason':f'no alc scores with attack and base score intervals < {self.max_score_interval}'}
        best_alc_score, alc_scores = si_halt.split_scores(alc_scores)
        if best_alc_score is None:
            return {'halted': False, 'reason':f'no prc scores with attack and base score intervals < {self.halt_interval_thresh}'}

        ret = best_alc_score
        if ret['alc_high'] < self.halt_thresh_low:
            ret.update({'halted':True, 'reason':'alc extremely low'})
            return ret
        if ret['alc_low'] > self.halt_thresh_high:
            ret.update({'halted':True, 'reason':'alc extremely high'})
            return ret
        # We didn't halt because of extreme high or low ALC, so now we want to determine
        # if further attacks are likely to yield a better ALC.
        # Sort alc_scores from highest to lowest 'attack_recall'
        attack_prcs = self._get_significant_attack_prcs(alc_scores)
        if len(attack_prcs) == len(alc_scores) and len(si_halt.df_base) > 200 or len(si_halt.df_attack) > 200:
            # This occurs if all alc_scores are significant (have attack_si and
            # base_si < halt_interval_thresh). Further attacks will only normally reduce
            # the already measured precision score intervals, not yield still lower
            # recall values.
            ret.update({'halted':True, 'reason':f'all {len(attack_prcs)} attack prc measures significant'})
            return ret
        if len(attack_prcs) < self.halt_min_significant_attack_prcs:
            ret.update({'halted':False, 'reason':f'too few significant attack prc measures ({len(attack_prcs)})'})
            return ret
        if attack_prcs[-1] - attack_prcs[-2] < self.halt_min_prc_improvement:
            ret.update({'halted':True, 'reason':f'attack prc measures not improving more than {self.halt_min_prc_improvement}'})
            return ret
        ret.update({'halted':False, 'reason':'halt conditions not met'})
        return ret

    # Following are the methods that use DataFiles
    def get_pre_discretized_column(self, secret_column: str) -> str:
        return self.df.get_pre_discretized_column(secret_column)

    def get_discretized_column(self, secret_column: str) -> str:
        return self.df.get_discretized_column(secret_column)

    def decode_value(self, column: str, encoded_value: int) -> Any:
        return self.df.decode_value(column, encoded_value)

    def get_column_classification(self, column: str) -> str:
        return self.df.get_column_classification(column)

    def get_column_classification_dict(self) -> Dict:
        return self.df.column_classification.copy()

    # Following are the methods that use BasePredictor 
    def _init_cntl_and_build_model(self, known_columns: List[str], secret_col: str,  
                                  ) -> None:
        is_assigned = self.df.assign_first_cntl_block()
        if is_assigned is False:
            raise ValueError("Error: Control block initialization failed")
        self.base_pred.build_model(self.df.orig, known_columns, secret_col, self.random_state)

    def _next_cntl_and_build_model(self) -> bool:
        is_assigned = self.df.assign_next_cntl_block()
        if is_assigned is False:
            return False
        self.base_pred.build_model(self.df.orig)
        return True

    def predict(self, df_row: pd.DataFrame) -> Tuple[Any, float]:
        return self.base_pred.predict(df_row)

    # Following are methods that use Reporter
    def get_results_df(self,
                       known_columns: Optional[List[str]] = None,
                       secret_column: Optional[str] = None) -> Optional[pd.DataFrame]:
        if self.attack_in_progress:
            self.logger.warning("Warning: Attack is still in progress. Cannot get results.")
            return None
        return self.rep.get_results_df(known_columns, secret_column)

    def alc_per_secret_and_known_df(self,
                                 known_columns: Optional[List[str]] = None,
                                 secret_column: Optional[str] = None) -> Optional[pd.DataFrame]:
        if self.attack_in_progress:
            self.logger.warning("Warning: Attack is still in progress. Cannot get results.")
            return None
        return self.rep.alc_per_secret_and_known_df(known_columns, secret_column)