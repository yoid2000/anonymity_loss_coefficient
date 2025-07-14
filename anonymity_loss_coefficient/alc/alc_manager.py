import os
from typing import Optional, Dict, List, Union, Any, Tuple, Generator
import numpy as np
import pandas as pd
import logging
import random
import time
from .reporting import *
from .data_files import DataFiles
from .baseline_predictor import BaselinePredictor
from .score_interval import ScoreInterval
from .anonymity_loss_coefficient import AnonymityLossCoefficient
from .reporting import Reporter
from .params import ALCParams
from anonymity_loss_coefficient.utils import setup_logging
import pprint
pp = pprint.PrettyPrinter(indent=4)


class ALCManager:
    def __init__(self, df_original: pd.DataFrame,
                       anon: Union[pd.DataFrame, List[pd.DataFrame]],
                       results_path: str,
                       attack_name: str = '',
                       logger: Optional[logging.Logger] = None,
                       flush: bool = False,
                       attack_tags: Optional[Dict[str, Any]] = None,
                       # ALCManager parameters
                       halt_thresh_low: Optional[float] = None,
                       halt_thresh_high: Optional[float] = None,
                       halt_interval_thresh: Optional[float] = None,
                       halt_min_significant_attack_prcs: Optional[int] = None,
                       halt_min_prc_improvement: Optional[float] = None,
                       halt_check_count: Optional[int] = None,
                       # AnonymityLossCoefficient parameters
                       prc_abs_weight: Optional[float] = None,
                       recall_adjust_min_intercept: Optional[float] = None,
                       recall_adjust_strength: Optional[float] = None,
                       # DataFiles parameters
                       disc_max: Optional[int] = None,
                       disc_bins: Optional[int] = None,
                       discretize_in_place: Optional[bool] = None,
                       max_cntl_size: Optional[int] = None,
                       max_cntl_percent: Optional[float] = None,
                       # ScoreInterval parameters
                       si_type: Optional[str] = None,
                       si_confidence: Optional[float] = None,
                       max_score_interval: Optional[float] = None,

                       prior_experiment_swap_fraction: float = -1.0,
                       random_state: Optional[int] = None
                       ) -> None:
        self.alcp = ALCParams()
        self.alcp.set_param(self.alcp.alcm, 'halt_thresh_low', halt_thresh_low)
        self.alcp.set_param(self.alcp.alcm, 'halt_thresh_high', halt_thresh_high)
        self.alcp.set_param(self.alcp.alcm, 'halt_interval_thresh', halt_interval_thresh)
        self.alcp.set_param(self.alcp.alcm, 'halt_min_significant_attack_prcs', halt_min_significant_attack_prcs)
        self.alcp.set_param(self.alcp.alcm, 'halt_min_prc_improvement', halt_min_prc_improvement)
        self.alcp.set_param(self.alcp.alcm, 'halt_check_count', halt_check_count)

        self.alcp.set_param(self.alcp.alc, 'prc_abs_weight', prc_abs_weight)
        self.alcp.set_param(self.alcp.alc, 'recall_adjust_min_intercept', recall_adjust_min_intercept)
        self.alcp.set_param(self.alcp.alc, 'recall_adjust_strength', recall_adjust_strength)

        self.alcp.set_param(self.alcp.si, 'si_type', si_type)
        self.alcp.set_param(self.alcp.si, 'si_confidence', si_confidence)
        self.alcp.set_param(self.alcp.si, 'max_score_interval', max_score_interval)

        self.alcp.set_param(self.alcp.df, 'disc_max', disc_max)
        self.alcp.set_param(self.alcp.df, 'disc_bins', disc_bins)
        self.alcp.set_param(self.alcp.df, 'discretize_in_place', discretize_in_place)
        self.alcp.set_param(self.alcp.df, 'max_cntl_size', max_cntl_size)
        self.alcp.set_param(self.alcp.df, 'max_cntl_percent', max_cntl_percent)

        if attack_tags is not None:
            # add attack parameters to later record
            self.alcp.add_group('atk')
            for k, v in attack_tags.items():
                self.alcp.set_param(self.alcp.atk, k, v)

        self.df = DataFiles(
                 df_original=df_original,
                 anon=anon,
                 disc_max=self.alcp.df.disc_max,
                 disc_bins=self.alcp.df.disc_bins,
                 discretize_in_place=self.alcp.df.discretize_in_place,
                 max_cntl_size=self.alcp.df.max_cntl_size,
                 max_cntl_percent=self.alcp.df.max_cntl_percent,
                 random_state=random_state,
        )
        self.prior_experiment_swap_fraction = prior_experiment_swap_fraction     # experimental purposes
        self.logger = logger
        if self.logger is None:
            logger_path = os.path.join(results_path, 'alc_manager.log')
            self.logger = setup_logging(log_file_path=logger_path)
        self.base_pred = BaselinePredictor(logger=self.logger)
        self.alc = AnonymityLossCoefficient(
            prc_abs_weight=self.alcp.alc.prc_abs_weight,
            recall_adjust_min_intercept=self.alcp.alc.recall_adjust_min_intercept,
            recall_adjust_strength=self.alcp.alc.recall_adjust_strength,
        )
        self.model_name = None
        self.random_state = random_state
        self.max_score_interval = self.alcp.si.max_score_interval
        self.halt_thresh_low = self.alcp.alcm.halt_thresh_low
        self.halt_thresh_high = self.alcp.alcm.halt_thresh_high
        self.halt_interval_thresh = self.alcp.alcm.halt_interval_thresh
        self.halt_min_significant_attack_prcs = self.alcp.alcm.halt_min_significant_attack_prcs
        self.halt_min_prc_improvement = self.alcp.alcm.halt_min_prc_improvement
        self.halt_check_count = self.alcp.alcm.halt_check_count
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
        # This is used when an abstention is made, and the baseline prediction is used
        self.encoded_predicted_value_model = None
        # These contain information that the attacker can use to determine
        # why an attack loop halted
        self.halt_info = None
        self.do_early_halt = False
        # Other
        self.start_time = None

    def close_logger(self):
        """Closes all handlers attached to the logger."""
        if hasattr(self, 'logger') and self.logger:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()

    def predictor(self,
                  known_columns: List[str],
                  secret_column: str,
                  ) -> Generator[Tuple[pd.DataFrame, Any, Any, bool], None, None]:
        """
        This is the main method of the ALCManager class. It is a generator that yields
        the rows to use in an attack. It makes one baseline prediction per loop. The
        caller must make an attack prediction in each loop. This method determines
        when enough predictions have been made to produce a good ALC score.
        """
        self.start_time = time.time()
        self.do_early_halt = False
        self.num_prc_measures = self.halt_min_significant_attack_prcs
        # First check if we have already run this attack.
        if self.rep.already_attacked(secret_column, known_columns):
            self.logger.info(f"Already ran attack on {secret_column} with known columns {known_columns}. Skipping.")
            self.halt_info = {'halted': True, 'reason': 'attack already run. skipping.', 'num_attacks': 0, 'halt_code': 'skip'}
            return

        # Establish the targets to ignore, if any, and make a ScoreInterval object
        # for the halting decision.
        ignore_value, ignore_fraction = self._get_target_to_ignore_for_halting(secret_column)
        if ignore_value is not None:
            if ignore_fraction == 0.0:
                self.logger.info(f"Only one value in {secret_column}, so halt.")
                self.halt_info = {'halted': True, 'reason': 'Only one value to attack. skipping.', 'num_attacks': 0, 'halt_code': 'one_value'}
                return
            decoded_ignore_value = self.decode_value(secret_column, ignore_value)
            self.logger.info(f"The value {decoded_ignore_value} constitutes {round((100*(1-ignore_fraction)),2)} % of column {secret_column}, so we'll ignore a proportional fraction of those values in the attacks so that our results are better balanced.")
        si_halt = ScoreInterval(si_type=self.alcp.si.si_type,
                                halt_interval_thresh=self.alcp.alcm.halt_interval_thresh,
                                si_confidence=self.alcp.si.si_confidence,
                                max_score_interval=self.alcp.si.max_score_interval,
                                logger=self.logger)

        # Initialize the first set of control rows
        self._init_cntl_and_build_model(known_columns, secret_column)

        num_attacks = 0
        self.halt_info = {'halted': False, 'reason': 'halt not yet checked', 'num_attacks': 1, 'halt_code': 'none'}
        self.attack_in_progress = True
        while True:
            for i in range(len(self.df.cntl)):
                # Get one base and attack measure at a time, and continue until we have
                # enough confidence in the results
                atk_row = self.df.cntl.iloc[[i]]

                # For the purpose of determining if the attack prediction is True or False,
                # we determine the true value. The decoded_true_value is the true value 
                # after decoding from the encoding done in pre-processing
                encoded_true_value = atk_row[secret_column].iloc[0]
                decoded_true_value = self.decode_value(secret_column, encoded_true_value)

                # Determine if the row should be ignored or not for the purpose of
                # the halting criteria.
                if (ignore_value is not None and
                    encoded_true_value == ignore_value and
                    random.random() > ignore_fraction):
                    continue
                num_attacks += 1
                self._model_prediction(atk_row, secret_column, known_columns, si_halt)

                # The attacker only needs to know the values of the known_columns for the
                # purpose of running the attack, so we separate them out.
                df_query = atk_row[known_columns]
                self.prediction_made = False
                yield df_query, encoded_true_value, decoded_true_value

                if self.prediction_made is False:
                    raise ValueError("Error: No prediction or abstention was made in the predictor method. The caller must call the ALCManager prediction() or abstention() method in each loop of the predictor.")
                self.prediction_made = False
                decoded_predicted_value = self.decode_value(secret_column, self.encoded_predicted_value)
                self._add_result(predict_type='attack',
                                known_columns=known_columns,
                                secret_column=secret_column,
                                decoded_predicted_value=decoded_predicted_value,
                                decoded_true_value=decoded_true_value,
                                encoded_predicted_value=self.encoded_predicted_value,
                                encoded_true_value=encoded_true_value,
                                attack_confidence=self.prediction_confidence,
                                si_halt=si_halt)
                if num_attacks % self.halt_check_count == 0:
                    # This check is a bit expensive, so by default don't do it every time
                    self.halt_info = self._ok_to_halt(si_halt)
                self.halt_info.update({'num_attacks': num_attacks})
                self.logger.debug(pp.pformat(self.halt_info))
                end_time = time.time()
                elapsed_time = round(end_time - self.start_time, 4)
                self.halt_info.update({'elapsed_time': elapsed_time})
                if self.halt_info['halted'] is True:
                    self.attack_in_progress = False
                    self.rep.consolidate_results(si_halt.get_alc_scores(self.num_prc_measures), self.halt_info['halt_code'], self.halt_info['elapsed_time'], self.model_name, self.alcp)
                    return
            is_assigned = self._next_cntl_and_build_model()
            if is_assigned is False:
                end_time = time.time()
                elapsed_time = round(end_time - self.start_time, 4)
                self.halt_info = {'halted': True, 'reason': 'exhausted all rows',  'num_attacks': num_attacks, 'halt_code': 'exhausted', 'elapsed_time': elapsed_time}
                self.attack_in_progress = False
                self.rep.consolidate_results(si_halt.get_alc_scores(self.num_prc_measures), self.halt_info['halt_code'], self.halt_info['elapsed_time'], self.model_name, self.alcp)
                return

    def abstention(self) -> bool:
        ''' In an abstention, we use the baseline prediction as our best guess, and set the
            confidence to 0.0.
        '''
        self.prediction_made = True
        self.encoded_predicted_value = self.encoded_predicted_value_model
        self.prediction_confidence = 0.0


    def prediction(self, encoded_predicted_value: Any, prediction_confidence: float) -> None:
        self.prediction_made = True
        self.encoded_predicted_value = encoded_predicted_value
        self.prediction_confidence = prediction_confidence


    def results(self, known_columns: Optional[List[str]] = None,
                      secret_column: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.alc_per_secret_and_known_df(known_columns, secret_column)

    def prediction_results(self,
                       known_columns: Optional[List[str]] = None,
                       secret_column: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.get_results_df(known_columns, secret_column)

    def _model_prediction(self, row: pd.DataFrame,
                     secret_column: str,
                     known_columns: List[str],
                     si_halt: Optional[ScoreInterval]) -> None:
        # get the prediction for the row
        df_row = row[known_columns]  # This is already a DataFrame
        encoded_predicted_value_model, proba = self.predict(df_row)
        # We save the model prediction in case the caller makes an abstention
        self.encoded_predicted_value_model = encoded_predicted_value_model
        encoded_true_value = row[secret_column].iloc[0]
        decoded_true_value = self.decode_value(secret_column, encoded_true_value)
        if self.prior_experiment_swap_fraction > 0:
            # Purely for experimentation and should not be used otherwise
            encoded_predicted_value_model, _ = _best_row_attack(row, self.df.anon, secret_column, self.get_column_classification_dict())
            proba = 1.0
        decoded_predicted_value_model = self.decode_value(secret_column, encoded_predicted_value_model)
        self._add_result(predict_type='base',
                         known_columns=known_columns,
                         secret_column=secret_column,
                         decoded_predicted_value=decoded_predicted_value_model,
                         decoded_true_value=decoded_true_value,
                         encoded_predicted_value=encoded_predicted_value_model,
                         encoded_true_value=encoded_true_value,
                         base_confidence=proba,
                         si_halt=si_halt)

    def _get_target_to_ignore_for_halting(self, column: str) -> Tuple[Optional[Any], Optional[float]]:
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
        return None, None
    
    
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
                   secret_column: str,
                   decoded_predicted_value: Any,
                   decoded_true_value: Any,
                   encoded_predicted_value: Any,
                   encoded_true_value: Any,
                   base_confidence: Optional[float] = None,
                   attack_confidence: Optional[float] = None,
                   si_halt: Optional[ScoreInterval] = None,
                   ) -> None:

        self.rep.add_known_columns(known_columns)
        self.rep.add_secret_column(secret_column)
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
            'secret_column': secret_column,
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

    def _check_for_early_halt(self, alc_scores: List[Dict[str, Any]]) -> str:
        early_halt_high = 0
        early_halt_low = 0
        for score in alc_scores:
            if score['alc_high'] < self.halt_thresh_low:
                early_halt_low += 1
            elif score['alc_low'] > self.halt_thresh_high:
                early_halt_high += 1
            else:
                return 'none'
        if early_halt_high == len(alc_scores):
            return 'high'
        elif early_halt_low == len(alc_scores):
            return 'low'
        return 'none'

    def _ok_to_halt(self, si_halt: ScoreInterval) -> Dict[str, Any]:
        if len(si_halt.df_base) < 50 or len(si_halt.df_attack) < 50:
            return {'halted': False, 'reason': f'not enough samples {len(si_halt.df_base)} base, {len(si_halt.df_attack)} attack', 'halt_code': 'none'}
        alc_scores = si_halt.get_alc_scores(self.num_prc_measures)
        if self.do_early_halt is False:
            early_halt_status = self._check_for_early_halt(alc_scores)
            if early_halt_status == 'high':
                self.do_early_halt = True
                return {'halted': False, 'reason': 'do extremely high', 'halt_code': 'none'}
            elif early_halt_status == 'low':
                self.do_early_halt = True
                return {'halted': False, 'reason': 'do extremely low', 'halt_code': 'none'}
        # put the values of alc_scores['paired'] into a list called paired
        if len(alc_scores) == 0:
            return {'halted': False, 'reason':f'no alc scores with attack and base score intervals < {self.max_score_interval} (early halt {self.do_early_halt})', 'halt_code': 'none'}
        best_alc_score, alc_scores = si_halt.split_scores(alc_scores)
        if best_alc_score is None:
            return {'halted': False, 'reason':f'no prc scores with attack and base score intervals < {self.halt_interval_thresh} (early halt {self.do_early_halt})', 'halt_code': 'none'}

        ret = best_alc_score
        if self.do_early_halt is True:
            ret.update({'halted':True, 'reason':'early halt', 'halt_code': 'early_halt'})
            return ret
        if ret['alc_high'] < self.halt_thresh_low:
            ret.update({'halted':True, 'reason':'alc extremely low', 'halt_code': 'extreme_low'})
            return ret
        if ret['alc_low'] > self.halt_thresh_high:
            ret.update({'halted':True, 'reason':'alc extremely high', 'halt_code': 'extreme_high'})
            return ret
        # We didn't halt because of extreme high or low ALC, so now we want to determine
        # if further attacks are likely to yield a better ALC.
        # Sort alc_scores from highest to lowest 'attack_recall'
        sig_attack_prcs = self._get_significant_attack_prcs(alc_scores)
        if len(sig_attack_prcs) == len(alc_scores):
            # This occurs if all alc_scores are significant (have attack_si and
            # base_si < halt_interval_thresh). Make sure we have 200 predictions
            # so that we've given an adequate opportunity to get all the confidence
            # values we're likely to get.
            if len(si_halt.df_base) < 200 and len(si_halt.df_attack) < 200:
                ret.update({'halted':False, 'reason':f'not enough samples even with attack prc measures {len(sig_attack_prcs)} significant (early halt {self.do_early_halt})', 'halt_code': 'none'})
                return ret

            # Check if there are simply not very many different confidence values,
            # because if so, then more attacks aren't going to help much because we
            # aren't likely to get more different confidence values.
            if len(sig_attack_prcs) < self.num_prc_measures:
                ret.update({'halted':True, 'reason':f'all {len(sig_attack_prcs)} attack prc measures significant', 'halt_code': 'all_sig'})
                return ret
            
            # Check to see if we aren't making significant improvement by reducing
            # recall. If not we halt, but if we are, we increase num_prc_measures so
            # as to dig out still lower recall values.
            if ((sig_attack_prcs[-1] - sig_attack_prcs[-2] >= self.halt_min_prc_improvement)
                and (sig_attack_prcs[-2] - sig_attack_prcs[-3] >= self.halt_min_prc_improvement)
               ): 
                self.num_prc_measures += 1
                ret.update({'halted':False, 'reason':f'still improving at {len(sig_attack_prcs)} significant attack prc measures (early halt {self.do_early_halt})', 'halt_code': 'none'})
                return ret
            ret.update({'halted':True, 'reason':f'all {len(sig_attack_prcs)} attack prc measures significant with no improvement', 'halt_code': 'no_improve_all_sig'})
            return ret
        # If we get here, it means that there are some PRC measures that are
        # not yet significant. This means that we have a chance to get better
        # PRC scores with the set of confidence values we currently have.

        # Let's not give up if we don't have many significant attack prc measures.
        if len(sig_attack_prcs) < self.halt_min_significant_attack_prcs:
            ret.update({'halted':False, 'reason':f'too few significant attack prc measures ({len(sig_attack_prcs)}) (early halt {self.do_early_halt})', 'halt_code': 'none'})
            return ret
        ret.update({'halted':False, 'reason':'halt conditions not met (early halt {self.do_early_halt})', 'halt_code': 'none'})
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
    def _init_cntl_and_build_model(self, known_columns: List[str], secret_column: str,  
                                  ) -> None:
        is_assigned = self.df.assign_first_cntl_block()
        if is_assigned is False:
            raise ValueError("Error: Control block initialization failed")
        df = self.df.orig
        self.model_name = self.base_pred.select_model(self.df.orig_all, known_columns, secret_column, self.get_column_classification_dict(), self.random_state)
        self.base_pred.build_model(df, self.random_state)
        # ######### debug stuff #########
        #debug_baseline_predictor(df, known_columns, 'education', self.base_pred)
        #analyze_education_data(df)
        #check_common_issues(df, self.base_pred)
        # ######### end debug stuff #########
        if self.prior_experiment_swap_fraction > 0:
            # This is purely for experimentation and should not be used otherwise
            # self.df.orig contains the sampled original data used for baseline
            # self.df.cntl contains the samples original data used for attack
            # So we want to anonymize self.df.orig but leave it in self.df.orig
            self.df.orig = _swap_anonymize(self.df.orig, self.prior_experiment_swap_fraction)

    def _next_cntl_and_build_model(self) -> bool:
        is_assigned = self.df.assign_next_cntl_block()
        if is_assigned is False:
            return False
        df = self.df.orig
        self.base_pred.build_model(df)
        if self.prior_experiment_swap_fraction > 0:
            # This is purely for experimentation and should not be used otherwise
            self.df.orig = _swap_anonymize(self.df.orig, self.prior_experiment_swap_fraction)
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
            

# Everything after this gets removed after experimentation completed
def _swap_anonymize(df: pd.DataFrame, swap_fraction: float) -> pd.DataFrame:
    # Precompute column indices for faster access
    column_indices = {column: df.columns.get_loc(column) for column in df.columns}

    # Swap values in each column
    for column, column_index in column_indices.items():
        print(f"    Swapping column: {column} at {swap_fraction} fraction")
        df = _swap_random_values(df, column_index, swap_fraction)
    return df

def _swap_random_values(df: pd.DataFrame, column_index: int, swap_fraction: float) -> pd.DataFrame:
    """
    Randomly swaps values in a column for a given fraction of rows.
    """
    num_rows = len(df)
    num_swaps = int(num_rows * swap_fraction)
    if num_swaps < 1:
        return df  # Skip if there are too few rows to swap

    # Select random pairs of indices to swap
    indices = np.random.choice(num_rows, num_swaps * 2, replace=False)
    for i in range(0, len(indices), 2):
        idx1, idx2 = indices[i], indices[i + 1]
        df.iloc[idx1, column_index], df.iloc[idx2, column_index] = (
            df.iloc[idx2, column_index],
            df.iloc[idx1, column_index],
        )
    return df


def _best_row_attack(row: pd.DataFrame,
                     anon: list[pd.DataFrame],
                     secret_column: str,
                     column_classifications: Dict[str, str]) -> Tuple[Any, float]:
    from anonymity_loss_coefficient.utils import find_best_matches, modal_fraction, best_match_confidence
    min_gower_distance, secret_values = find_best_matches(anon=anon,
                                                df_query=row,
                                                secret_column=secret_column,
                                                column_classifications=column_classifications,
                                                )
    number_of_min_gower_distance_matches = len(secret_values)
    pred_value, modal_count = modal_fraction(secret_values)
    modal_frac = modal_count / number_of_min_gower_distance_matches
    confidence = best_match_confidence(
                                    gower_distance=min_gower_distance,
                                    modal_fraction=modal_frac,
                                    match_count=number_of_min_gower_distance_matches)
    return pred_value, confidence

# Add this to your test script or create a new debug script
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def debug_baseline_predictor(df, known_columns, secret_column, predictor):
    """Debug utility to inspect BaselinePredictor behavior"""
    print("=== DEBUGGING BASELINE PREDICTOR ===")
    
    # 1. Check data quality
    print(f"\n1. DATA QUALITY:")
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target column '{secret_column}' unique values: {df[secret_column].nunique()}")
    print(f"   Target dtype: {df[secret_column].dtype}")
    print(f"   Target distribution:\n{df[secret_column].value_counts()}")
    print(f"   Missing values: {df[known_columns + [secret_column]].isnull().sum().sum()}")
    
    # 2. Check feature classifications before and after
    print(f"\n2. FEATURE CLASSIFICATIONS:")
    print(f"   Categorical columns: {predictor.categorical_columns}")
    print(f"   Continuous columns: {predictor.continuous_columns}")
    
    # 3. Check the actual correlations
    print(f"\n3. FEATURE-TARGET RELATIONSHIPS:")
    for col in known_columns:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"   {col}: {unique_vals} unique values")
            
            # Check 1-1 correlation
            feature_to_target = df.groupby(col)[secret_column].apply(lambda x: x.unique())
            one_to_one = all(len(targets) == 1 for targets in feature_to_target)
            print(f"     - 1-1 correlation: {one_to_one}")
            
            # Show some example mappings
            sample_mapping = df[[col, secret_column]].drop_duplicates().head(5)
            print(f"     - Sample mappings:\n{sample_mapping.to_string(index=False)}")
    
    # 4. Check processed features
    print(f"\n4. PROCESSED FEATURES:")
    if predictor.categorical_columns:
        X_cat = predictor.encoder.transform(df[predictor.categorical_columns])
        print(f"   One-hot encoded shape: {X_cat.shape}")
        print(f"   One-hot feature names: {predictor.encoder.get_feature_names_out()[:10]}...")
    else:
        X_cat = np.empty((len(df), 0))
        print(f"   No categorical features to encode")
    
    X_cont = df[predictor.continuous_columns].values if predictor.continuous_columns else np.empty((len(df), 0))
    print(f"   Continuous features shape: {X_cont.shape}")
    
    X = np.hstack([X_cont, X_cat])
    print(f"   Final feature matrix shape: {X.shape}")
    
    # 5. Check if features have variance
    print(f"\n5. FEATURE VARIANCE:")
    feature_vars = np.var(X, axis=0)
    zero_var_count = np.sum(feature_vars == 0)
    print(f"   Features with zero variance: {zero_var_count}/{X.shape[1]}")
    if zero_var_count > 0:
        print(f"   WARNING: {zero_var_count} features have no variance!")
    
    # 6. Check model state (DON'T refit the model)
    print(f"\n6. MODEL STATE:")
    if predictor.model is not None:
        print(f"   Model type: {type(predictor.model).__name__}")
        print(f"   Model classes: {predictor.model.classes_}")
        print(f"   Number of model classes: {len(predictor.model.classes_)}")
        
        # Test a single prediction to see if the model works
        try:
            sample_row = df[known_columns].iloc[[0]]
            prediction, confidence = predictor.predict(sample_row)
            print(f"   Sample prediction: {prediction} (confidence: {confidence})")
            print(f"   ✅ Model prediction works!")
        except Exception as e:
            print(f"   ❌ Model prediction failed: {e}")
    else:
        print(f"   ❌ Model is None - not yet built")
# Usage in your code:
# debug_baseline_predictor(df, known_columns, secret_column, predictor)

# Add this to check the specific relationships in your data
def analyze_education_data(df):
    """Specific analysis for education dataset"""
    print("=== EDUCATION DATA ANALYSIS ===")
    
    # Check education-num to education mapping
    if 'education-num' in df.columns and 'education' in df.columns:
        mapping = df[['education-num', 'education']].drop_duplicates().sort_values('education-num')
        print(f"\nEducation-num to Education mapping:")
        print(mapping.to_string(index=False))
        
        # Check if it's truly 1-1
        edu_num_to_edu = df.groupby('education-num')['education'].nunique()
        edu_to_edu_num = df.groupby('education')['education-num'].nunique()
        
        print(f"\nEducation-num values mapping to multiple educations: {(edu_num_to_edu > 1).sum()}")
        print(f"Education values mapping to multiple education-nums: {(edu_to_edu_num > 1).sum()}")
    
    # Check marital-status to education relationship
    if 'marital-status' in df.columns and 'education' in df.columns:
        crosstab = pd.crosstab(df['marital-status'], df['education'])
        print(f"\nMarital-status vs Education crosstab:")
        print(crosstab)
        
        # Check if marital-status actually has a monotonic relationship
        # This might be the problem!
        marital_edu_mean = df.groupby('marital-status')['education-num'].mean().sort_values()
        print(f"\nAverage education-num by marital-status:")
        print(marital_edu_mean)

# Check these potential problems:

def check_common_issues(df, predictor):
    """Check for common issues that cause poor performance"""
    
    print("=== CHECKING COMMON ISSUES ===")
    
    # Issue 1: Marital-status might not actually be monotonic with education
    if 'marital-status' in predictor.continuous_columns:
        print("\nWARNING: marital-status is being treated as continuous!")
        print("This is likely incorrect - marital status is typically categorical.")
        
        # Check the actual relationship
        if 'marital-status' in df.columns and 'education' in df.columns:
            # Convert education to numeric for correlation
            if df['education'].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                education_numeric = le.fit_transform(df['education'])
            else:
                education_numeric = df['education']
                
            correlation = np.corrcoef(df['marital-status'], education_numeric)[0, 1]
            print(f"Actual correlation between marital-status and education: {correlation:.4f}")
            
            if abs(correlation) < 0.3:
                print("PROBLEM: Weak correlation! Marital-status shouldn't be continuous.")
    
    # Issue 2: Check if we're losing important categorical structure
    if len(predictor.categorical_columns) == 0:
        print("\nWARNING: No categorical columns remaining!")
        print("This might mean all categorical features were incorrectly reclassified.")
    
    # Issue 3: Check for label encoding issues
    if predictor.continuous_columns:
        print(f"\nContinuous columns: {predictor.continuous_columns}")
        for col in predictor.continuous_columns:
            if col in df.columns:
                vals = df[col].unique()
                print(f"{col} values: {sorted(vals)[:10]}...")
                
                # Check if these look like they should be categorical
                if len(vals) < 10 and all(isinstance(v, (int, float)) and v == int(v) for v in vals):
                    print(f"  WARNING: {col} has few integer values - might be better as categorical")