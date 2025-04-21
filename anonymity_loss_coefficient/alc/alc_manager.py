import os
import json
from typing import Optional, Dict, List, Union, Any, Tuple, Generator
import numpy as np
import pandas as pd
from .reporting import *
from .data_files import DataFiles
from .baseline_predictor import BaselinePredictor
from .score_interval import ScoreInterval


class ALCManager:
    def __init__(self, df_original: pd.DataFrame,
                       df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                       disc_max: int = 50,
                       disc_bins: int = 20,
                       discretize_in_place: bool = False,
                       si_type: str = 'wilson_score_interval',
                       si_confidence: float = 0.95,
                       max_score_interval: float = 0.5,
                       halt_thresh_low = 0.4,
                       halt_thresh_high = 0.9,
                       halt_interval_thresh = 0.1,
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
        self.base_pred = BaselinePredictor()
        self.random_state = random_state
        self.max_score_interval = max_score_interval
        self.halt_thresh_low = halt_thresh_low
        self.halt_thresh_high = halt_thresh_high
        self.halt_interval_thresh = halt_interval_thresh
        self.summary_path_csv = None
        self.all_known_columns = []
        self.all_secret_columns = []
        self.results = []
        self.si_confidence = si_confidence
        self.si_type = si_type
        self.si = ScoreInterval(measure=self.si_type, confidence_level=self.si_confidence)
        self.si_secret = ''
        self.si_known_columns = []
        self.df_secret_known_results = None
        self.df_secret_results = None
        self.df_results = None

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
                  ) -> Generator[Tuple[pd.DataFrame, Any, Any], None, None]:
        """
        This is the main method of the ALCManager class. It is a generator that yields
        the rows to use in an attack. It makes one baseline prediction per loop. The
        caller must make an attack prediction in each loop. This method determines
        when enough predictions have been made to produce a good ALC score.
        """
        # Establish the targets to ignore, if any, and make a ScoreInterval object
        # for the halting decision.
        ignore_encoded_targets = self._get_targets_to_ignore_for_halting(secret_col)
        si_halt = ScoreInterval(measure=self.si_type, confidence_level=self.si_confidence)

        # Initialize the first set of control rows
        self.init_cntl_and_build_model(known_columns, secret_col)

        num_attacks = 0
        self.halt_info = {'halted': False, 'reason': 'loop not started', 'num_attacks': 0}
        while True:
            for i in range(len(self.df.cntl)):
                num_attacks += 1
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
                if self.encoded_true_value not in ignore_encoded_targets:
                    si_halt_to_use = si_halt
                else:
                    si_halt_to_use = None
                self._model_prediction(atk_row, secret_col, known_columns, si_halt_to_use)

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
                                si_halt=si_halt_to_use)
                self.halt_info = self._ok_to_halt(si_halt)
                self.halt_info.update({'num_attacks': num_attacks})
                if self.halt_info['halted'] is True:
                    return
            is_assigned = self.next_cntl_and_build_model()
            if is_assigned is False:
                self.halt_info = {'halted': False, 'reason': 'exhausted all rows',  'num_attacks': num_attacks}
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

    def _get_targets_to_ignore_for_halting(self, column: str) -> list:
        '''
        With respect to the halting decision, we want to ignore column values that are
        too common because then we won't get an adequate sampling of other column values.
        We put the threshold above 0.5 because we don't want to ignore a value in a
        well-balanced binary column.
        '''
        value_counts = self.df.orig_all[column].value_counts(normalize=True)
        ignore_targets = value_counts[(value_counts > 0.6)].index.tolist()
        return ignore_targets


    def _get_results_dict(self) -> List[Dict[str, Any]]:
        return self.results
    
    def get_results_df(self,
                       known_columns: Optional[List[str]] = None,
                       secret_column: Optional[str] = None) -> pd.DataFrame:
        if self.df_results is None:
            self._get_results_df()
        return self._filter_df(self.df_results, known_columns, secret_column)

    def _get_results_df(self) -> None:
        self.df_results = pd.DataFrame(self.results)

    def alc_per_secret_df(self,
                       known_columns: Optional[List[str]] = None,
                       secret_column: Optional[str] = None) -> pd.DataFrame:
        if self.df_results is None:
            self._get_results_df()
        if self.df_secret_results is None:
            self._alc_per_secret_df()
        return self._filter_df(self.df_secret_results, known_columns, secret_column)

    def _alc_per_secret_df(self) -> None:
        if self.df_results is None:
            self._get_results_df()
        df_in = self.df_results
        df_in['prediction'] = df_in['predicted_value'] == df_in['true_value']
        rows = []
        grouped = df_in.groupby('secret_column', as_index=False)
        for secret_col, group in grouped:
            base_group = group[group['predict_type'] == 'base']
            attack_group = group[group['predict_type'] == 'attack']
            base_count = len(base_group)
            attack_count = len(attack_group)
            score_info = self.si.get_alc_scores(base_group, attack_group, max_score_interval=self.max_score_interval)
            for score in score_info:
                score['secret_column'] = secret_col
                score['base_count'] = base_count
                score['attack_count'] = attack_count
                rows.append(score)
        self.df_secret_results = pd.DataFrame(rows)
    
    def alc_per_secret_and_known_df(self,
                                 known_columns: Optional[List[str]] = None,
                                 secret_column: Optional[str] = None) -> pd.DataFrame:
        if self.df_results is None:
            self._get_results_df()
        if self.df_secret_known_results is None:
            self._alc_per_secret_and_known_df()
        return self._filter_df(self.df_secret_known_results, known_columns, secret_column)

    def _alc_per_secret_and_known_df(self) -> None:
        if self.df_results is None:
            self._get_results_df()
        df_in = self.df_results
        df_in['prediction'] = df_in['predicted_value'] == df_in['true_value']
        rows = []
        # known_columns is a string here
        grouped = df_in.groupby(['known_columns', 'secret_column'])
        for (known_columns, secret_col), group in grouped:
            base_group = group[group['predict_type'] == 'base']
            attack_group = group[group['predict_type'] == 'attack']
            base_count = len(base_group)
            attack_count = len(attack_group)
            num_known_columns = group['num_known_columns'].iloc[0]
            score_info = self.si.get_alc_scores(base_group, attack_group, max_score_interval=self.max_score_interval)
            for score in score_info:
                score['secret_column'] = secret_col
                score['known_columns'] = known_columns
                score['num_known_columns'] = num_known_columns
                score['base_count'] = base_count
                score['attack_count'] = attack_count
                rows.append(score)
        self.df_secret_known_results = pd.DataFrame(rows)


    def _filter_df(self, df: pd.DataFrame,
                  known_columns: Optional[List[str]] = None,
                  secret_column: Optional[str] = None) -> pd.DataFrame:
        if known_columns is not None:
            known_columns_str = self._make_known_columns_str(known_columns)
            df = df[df['known_columns'] == known_columns_str]
        if secret_column is not None:
            df = df[df['secret_column'] == secret_column]
        return df

    def _make_known_columns_str(self, known_columns: List[str]) -> str:
        return json.dumps(sorted(known_columns))

    def summarize_results(self,
                          results_path: str,
                          attack_name: str = '',
                          strong_thresh: float = 0.5,
                          risk_thresh: float = 0.7,
                          with_text: bool = True,
                          with_plot: bool = True) -> None:
        os.makedirs(results_path, exist_ok=True)
        df = self.get_results_df()
        self.save_to_csv(results_path, df, 'summary_raw.csv')
        df_secret_known = self.alc_per_secret_and_known_df()
        self.save_to_csv(results_path, df_secret_known, 'summary_secret_known.csv')
        df_secret = self.alc_per_secret_df()
        self.save_to_csv(results_path, df_secret, 'summary_secret.csv')
        if with_text:
            text_summary = make_text_summary(df_secret_known,
                                                strong_thresh,
                                                risk_thresh,
                                                self.all_secret_columns,
                                                self.all_known_columns,
                                                attack_name)
            self.save_to_text(results_path, text_summary, 'summary.txt')
        if with_plot:
            plot_alc(df_secret_known,
                        strong_thresh,
                        risk_thresh,
                        attack_name,
                        os.path.join(results_path, 'alc_plot.png'))
            plot_alc_prec(df_secret_known,
                        strong_thresh,
                        risk_thresh,
                        attack_name,
                        os.path.join(results_path, 'alc_prec_plot.png'))
            plot_alc_best(df_secret_known,
                        strong_thresh,
                        risk_thresh,
                        attack_name,
                        os.path.join(results_path, 'alc_plot_best.png'))
            plot_alc_prec_best(df_secret_known,
                        strong_thresh,
                        risk_thresh,
                        attack_name,
                        os.path.join(results_path, 'alc_prec_plot_best.png'))


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
        # Reset the results dataframes because they will be out of date after this add
        self.df_secret_known_results = None
        self.df_secret_results = None
        self.df_results = None

        if base_confidence is not None and base_confidence == 0:
            base_confidence = None
        if attack_confidence is not None and attack_confidence == 0:
            attack_confidence = None
        if secret_col not in self.all_secret_columns:
            self.all_secret_columns.append(secret_col)
        for col in known_columns:
            if col not in self.all_known_columns:
                self.all_known_columns.append(col)
        # sort known_columns
        known_columns = sorted(known_columns)
        self._check_for_si_reset(known_columns, secret_col)
        # Check if predicted_value is a numpy type
        if isinstance(decoded_predicted_value, np.generic):
            decoded_predicted_value = decoded_predicted_value.item()
        if decoded_predicted_value == decoded_true_value:
            prediction = True
        else:
            prediction = False
        self.results.append({
            'predict_type': predict_type,
            'known_columns': self._make_known_columns_str(known_columns),
            'num_known_columns': len(known_columns),
            'secret_column': secret_col,
            'predicted_value': decoded_predicted_value,
            'true_value': decoded_true_value,
            'encoded_predicted_value': encoded_predicted_value,
            'encoded_true_value': encoded_true_value,
            'prediction': prediction,
            'base_confidence': base_confidence,
            'attack_confidence': attack_confidence,
        })
        confidence = base_confidence
        if predict_type == 'attack':
            confidence = attack_confidence
        self.si.add_prediction(prediction, confidence, predict_type)
        if si_halt is not None:
            si_halt.add_prediction(prediction, confidence, predict_type)

    def _check_for_si_reset(self, known_columns: List[str], secret_col: str) -> None:
        if self.si_secret != secret_col or self.si_known_columns != known_columns:
            self.si_secret = secret_col
            self.si_known_columns = known_columns
            self.si.reset()

    def _ok_to_halt(self, si_halt: ScoreInterval) -> Dict[str, Any]:
        if len(si_halt.df_base) < 10 or len(si_halt.df_attack) < 10:
            return {'halted': False, 'reason': 'not enough samples'}
        alc_scores = si_halt.get_alc_scores(si_halt.df_base, si_halt.df_attack, max_score_interval=self.max_score_interval)
        if len(alc_scores) == 0:
            return {'halted': False, 'reason':f'no alc scores with attack_si and base_si < {self.max_score_interval}'}
        # sort alc_scores by 'alc' descending
        alc_scores.sort(key=lambda x: x['alc'], reverse=True)
        max_alc = alc_scores[0]
        ret = {'alc_low': float(round(max_alc['alc_low'], 3)),
               'alc_high': float(round(max_alc['alc_high'], 3)),
               'alc': float(round(max_alc['alc'], 3)),
               'attack_si': float(round(max_alc['attack_si'], 3)),
               'base_si': float(round(max_alc['base_si'], 3))}
        if ret['alc_high'] < self.halt_thresh_low:
            ret.update({'halted':True, 'reason':'alc extremely low'})
            return ret
        if ret['alc_low'] > self.halt_thresh_high:
            ret.update({'halted':True, 'reason':'alc extremely high'})
            return ret
        if (max_alc['attack_si'] <= self.halt_interval_thresh and
            max_alc['base_si'] <= self.halt_interval_thresh):
            ret.update({'halted':True, 'reason':'attack and base precision interval within bounds'})
            return ret
        return {'halted': False, 'reason': 'halt conditions not met'}

    def save_to_text(self, results_path: str, text_summary: str, file_name: str) -> None:
        save_path = os.path.join(results_path, file_name)
        try:
            with open(save_path, 'w') as f:
                f.write(text_summary)
        except PermissionError:
            print(f"Warning: The file at {save_path} is currently open in another application. You might want to make a copy of the summary file in order to view it while the attack is still executing.")
        except Exception as e:
            print(f"Error: Failed to write {save_path}: {e}")

    def save_to_csv(self, results_path, df: pd.DataFrame, file_name: str) -> None:
        save_path = os.path.join(results_path, file_name)
        try:
            df.to_csv(save_path, index=False)
        except PermissionError:
            print(f"Warning: The file at {save_path} is currently open in another application. You might want to make a copy of the summary file in order to view it while the attack is still executing.")
        except Exception as e:
            print(f"Error: Failed to write {save_path}: {e}")
    
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
    def init_cntl_and_build_model(self, known_columns: List[str], secret_col: str,  
                                  ) -> None:
        is_assigned = self.df.assign_first_cntl_block()
        if is_assigned is False:
            raise ValueError("Error: Control block initialization failed")
        self.base_pred.build_model(self.df.orig, known_columns, secret_col, self.random_state)

    def next_cntl_and_build_model(self) -> bool:
        is_assigned = self.df.assign_next_cntl_block()
        if is_assigned is False:
            return False
        self.base_pred.build_model(self.df.orig)
        return True

    def predict(self, df_row: pd.DataFrame) -> Tuple[Any, float]:
        return self.base_pred.predict(df_row)
