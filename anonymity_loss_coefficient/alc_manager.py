import os
import json
from typing import Optional, Dict, List, Union, Any, Tuple
import numpy as np
import pandas as pd
from .reporting import *
from .data_files import DataFiles
from .baseline_predictor import BaselinePredictor
from .score_interval import ScoreInterval


class ALCManager:
    def __init__(self, df_original: pd.DataFrame,
                       df_control: pd.DataFrame,
                       df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                       disc_max: int = 50,
                       disc_bins: int = 20,
                       discretize_in_place: bool = False,
                       si_type: str = 'wilson_score_interval',
                       si_confidence: float = 0.95,
                       halt_thresh_low = 0.4,
                       halt_thresh_high = 0.9,
                       halt_interval_thresh = 0.1,
                       ) -> None:
        self.df = DataFiles(
                 df_original=df_original,
                 df_control=df_control,
                 df_synthetic=df_synthetic,
                 disc_max=disc_max,
                 disc_bins=disc_bins,
                 discretize_in_place=discretize_in_place,
        )
        self.base_pred = BaselinePredictor(self.df.orig)
        self.halt_thresh_low = halt_thresh_low
        self.halt_thresh_high = halt_thresh_high
        self.halt_interval_thresh = halt_interval_thresh
        self.si_confidence = si_confidence
        self.si_type = si_type
        self.summary_path_csv = None
        self.all_known_columns = []
        self.all_secret_columns = []
        self.results = []
        self.si = ScoreInterval()
        self.si_secret = ''
        self.si_known_columns = []
        self.df_secret_known_results = None
        self.df_secret_results = None
        self.df_results = None

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
            score_info = self.si.get_alc_scores(base_group, attack_group)
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
            score_info = self.si.get_alc_scores(base_group, attack_group)
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

    def add_base_result(self,
                   known_columns: List[str],
                   secret_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   base_confidence: float = None,
                   ) -> None:
        self.add_result('base', known_columns, secret_col, predicted_value, true_value, base_confidence, None)

    def add_attack_result(self,
                   known_columns: List[str],
                   secret_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   attack_confidence: float = None,
                   ) -> None:
        self.add_result('attack', known_columns, secret_col, predicted_value, true_value, None, attack_confidence)

    def add_result(self,
                   predict_type: str,
                   known_columns: List[str],
                   secret_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   base_confidence: float = None,
                   attack_confidence: float = None,
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
        self.check_for_si_reset(known_columns, secret_col)
        # Check if predicted_value is a numpy type
        if isinstance(predicted_value, np.generic):
            predicted_value = predicted_value.item()
        if predicted_value == true_value:
            prediction = True
        else:
            prediction = False
        self.results.append({
            'predict_type': predict_type,
            'known_columns': self._make_known_columns_str(known_columns),
            'num_known_columns': len(known_columns),
            'secret_column': secret_col,
            'predicted_value': predicted_value,
            'true_value': true_value,
            'prediction': prediction,
            'base_confidence': base_confidence,
            'attack_confidence': attack_confidence,
        })
        confidence = base_confidence
        if predict_type == 'attack':
            confidence = attack_confidence
        self.si.add_prediction(prediction, confidence, predict_type)

    def check_for_si_reset(self, known_columns: List[str], secret_col: str) -> None:
        if self.si_secret != secret_col or self.si_known_columns != known_columns:
            self.si_secret = secret_col
            self.si_known_columns = known_columns
            self.si.reset()

    def ok_to_halt(self) -> Tuple[bool, Optional[Dict], str]:
        if len(self.si.df_base) < 10 or len(self.si.df_attack) < 10:
            return False, None, 'not enough samples'
        alc_scores = self.si.get_alc_scores(self.si.df_base, self.si.df_attack)
        # get alc_score from alc_scores with the maximum 'alc'
        max_alc = max(alc_scores, key=lambda x: x['alc'])
        ret = {'alc_low': max_alc['alc_low'],
               'alc_high': max_alc['alc_high'],
               'alc': max_alc['alc'],
               'attack_si': max_alc['attack_si'],
               'base_si': max_alc['base_si']}
        if ret['alc_high'] < self.halt_thresh_low:
            return True, ret, 'alc extremely low'
        if ret['alc_low'] > self.halt_thresh_high:
            return True, ret, 'alc extremely high'
        if ret['alc_high'] - ret['alc_low'] < self.halt_interval_thresh:
            # The ALC score interval is small enough that we can stop
            return True, ret, 'alc within bounds'
        if (max_alc['attack_si'] <= self.halt_interval_thresh and 
            max_alc['base_si'] <= self.halt_interval_thresh):
            return True, ret, 'attack and base precision interval within bounds'
        return False, ret, 'halt conditions not met'

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
    def build_model(self, known_columns: List[str], secret_col: str,  random_state: Optional[int] = None) -> None:
        return self.base_pred.build_model(known_columns, secret_col, random_state)

    def predict(self, df_row: pd.DataFrame) -> Tuple[Any, float]:
        return self.base_pred.predict(df_row)
