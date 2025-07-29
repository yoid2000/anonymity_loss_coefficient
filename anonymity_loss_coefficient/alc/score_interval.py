
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import norm
from .anonymity_loss_coefficient import AnonymityLossCoefficient

class ScoreInterval:
    def __init__(self, si_type: str,
                       si_confidence: float,
                       halt_interval_thresh: float,
                       max_score_interval: float,
                       logger: logging.Logger = None,
                       ) -> None:
        if si_type not in self.valid_measures():
            raise ValueError(f"Error: Invalid measure {si_type}. Use one of {self.valid_measures()}")
        if si_confidence < 0 or si_confidence > 1:
            raise ValueError(f"Error: Invalid condifence {si_confidence}. Must be between 0 and 1")
        self.logger = logger
        self.progress = []
        self.best_base_prc = None
        self.best_attack_prc = None
        self.halt_interval_thresh = halt_interval_thresh
        self.max_score_interval = max_score_interval
        self.si_type = si_type
        self.si_confidence = si_confidence
        self.df_base = pd.DataFrame(columns=['prediction', 'base_confidence'])
        self.df_attack = pd.DataFrame(columns=['prediction', 'attack_confidence'])
        self.alc = AnonymityLossCoefficient()

    def reset(self) -> None:
        self.df_base = pd.DataFrame(columns=['prediction', 'base_confidence'])
        self.df_attack = pd.DataFrame(columns=['prediction', 'attack_confidence'])

    def _add_row(self, df: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            # Directly return the row as the new DataFrame if df is empty
            return row.reset_index(drop=True)
        else:
            # Concatenate the DataFrames if df is not empty
            return pd.concat([df, row], ignore_index=True)

    def add_prediction(self, prediction: bool,
                       confidence: Optional[float],
                       predict_type: str) -> None:
        if predict_type == 'base':
            new_row = pd.DataFrame({'prediction': [prediction], 'base_confidence': [confidence]})
            self.df_base = self._add_row(self.df_base, new_row)
        else:
            new_row = pd.DataFrame({'prediction': [prediction], 'attack_confidence': [confidence]})
            self.df_attack = self._add_row(self.df_attack, new_row)


    def compute_precision_interval(self, n: int,
                                   precision: float) -> Tuple[float, float]:
        '''
        If n and precision are provided, use those values.
        returns precision, lower_bound, upper_bound, n
        '''
        if self.si_type == "wilson_score_interval":
            if n == 0:
                return 0.0, 0.0
            return self.compute_wilson_score_interval(n, precision, self.si_confidence)

    def valid_measures(self) -> List[str]:
        return ["wilson_score_interval"]

    def compute_wilson_score_interval(self, n: int, precision: float, confidence_level: float = 0.95) -> Tuple[float, float]:

        z = norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        center_adjusted_probability = precision + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((precision * (1 - precision) + z**2 / (4 * n)) / n)
        lower_bound = float((center_adjusted_probability - z * adjusted_standard_deviation) / denominator)
        upper_bound = float((center_adjusted_probability + z * adjusted_standard_deviation) / denominator)

        return float(round(lower_bound, 4)), float(round(upper_bound, 4))

    def get_basic_data(self, total_predict: int = 0) -> Dict:
        return {
            'n': 0,
            'total_predict': total_predict,
            'si_low': 0.0,
            'si_high': 0.0,
            'si': 0.0,
            'prec': 0.0,
            'recall': 0.0,
            'prc': 0.0,
        }

    def get_dummy_data(self) -> Dict:
        return self._compile_data(
            base_data=self.get_basic_data(),
            attack_data=self.get_basic_data()
        )

    def _compute_best_prc(self, pred_type: str) -> Dict:
        if pred_type not in ['base', 'attack']:
            raise ValueError(f"Error: Invalid prediction type {pred_type}. Use 'base' or 'attack'")
        conf = f'{pred_type}_confidence'
        if pred_type == 'base':
            return self.compute_best_prc(df=self.df_base, conf=conf)
        else:
            return self.compute_best_prc(df=self.df_attack, conf=conf)

    def compute_best_prc(self, df: pd.DataFrame, conf: Optional[str] = 'confidence') -> Dict:
        ''' This is a helper routine that computes the best PRC for a given set
            of predictions and confidence scores.
            
            df is a DataFrame with columns 'prediction' and conf,
            where conf is a string representing the confidence column name.

            Returns a dictionary with the best PRC and related values.
        '''
        max_prc = 0
        used_confidence = None
        df_sorted_pred = df.sort_values(by=conf, ascending=False)
        total_rows = len(df_sorted_pred)

        for n in range(1, total_rows + 1):
            top_n_rows = df_sorted_pred.head(n)
            correct_predictions = top_n_rows['prediction'].sum()
            prec = correct_predictions / n
            recall = n / total_rows
            low, high = self.compute_precision_interval(n = n, precision = prec)
            prec_interval = high - low
            
            if prec_interval < 0.1:
                mid_prec = low + ((high - low) / 2)  # Use midpoint for precision
                prc = self.alc.prc(prec=mid_prec, recall=recall)
                if prc > max_prc:
                    max_prc = prc
                    used_confidence = top_n_rows.iloc[-1][conf]

        # compute precision, recall, and prc for all rows with
        # confidence >= used_confidence. Needed because the above loop may not have
        # included all rows at the used_confidence level.
        # If the following is changed, also change get_dummy_data()
        res = self.get_basic_data(total_predict=total_rows)
        if used_confidence is not None:
            final_rows = df_sorted_pred[df_sorted_pred[conf] >= used_confidence]
            final_correct = final_rows['prediction'].sum()
            res['n'] = len(final_rows)
            final_precision = final_correct / res['n'] if res['n'] > 0 else 0
            res['si_low'], res['si_high'] = self.compute_precision_interval(n = res['n'], precision = final_precision)
            res['si'] = round(res['si_high'] - res['si_low'], 4)
            res['prec'] = round(res['si_low'] + ((res['si_high'] - res['si_low']) / 2), 4)  # Use midpoint for precision
            res['recall'] = round(res['n'] / total_rows, 4)
            res['prc'] = self.alc.prc(prec=res['prec'], recall=res['recall'])
        return res

    def _compile_data(self, base_data: Dict, attack_data: Dict) -> Dict:
        data = {}
        prefix = 'base_'
        for key, value in base_data.items():
            data[prefix + key] = value
        prefix = 'attack_'
        for key, value in attack_data.items():
            data[prefix + key] = value
        data['alc'] = self.alc.alc(p_base=data['base_prec'], r_base=data['base_recall'],
                           p_attack=data['attack_prec'], r_attack=data['attack_recall'])
        return data

    def ok_to_halt(self) -> Dict[str, Any]:
        ''' Monitors the progress of the attack and baseline, and determines if
        the results of both are significant enough to halt the predictor loop.
        Operates by peridocally computing the best PRC scores for both attack
        and baseline using _compute_best_prc(), and checking for
        diminishing returns in improving PRC.
        It makes its first prc measures when there are 100 predictions. Subsequently,
        it measures prcs every 50 additional predictions, and compares the prc values
        to the prior measures.
        '''
        if len(self.df_base) != len(self.df_attack):
            # throw and exception
            raise ValueError(f"Error: The number of base and attack predictions are not the same. Base: {len(self.df_base)}, Attack: {len(self.df_attack)}")

        if len(self.df_base) < 100 or len(self.df_attack) < 100:
            return {'halted': False,
                    'reason': f"not enough samples",
                    'halt_code': 'none', 'data': None,}
        if len(self.df_base) % 50 != 0:
            return {'halted': False,
                    'reason': f'not measure checkpoint',
                    'halt_code': 'none', 'data': None,}
        base_prc_res = self._compute_best_prc('base')
        attack_prc_res = self._compute_best_prc('attack')
        data = self._compile_data(base_prc_res, attack_prc_res)
        self.logger.info(f"\nCheckpoint: {len(self.df_base)} predictions: {data}")
        if base_prc_res['n'] == 0 or attack_prc_res['n'] == 0:
            return {'halted': False,
                    'alc': data['alc'],
                    'reason': f"significant intervals not reached (base: {base_prc_res['si']}, attack: {attack_prc_res['si']})",
                    'halt_code': 'none',
                    'data': data,}
        if ((self.best_base_prc is None and self.best_attack_prc is not None) or
           (self.best_base_prc is not None and self.best_attack_prc is None)):
            raise ValueError("Error: Only one of self.best_base_prc and self.best_attack_prc is None. This should not happen.")
        if self.best_base_prc is None:
            # First checkpoint
            self.progress.append({'base_res': base_prc_res, 'attack_res': attack_prc_res})
            self.best_base_prc = base_prc_res
            self.best_attack_prc = attack_prc_res
            return {'halted': False,
                    'alc': data['alc'],
                    'reason': f"first checkpoint: base_prc: {base_prc_res['prc']}, attack_prc: {attack_prc_res['prc']}",
                    'halt_code': 'none',
                    'data': data,}
        # Beyond this point, each checkpoint is compared to the last
        if self.best_base_prc is None or self.best_attack_prc is None:
            raise ValueError("Error: last_base_prc or last_attack_prc is None. This should not happen.")
        base_improv_thresh = max(0.01, (1.0 - self.best_base_prc['prc']) * 0.05)
        attack_improv_thresh = max(0.01, (1.0 - self.best_attack_prc['prc']) * 0.05)
        base_progress = (base_prc_res['prc'] - self.best_base_prc['prc']) >= base_improv_thresh
        attack_progress = (attack_prc_res['prc'] - self.best_attack_prc['prc']) >= attack_improv_thresh
        if base_prc_res['prc'] > self.best_base_prc['prc']:
            self.best_base_prc = base_prc_res
        if attack_prc_res['prc'] > self.best_attack_prc['prc']:
            self.best_attack_prc = attack_prc_res
        if base_progress is False and attack_progress is False:
            data = self._compile_data(self.best_base_prc, self.best_attack_prc)
            self.logger.info(f"\nCheckpoint: {len(self.df_base)} predictions: {data}")
            return {'halted': True,
                    'alc': data['alc'],
                    'reason': f"diminishing returns: base_prc: {base_prc_res['prc']}, attack_prc: {attack_prc_res['prc']}",
                    'halt_code': 'diminishing_returns',
                    'data': data,}
        return {'halted': False,
                'alc': data['alc'],
                'reason': f"still making progress: base_prc: {base_prc_res['prc']}, attack_prc: {attack_prc_res['prc']}",
                'halt_code': 'none',
                'data': data,}


def _select_evenly_distributed_values(sorted_list: list, num_prc_measures: int) -> List[float]:
    '''
    This splits the confidence values into num_prc_measures evenly spaced values.
    Note by the way that this is being used on the attack confidence values.
    '''
    if len(sorted_list) <= num_prc_measures:
        return sorted_list
    selected_values = [sorted_list[0]]
    step_size = (len(sorted_list) - 1) / (num_prc_measures - 1)
    for i in range(1, num_prc_measures - 1):
        index = int(round(i * step_size))
        selected_values.append(sorted_list[index])
    selected_values.append(sorted_list[-1])
    return selected_values


def _get_base_subset(df_base: pd.DataFrame, num_predictions: int) -> pd.DataFrame:
    """
    Returns a subset of df_base with num_rows rows, where num_rows is as close as possible
    to num_predictions while satisfying the following constraints:
    1. The row at index num_rows-1 has a different base_confidence value than the row at index num_rows.
    2. num_rows must not be zero.

    Args:
        df_base (pd.DataFrame): The input DataFrame, sorted by 'base_confidence' in descending order.
        num_predictions (int): The target number of rows to include in the subset.

    Returns:
        pd.DataFrame: A subset of df_base with num_rows rows.
    """
    # Ensure num_predictions is valid
    if num_predictions <= 0:
        raise ValueError("num_predictions must be greater than 0.")
    if num_predictions > len(df_base):
        raise ValueError("num_predictions must be no more than the length of df_base.")

    # Get the unique base_confidence values
    unique_confidences = sorted(df_base['base_confidence'].unique(), reverse=True)

    # Initialize variables to track the closest cut-point
    cumulative_count = 0
    previous_count = 0
    num_rows = 0

    # Iterate through unique confidence values to find the closest cut-point
    for confidence in unique_confidences:
        # Count rows with the current confidence value
        count = (df_base['base_confidence'] == confidence).sum()
        previous_count = cumulative_count
        cumulative_count += count

        # Check if we've reached or exceeded num_predictions
        if cumulative_count >= num_predictions:
            # Decide whether to use the previous cut-point or the current one
            if (abs(previous_count - num_predictions) <= abs(cumulative_count - num_predictions)) and previous_count != 0:
                num_rows = previous_count
            else:
                num_rows = cumulative_count
            break

    # Return the subset of df_base
    return df_base.head(num_rows)

def _do_rounding(alc_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for score in alc_scores:
        for key in score.keys():
            if isinstance(score[key], float) or isinstance(score[key], np.float64):
                score[key] = float(round(score[key], 4))
    return alc_scores