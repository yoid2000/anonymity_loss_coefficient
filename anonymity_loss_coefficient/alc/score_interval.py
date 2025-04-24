
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import norm
from .anonymity_loss_coefficient import AnonymityLossCoefficient
from .defaults import defaults

class ScoreInterval:
    def __init__(self, si_type: str = defaults['si_type'],
                       si_confidence: float = defaults['si_confidence'],
                       halt_interval_thresh: float = defaults['halt_interval_thresh'],
                       ) -> None:
        if si_type not in self.valid_measures():
            raise ValueError(f"Error: Invalid measure {si_type}. Use one of {self.valid_measures()}")
        if si_confidence < 0 or si_confidence > 1:
            raise ValueError(f"Error: Invalid condifence {si_confidence}. Must be between 0 and 1")
        self.halt_interval_thresh = halt_interval_thresh
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
        elif confidence is not None:
            # If confidence is None, that is a kind of abstention, and we don't want to
            # use it as part of our score interval calculations
            new_row = pd.DataFrame({'prediction': [prediction], 'attack_confidence': [confidence]})
            self.df_attack = self._add_row(self.df_attack, new_row)



    def get_alc_scores(self, df_base: pd.DataFrame,
                             df_attack: pd.DataFrame,
                             max_score_interval: float = 0.5,
                             ) -> List[Dict]:
        '''
        df_base and df_attack are the dataframes containing only the set of predictions
        of interest (i.e. already grouped in some way).
        '''
        alc_scores = []
        # sort df_base by base_confidence descending
        df_base = df_base.sort_values(by='base_confidence', ascending=False)
        atk_confs = sorted(df_attack['attack_confidence'].unique(), reverse=True)
        # limit atk_confs to 10 values, because there can be very many
        atk_confs = _select_evenly_distributed_values(atk_confs)
        for atk_conf in atk_confs:
            df_atk_conf = df_attack[df_attack['attack_confidence'] >= atk_conf]
            num_predictions = len(df_atk_conf)
            df_base_conf = _get_base_subset(df_base, num_predictions)
            # df_atk_conf and df_base_conf are the rows that pertain to the specific
            # prediction quality (confidence) of interest
            base_prec_as_sampled = df_base_conf['prediction'].mean()
            base_recall = len(df_base_conf) / len(df_base)
            attack_prec_as_sampled = df_atk_conf['prediction'].mean()
            attack_recall = len(df_atk_conf) / len(df_attack)
            base_si = None
            attack_si = None
            base_low, base_high = self.compute_precision_interval(n = len(df_base_conf),
                                                              precision = base_prec_as_sampled)
            base_si = base_high - base_low
            base_prec = base_low + (base_si/2)
            base_prc = self.alc.prc(prec=base_prec, recall=base_recall)
            attack_low, attack_high = self.compute_precision_interval(n = len(df_atk_conf),
                                                              precision = attack_prec_as_sampled)
            attack_si = attack_high - attack_low
            if attack_si > max_score_interval or base_si > max_score_interval:
                continue
            alc_as_sampled = self.alc.alc(p_base=base_prec_as_sampled, r_base=base_recall, p_attack=attack_prec_as_sampled, r_attack=attack_recall)
            attack_prec = attack_low + (attack_si/2)
            attack_prc = self.alc.prc(prec=attack_prec, recall=attack_recall)
            # The lower bound uses the si_low of the attack and si_high of the base
            alc_low = self.alc.alc(p_base=base_high, r_base=base_recall,
                                p_attack=attack_low, r_attack=attack_recall)
            # The upper bound is the reverse
            alc_high = self.alc.alc(p_base=base_low, r_base=base_recall,
                                p_attack=attack_high, r_attack=attack_recall)
            alc = self.alc.alc(p_base=base_prec, r_base=base_recall,
                                p_attack=attack_prec, r_attack=attack_recall)
            alc_scores.append({
                'paired': True,
                'alc': alc,
                'base_recall': base_recall,
                'base_prec': base_prec,
                'base_prc': base_prc,
                'base_si': base_si,
                'base_n': len(df_base_conf),
                'attack_recall': attack_recall,
                'attack_prec': attack_prec,
                'attack_prc': attack_prc,
                'attack_si': attack_si,
                'attack_n': len(df_atk_conf),
                'base_prec_as_sampled': base_prec_as_sampled,
                'attack_prec_as_sampled': attack_prec_as_sampled,
                'alc_as_sampled': alc_as_sampled,
                'alc_low': alc_low,
                'alc_high': alc_high,
                'base_si_low': base_low,
                'base_si_high': base_high,
                'attack_si_low': attack_low,
                'attack_si_high': attack_high,
            })
        # We alc_scores so far have closely matching base and attack recalls. However,
        # we want to compute an ALC score using the best base and attack PRC values,
        # even though they may have different recall values.
        base_best_score, attack_best_score = self._get_score_from_max_significant_prcs(alc_scores)
        if base_best_score is not None and attack_best_score is not None:
            # Add non-paired "best" ALC score
            # The lower bound ALC uses the si_low of the attack and si_high of the base
            alc_low = self.alc.alc(p_base=base_best_score['base_si_high'],
                                r_base=base_best_score['base_recall'],
                                p_attack=attack_best_score['attack_si_low'],
                                r_attack=attack_best_score['attack_recall'])
            #print(f"\nalc_low: {alc_low} from p_base: {base_best_score['base_si_high']} p_attack: {attack_best_score['attack_si_low']}")
            # The upper bound ALC is the reverse
            alc_high = self.alc.alc(p_base=base_best_score['base_si_low'],
                                    r_base=base_best_score['base_recall'],
                                    p_attack=attack_best_score['attack_si_high'],
                                    r_attack=attack_best_score['attack_recall'])
            #print(f"\nalc_high: {alc_low} from p_base: {base_best_score['base_si_low']} p_attack: {attack_best_score['attack_si_high']}")

            alc = self.alc.alc(p_base=base_best_score['base_prec'],
                                    r_base=base_best_score['base_recall'],
                                    p_attack=attack_best_score['attack_prec'],
                                    r_attack=attack_best_score['attack_recall'])
            alc_as_sampled = self.alc.alc(p_base=base_best_score['base_prec_as_sampled'],
                                    r_base=base_best_score['base_recall'],
                                    p_attack=attack_best_score['attack_prec_as_sampled'],
                                    r_attack=attack_best_score['attack_recall'])
            alc_scores.append({
                'paired': False,
                'alc': alc,
                'base_recall': base_best_score['base_recall'],
                'base_prec': base_best_score['base_prec'],
                'base_prc': base_best_score['base_prc'],
                'base_si': base_best_score['base_si'],
                'base_n': base_best_score['base_n'],
                'attack_recall': attack_best_score['attack_recall'],
                'attack_prec': attack_best_score['attack_prec'],
                'attack_prc': attack_best_score['attack_prc'],
                'attack_si': attack_best_score['attack_si'],
                'attack_n': attack_best_score['attack_n'],
                'base_prec_as_sampled': base_best_score['base_prec_as_sampled'],
                'attack_prec_as_sampled': attack_best_score['attack_prec_as_sampled'],
                'alc_as_sampled': alc_as_sampled,
                'alc_low': alc_low,
                'alc_high': alc_high,
                'base_si_low': base_best_score['base_si_low'],
                'base_si_high': base_best_score['base_si_high'],
                'attack_si_low': attack_best_score['attack_si_low'],
                'attack_si_high': attack_best_score['attack_si_high'],
            })
        return _do_rounding(alc_scores)


    def split_scores(self, alc_scores: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        best_score = None
        for score in alc_scores:
            if score['paired'] is False:
                best_score = score
                # remove score from alc_scores
                alc_scores.remove(score)
        return best_score, alc_scores


    def _get_score_from_max_significant_prcs(self, alc_scores: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        base_alc_score = None
        max_base_prc = -1.0
        attack_alc_score = None
        max_attack_prc = -1.0
        for score in alc_scores:
            if score['base_si'] > self.halt_interval_thresh or score['attack_si'] > self.halt_interval_thresh:
                continue
            if score['base_prc'] > max_base_prc:
                max_base_prc = score['base_prc']
                base_alc_score = score
            if score['attack_prc'] > max_attack_prc:
                max_attack_prc = score['attack_prc']
                attack_alc_score = score
        return base_alc_score, attack_alc_score

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
        lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator

        return lower_bound, upper_bound


def _select_evenly_distributed_values(sorted_list):
    '''
    This limits the number of values in the list to 10 values, evenly distributed.
    Note by the way that this is being used on the attack confidence values.
    '''
    if len(sorted_list) <= 10:
        return sorted_list
    selected_values = [sorted_list[0]]
    step_size = (len(sorted_list) - 1) / 9
    for i in range(1, 9):
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