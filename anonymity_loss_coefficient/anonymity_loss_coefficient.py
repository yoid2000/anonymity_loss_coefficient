import numpy as np
import pandas as pd
import os
import json
from typing import Optional, Dict, List, Union, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import norm
from .reporting import *


class DataFiles:
    def __init__(self,
                 df_original: pd.DataFrame,
                 df_control: pd.DataFrame,
                 df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                 col_types: Optional[Dict[str, str]] = None,
                 ) -> None:
        self._encoders = {}
        self.orig = df_original
        self.cntl = df_control

        if col_types is None:
            self.col_types = self.estimate_column_types(self.orig)
        else:
            # make sure that every key in col_types is a column of self.orig, and
            # that every value is either 'categorical' or 'continuous'
            if not all(col in self.orig.columns for col in col_types.keys()):
                raise ValueError("All keys in col_types must be columns of df_original")
            if not all(val in ['categorical', 'continuous'] for val in col_types.values()):
                raise ValueError("All values in col_types must be either 'categorical' or 'continuous'")
            self.col_types = col_types

        if isinstance(df_synthetic, pd.DataFrame):
            self.syn_list = [df_synthetic]
        elif isinstance(df_synthetic, list) and all(isinstance(df, pd.DataFrame) for df in df_synthetic):
            self.syn_list = df_synthetic
        else:
            raise ValueError("df_synthetic must be either a pandas DataFrame or a list of pandas DataFrames")
        self.orig = self.orig.dropna()
        self.cntl = self.cntl.dropna()
        self.syn_list = [df.dropna() for df in self.syn_list]
        columns_to_encode = [col for col in self.col_types.keys() if self.col_types[col] == 'categorical']
        self._encoders = self.fit_encoders(columns_to_encode, [self.orig, self.cntl] + self.syn_list)

        self.orig = self.transform_df(self.orig)
        self.cntl = self.transform_df(self.cntl)
        self.syn_list = [self.transform_df(df) for df in self.syn_list]

    def is_categorical(self, column: str) -> bool:
        if column not in self.col_types:
            raise ValueError(f"Column {column} not found in col_types")
        return self.col_types[column] == 'categorical'

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, encoder in self._encoders.items():
            df[col] = encoder.transform(df[col]).astype(int)
        return df

    def decode_value(self, column: str, encoded_value: int) -> Any:
        if column not in self._encoders:
            return encoded_value
        
        encoder: LabelEncoder = self._encoders[column]
        original_value = encoder.inverse_transform([encoded_value])
        
        return original_value[0]

    def fit_encoders(self, columns_to_encode: List[str], dfs: List[pd.DataFrame]) -> Dict[str, LabelEncoder]:
        encoders = {col: LabelEncoder() for col in columns_to_encode}

        for col in columns_to_encode:
            # Concatenate the values from all DataFrames for this column
            values = pd.concat(df[col] for df in dfs).unique()
            # Fit the encoder on the unique values
            encoders[col].fit(values)

        return encoders

    def estimate_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        column_types: Dict[str, str] = {}
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                column_types[col] = 'categorical'
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 10:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'continuous'
            else:
                column_types[col] = 'continuous'
        return column_types


class BaselinePredictor:
    '''
    '''
    def __init__(self, adf: DataFiles) -> None:
        self.adf = adf
        self.model = None

    def build_model(self, known_columns: List[str], target_col: str) -> None:
        X = self.adf.orig[list(known_columns)]
        y = self.adf.orig[target_col]

        # Build and train the model
        if self.adf.col_types[target_col] == 'categorical':
            try:
                model = RandomForestClassifier(random_state=42)
            except Exception as e:
                # raise error
                raise ValueError(f"Error building RandomForestClassifier {e}") from e
        elif self.adf.col_types == 'continuous':
            try:
                model = RandomForestRegressor(random_state=42)
            except Exception as e:
                raise ValueError(f"Error building RandomForestRegressor {e}") from e
        else:
            raise ValueError("target_type must be 'categorical' or 'continuous'")

        model.fit(X, y)
        self.model = model

    def predict(self, df_row: pd.DataFrame) -> Tuple[Any, float]:
        if self.model is None:
            raise ValueError("Model has not been built yet")
        prediction = self.model.predict(df_row)[0]
        if hasattr(self.model, "predict_proba"):
            base_confidence = self.model.predict_proba(df_row)[0]
            base_confidence = base_confidence[self.model.classes_.tolist().index(prediction)]
        else:
            base_confidence = None
        return prediction, base_confidence

class ConfidenceInterval:
    def __init__(self) -> None:
        self.predictions = []

    def reset(self) -> None:
        self.predictions = []

    def add_prediction(self, prediction: bool) -> None:
        self.predictions.append(prediction)

    def compute_precision_interval(self, measure: str = "wilson_score_interval",
                                   confidence: float = 0.95) -> Tuple[float, float, float, int]:
        if measure not in self.valid_measures():
            raise ValueError(f"Error: Invalid measure {measure}. Use one of {self.valid_measures()}")
        if confidence < 0 or confidence > 1:
            raise ValueError(f"Error: Invalid condifence {confidence}. Must be between 0 and 1")
        if measure == "wilson_score_interval":
            n = len(self.predictions)
            if n == 0:
                return 0.0, 0.0, 0.0, 0
            precision = np.mean(self.predictions)
            return self.compute_wilson_score_interval(n, precision, confidence)

    def valid_measures(self) -> List[str]:
        return ["wilson_score_interval"]

    def compute_wilson_score_interval(self, n: int, precision: float, confidence_level: float = 0.95) -> Tuple[float, float, float, int]:

        z = norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        center_adjusted_probability = precision + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((precision * (1 - precision) + z**2 / (4 * n)) / n)
        lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator

        return precision, lower_bound, upper_bound, n


class PredictionResults:
    def __init__(self, results_path: str = None,
                       strong_thresh: float = 0.5,
                       risk_thresh: float = 0.7) -> None:
        self.strong_thresh = strong_thresh
        self.risk_thresh = risk_thresh
        self.results_path = results_path
        self.summary_path_csv = None
        if self.results_path is not None:
            print(self.results_path)
            os.makedirs(self.results_path, exist_ok=True)
        self.all_known_columns = []
        self.all_target_columns = []
        self.results = []
        self.ci = {'base': {'ci': ConfidenceInterval(),
                            'target': '',
                            'known_columns': []},
                   'attack': {'ci': ConfidenceInterval(),
                              'target': '',
                              'known_columns': []}}

    def _get_results_dict(self) -> List[Dict[str, Any]]:
        return self.results
    
    def _get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def summarize_results(self,
                          with_text: bool = True,
                          with_plot: bool = True) -> None:
        if self.results_path is None:
            raise ValueError("PredictionResults called without a results_path")
        df = self._get_results_df()
        self.save_to_csv(df, 'summary_raw.csv')
        df_secret_known = self.alc_per_secret_and_known(df)
        self.save_to_csv(df_secret_known, 'summary_secret_known.csv')
        df_secret = self.alc_per_secret(df)
        self.save_to_csv(df_secret, 'summary_secret.csv')
        if with_text:
            text_summary = make_text_summary(df_secret_known,
                                                self.strong_thresh,
                                                self.risk_thresh,
                                                self.all_target_columns,
                                                self.all_known_columns)
            self.save_to_text(text_summary, 'summary.txt')
        if with_plot:
            plot_alc(df_secret_known,
                        self.strong_thresh,
                        self.risk_thresh,
                        os.path.join(self.results_path, 'alc_plot.png'))
            plot_alc_prec(df_secret_known,
                        self.strong_thresh,
                        self.risk_thresh,
                        os.path.join(self.results_path, 'alc_prec_plot.png'))

    def print_example_attack(self, df: pd.DataFrame) -> None:
        string = ''
        for _, row in df.iterrows():
            string += f"ALC: {round(row['alc'],2)}, base (prec: {round(row['base_prec'],2)}, recall: {round(row['base_recall'],2)}), attack (prec: {round(row['attack_prec'],2)}, recall: {round(row['attack_recall'],2)})\n"
            string +=f"    Secret: {row['target_column']}, Known: {row['known_columns']}\n"
        return string

    def alc_per_secret(self, df_in: pd.DataFrame) -> pd.DataFrame:
        alc = AnonymityLossCoefficient()
        df_in['prediction'] = df_in['predicted_value'] == df_in['true_value']
        rows = []
        grouped = df_in.groupby('target_column', as_index=False)
        for target_col, group in grouped:
            base_group = group[group['predict_type'] == 'base']
            syn_group = group[group['predict_type'] == 'attack']
            base_prec = base_group['prediction'].mean()
            syn_p = syn_group['prediction'].mean()
            alc_score = alc.alc(p_base=base_prec, c_base=1.0, p_attack=syn_p, c_attack=1.0)
            base_count = len(base_group)
            attack_count = len(syn_group)
            rows.append({
                'target_column': target_col,
                'base_prec': base_prec,
                'base_recall': 1.0,
                'attack_prec': syn_p,
                'attack_recall': 1.0,
                'alc': alc_score,
                'base_count': base_count,
                'attack_count': attack_count
            })
        return pd.DataFrame(rows)
    
    def alc_per_secret_and_known(self, df_in: pd.DataFrame) -> pd.DataFrame:
        alc = AnonymityLossCoefficient()
        df_in['prediction'] = df_in['predicted_value'] == df_in['true_value']
        rows = []
        # known_columns is a string here
        grouped = df_in.groupby(['known_columns', 'target_column'])
        for (known_columns, target_col), group in grouped:
            base_group = group[group['predict_type'] == 'base']
            syn_group = group[group['predict_type'] == 'attack']
            base_prec = base_group['prediction'].mean()
            syn_p = syn_group['prediction'].mean()
            alc_score = alc.alc(p_base=base_prec, c_base=1.0, p_attack=syn_p, c_attack=1.0)
            base_count = len(base_group)
            attack_count = len(syn_group)
            # get the value of num_known_columns from the first row
            num_known_columns = group['num_known_columns'].iloc[0]
            rows.append({
                'target_column': target_col,
                'known_columns': known_columns,
                'num_known_columns': num_known_columns,
                'base_prec': base_prec,
                'base_recall': 1.0,
                'attack_prec': syn_p,
                'attack_recall': 1.0,
                'alc': alc_score,
                'base_count': base_count,
                'attack_count': attack_count
            })
        return pd.DataFrame(rows)

    def add_base_result(self,
                   known_columns: List[str],
                   target_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   base_confidence: float = None,
                   attack_confidence: float = None,
                   ) -> None:
        self.add_result('base', known_columns, target_col, predicted_value, true_value, base_confidence, attack_confidence)

    def add_attack_result(self,
                   known_columns: List[str],
                   target_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   base_confidence: float = None,
                   attack_confidence: float = None,
                   ) -> None:
        self.add_result('attack', known_columns, target_col, predicted_value, true_value, base_confidence, attack_confidence)

    def check_for_ci_reset(self, ci: Dict[str, Any],
                             known_columns: List[str],
                             target_col: str) -> None:
        if ci['target'] != target_col or ci['known_columns'] != known_columns:
            ci['target'] = target_col
            ci['known_columns'] = known_columns
            ci['ci'].reset()
    
    def get_ci(self, ci_type: str, measure: str = "wilson_score_interval",
               confidence: float = 0.95) -> Tuple[float, float, float, int]:
        if ci_type not in ['base', 'attack']:
            raise ValueError(f"Error: Invalid ci_type {ci_type}. Must be 'base' or 'attack'")
        return self.ci[ci_type]['ci'].compute_precision_interval(measure, confidence)

    def add_result(self,
                   predict_type: str,
                   known_columns: List[str],
                   target_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   base_confidence: float = None,
                   attack_confidence: float = None,
                   ) -> None:
        if target_col not in self.all_target_columns:
            self.all_target_columns.append(target_col)
        for col in known_columns:
            if col not in self.all_known_columns:
                self.all_known_columns.append(col)
        # sort known_columns
        known_columns = sorted(known_columns)
        self.check_for_ci_reset(self.ci[predict_type], known_columns, target_col)
        # Check if predicted_value is a numpy type
        if isinstance(predicted_value, np.generic):
            predicted_value = predicted_value.item()
        if predicted_value == true_value:
            prediction = True
        else:
            prediction = False
        self.results.append({
            'predict_type': predict_type,
            'known_columns': json.dumps(known_columns),
            'num_known_columns': len(known_columns),
            'target_column': target_col,
            'predicted_value': predicted_value,
            'true_value': true_value,
            'prediction': prediction,
            'base_confidence': base_confidence,
            'attack_confidence': attack_confidence,
        })
        self.ci[predict_type]['ci'].add_prediction(prediction)

    def save_to_text(self, text_summary: str, file_name: str) -> None:
        save_path = os.path.join(self.results_path, file_name)
        try:
            with open(save_path, 'w') as f:
                f.write(text_summary)
        except PermissionError:
            print(f"Warning: The file at {save_path} is currently open in another application. You might want to make a copy of the summary file in order to view it while the attack is still executing.")
        except Exception as e:
            print(f"Error: Failed to write {save_path}: {e}")

    def save_to_csv(self, df: pd.DataFrame, file_name: str) -> None:
        save_path = os.path.join(self.results_path, file_name)
        try:
            df.to_csv(save_path, index=False)
        except PermissionError:
            print(f"Warning: The file at {save_path} is currently open in another application. You might want to make a copy of the summary file in order to view it while the attack is still executing.")
        except Exception as e:
            print(f"Error: Failed to write {save_path}: {e}")

class AnonymityLossCoefficient:
    '''
    AnonymityLossCoefficient is used to generate an anonymity loss coefficient (ALC).
    The max ALC is 1.0, which corresponds to complete anonymity loss, and is equivalent
    to publishing the original data. An ALC of 0.0 means that the there is
    no anonymity loss. What this means in practice is that the quality of
    attribute inferences about individuals in the synthetic dataset is
    statistically equivalent to the quality of attribute inferences made
    from a non-anonymized dataset about individuals that are not in that dataset.
    The ALC can be negative.  An ALC of 0.5 can be regarded conservatively as a
    safe amount of loss. In other words, the loss is little enough that it 
    eliminates attacker incentive.
    '''
    def __init__(self) -> None:
        # _pcc_abs_weight is the weight given to the absolute PCC difference
        self._pcc_abs_weight: float = 0.5
        # _cov_adjust_min_intercept is the coverage value below which precision
        # has no effect on the PCC
        self._cov_adjust_min_intercept: float = 1/10000
        # Higher _cov_adjust_strength leads to lower coverage adjustment
        self._cov_adjust_strength: float = 3.0

    def set_param(self, param: str, value: float) -> None:
        if param == 'pcc_abs_weight':
            self._pcc_abs_weight = value
        if param == 'cov_adjust_min_intercept':
            self._cov_adjust_min_intercept = value
        if param == 'cov_adjust_strength':
            self._cov_adjust_strength = value

    def get_param(self, param: str) -> Optional[float]:
        if param == 'pcc_abs_weight':
            return self._pcc_abs_weight
        if param == 'cov_adjust_min_intercept':
            return self._cov_adjust_min_intercept
        if param == 'cov_adjust_strength':
            return self._cov_adjust_strength
        return None

    def _cov_adjust(self, cov: float) -> float:
        adjust = (np.log10(cov) / np.log10(self._cov_adjust_min_intercept)) ** self._cov_adjust_strength
        return 1 - adjust

    def _pcc_improve_absolute(self, pcc_base: float, pcc_attack: float) -> float:
        return pcc_attack - pcc_base

    def _pcc_improve_relative(self, pcc_base: float, pcc_attack: float) -> float:
        return (pcc_attack - pcc_base) / (1.00001 - pcc_base)

    def _pcc_improve(self, pcc_base: float, pcc_attack: float) -> float:
        pcc_rel = self._pcc_improve_relative(pcc_base, pcc_attack)
        pcc_abs = self._pcc_improve_absolute(pcc_base, pcc_attack)
        pcc_improve = (self._pcc_abs_weight * pcc_abs) + ((1 - self._pcc_abs_weight) * pcc_rel)
        return pcc_improve

    def pcc(self, prec: float, cov: float) -> float:
        ''' Generates the precision-coverage-coefficient, PCC.
            prev is the precision of the attack, and cov is the coverage.
        '''
        if cov <= self._cov_adjust_min_intercept:
            return cov
        Cmin = self._cov_adjust_min_intercept
        alpha = self._cov_adjust_strength
        C = cov
        P = prec
        return (1 - ((np.log10(C) / np.log10(Cmin)) ** alpha)) * P

    def alc(self,
            p_base: Optional[float] = None,
            c_base: Optional[float] = None,
            p_attack: Optional[float] = None,
            c_attack: Optional[float] = None,
            pcc_base: Optional[float] = None,
            pcc_attack: Optional[float] = None
            ) -> Optional[float]:
        ''' alc can be called with either p_x and c_x, or pcc_x
        '''
        if pcc_base is None and p_base is not None and c_base is not None:
            # Adjust the precision based on the coverage to make the
            # precision-coverage-coefficient pcc
            pcc_base = self.pcc(p_base, c_base)
        if pcc_attack is None and p_attack is not None and c_attack is not None:
            pcc_attack = self.pcc(p_attack, c_attack)
        if pcc_base is not None and pcc_attack is not None:
            return self._pcc_improve(pcc_base, pcc_attack)
        return None

    # The following aren't necessary for the AnonymityLossCoefficient, but are just
    # for testing
    def prec_from_pcc_cov(self, pcc: float, cov: float) -> float:
        ''' Given a PCC and coverage, return the precision.
        '''
        Cmin = self._cov_adjust_min_intercept
        alpha = self._cov_adjust_strength
        C = cov
        PCC = pcc
        return PCC / (1 - (np.log10(C) / np.log10(Cmin)) ** alpha)

    def cov_from_pcc_prec(self, pcc: float, prec: float) -> float:
        ''' Given a PCC and precision, return the coverage.
        '''
        Cmin = self._cov_adjust_min_intercept
        alpha = self._cov_adjust_strength
        P = prec
        PCC = pcc
        return 10 ** (np.log10(Cmin) * (1 - PCC / P) ** (1 / alpha))

    def pccatk_from_pccbase_alc(self, pcc_base: float, alc: float) -> float:
        ''' Given a PCC and anonymity loss coefficient, return the PCC of the attack.
        '''
        pcc_atk = ((2 * alc) - (2 * alc) * pcc_base + (2 * pcc_base) - (pcc_base ** 2)) / (2 - pcc_base)
        return pcc_atk


