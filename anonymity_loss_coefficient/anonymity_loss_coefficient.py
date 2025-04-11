import numpy as np
import pandas as pd
import os
import json
from typing import Optional, Dict, List, Union, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import norm
from .reporting import *


class DataFiles:
    def __init__(self,
                 df_original: pd.DataFrame,
                 df_control: pd.DataFrame,
                 df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                 disc_max: int = 50,
                 disc_bins: int = 20,
                 discretize_in_place: bool = False,
                 ) -> None:
        self.disc_max = disc_max
        self.disc_bins = disc_bins
        self.discretize_in_place = discretize_in_place
        self.columns_for_discretization = []
        self._encoders = {}
        self.column_classification = {}
        self.orig = df_original
        self.original_columns = df_original.columns.tolist()
        self.cntl = df_control

        if isinstance(df_synthetic, pd.DataFrame):
            self.syn_list = [df_synthetic]
        elif isinstance(df_synthetic, list) and all(isinstance(df, pd.DataFrame) for df in df_synthetic):
            self.syn_list = df_synthetic
        else:
            raise ValueError("df_synthetic must be either a pandas DataFrame or a list of pandas DataFrames")
        self.orig = self.orig.dropna()
        self.cntl = self.cntl.dropna()
        self.syn_list = [df.dropna() for df in self.syn_list]

        # Find numeric columns with more than disc_max unique values in df_orig
        numeric_cols = self.orig.select_dtypes(include=[np.number]).columns
        self.columns_for_discretization = [col for col in numeric_cols if self.orig[col].nunique() > self.disc_max]
        self.new_discretized_columns = [f"{col}__discretized" for col in self.columns_for_discretization]

        # Determine the min and max values for each column to discretize from all DataFrames
        combined_min_max = pd.concat([self.orig, self.cntl] + self.syn_list)
        discretizers = {}
        for col in self.columns_for_discretization:
            min_val = combined_min_max[col].min()
            max_val = combined_min_max[col].max()
            bin_edges = np.linspace(min_val, max_val, num=self.disc_bins+1)
            discretizer = KBinsDiscretizer(n_bins=self.disc_bins, encode='ordinal', strategy='uniform')
            # Fit the discretizer with the combined DataFrame to include feature names
            discretizer.fit(combined_min_max[[col]])
            # Manually set the bin edges
            discretizer.bin_edges_ = np.array([bin_edges])
            discretizers[col] = discretizer

        # Discretize the columns in df_orig, df_cntl, and syn_list using the same bin widths
        self._discretize_df(self.orig, discretizers)
        self._discretize_df(self.cntl, discretizers)
        for i, df in enumerate(self.syn_list):
            self._discretize_df(df, discretizers)

        # set columns_to_encode to be all columns that are not integer and not
        # pre-discretized
        columns_to_encode = self.orig.select_dtypes(exclude=[np.int64, np.int32]).columns
        if self.discretize_in_place is False:
            columns_to_encode = [col for col in columns_to_encode if col not in self.columns_for_discretization]
        self._encoders = self.fit_encoders(columns_to_encode, [self.orig, self.cntl] + self.syn_list)

        self._transform_df(self.orig)
        self._transform_df(self.cntl)
        for i, df in enumerate(self.syn_list):
            self._transform_df(df)

        # As a final step, we want to classify all columns as categorical or continuous
        for col in self.orig.columns:
            if self.discretize_in_place is True or col not in self.columns_for_discretization:
                self.column_classification[col] = 'categorical'
            else:
                self.column_classification[col] = 'continuous'
        

    def get_column_classification(self, column: str) -> str:
        if column not in self.column_classification:
            raise ValueError(f"Column {column} not found in the DataFrame")
        return self.column_classification[column]

    def get_discretized_column(self, secret_column: str) -> str:
        if self.discretize_in_place is False:
            # We might have discritized the secret_column, in which case we want to
            # return the discretized column name
            if secret_column in self.columns_for_discretization:
                return f"{secret_column}__discretized"
        return secret_column

    def get_pre_discretized_column(self, secret_column: str) -> str:
        if self.discretize_in_place is False:
            # We might have discritized the secret_column, in which case we want to
            # return the discretized column name
            if secret_column in self.new_discretized_columns:
                return secret_column.replace("__discretized", "")
        return secret_column

    def _discretize_df(self, df: pd.DataFrame, discretizers: Dict[str, KBinsDiscretizer]) -> None:
        for col in self.columns_for_discretization:
            if col in discretizers:
                discretizer = discretizers[col]
                bin_indices = discretizer.transform(df[[col]]).astype(int).flatten()
                bin_edges = np.round(discretizer.bin_edges_[0], 2)
                bin_labels = [f"[{bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(bin_edges) - 1)]
                if self.discretize_in_place:
                    df.loc[:, col] = pd.Categorical.from_codes(bin_indices, bin_labels)
                else:
                    df.loc[:, f"{col}__discretized"] = pd.Categorical.from_codes(bin_indices, bin_labels)

    def _transform_df(self, df: pd.DataFrame) -> None:
        for col, encoder in self._encoders.items():
            transformed_values = encoder.transform(df[col]).astype(int)
            # Check if the column is of type 'category'
            if pd.api.types.is_categorical_dtype(df[col]):
                # Ensure the column's categories match the new categories
                df.loc[:, col] = pd.Categorical(transformed_values, categories=range(len(encoder.classes_)))
            else:
                # Assign the transformed values directly
                df.loc[:, col] = transformed_values

    def decode_value(self, column: str, encoded_value: int) -> Any:
        if column not in self._encoders:
            return encoded_value
        
        encoder: LabelEncoder = self._encoders[column]

        # Pass encoded_value directly to inverse_transform
        original_value = encoder.inverse_transform(np.array([encoded_value]))

        # Return the first element if only one value was decoded
        return original_value[0] if len(original_value) == 1 else original_value

    def fit_encoders(self, columns_to_encode: List[str], dfs: List[pd.DataFrame]) -> Dict[str, LabelEncoder]:
        encoders = {col: LabelEncoder() for col in columns_to_encode}

        for col in columns_to_encode:
            # Concatenate the values from all DataFrames for this column
            values = pd.concat(df[col] for df in dfs).unique()
            # Fit the encoder on the unique values
            encoders[col].fit(values)

        return encoders


class BaselinePredictor:
    '''
    '''
    def __init__(self, adf: DataFiles) -> None:
        self.adf = adf
        self.model = None

    def build_model(self, known_columns: List[str], secret_col: str,  random_state: Optional[int] = None) -> None:
        X = self.adf.orig[list(known_columns)]
        y = self.adf.orig[secret_col]
        y = y.values.ravel()  # Convert to 1D array

        # Build and train the model
        try:
            model = RandomForestClassifier(random_state=random_state)
        except Exception as e:
            # raise error
            raise ValueError(f"Error building RandomForestClassifier {e}") from e

        y = y.astype(int)  # Convert to integer if necessary
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
    def __init__(self, measure: str = "wilson_score_interval",
                       confidence_level: float = 0.95) -> None:
        if measure not in self.valid_measures():
            raise ValueError(f"Error: Invalid measure {measure}. Use one of {self.valid_measures()}")
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError(f"Error: Invalid condifence {confidence_level}. Must be between 0 and 1")
        self.measure = measure
        self.confidence_level = confidence_level
        self.df_base = pd.DataFrame(columns=['prediction', 'base_confidence'])
        self.df_attack = pd.DataFrame(columns=['prediction', 'attack_confidence'])

    def reset(self) -> None:
        self.df_base = pd.DataFrame(columns=['prediction', 'base_confidence'])
        self.df_attack = pd.DataFrame(columns=['prediction', 'attack_confidence'])

    def _add_row(self, df: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df = row
        else:
            df = pd.concat([df, row], ignore_index=True)
        return df

    def add_prediction(self, prediction: bool,
                       confidence: float,
                       predict_type: str) -> None:
        if predict_type == 'base':
            new_row = pd.DataFrame({'prediction': [prediction], 'base_confidence': [confidence]})
            self.df_base = self._add_row(self.df_base, new_row)
        else:
            new_row = pd.DataFrame({'prediction': [prediction], 'attack_confidence': [confidence]})
            self.df_attack = self._add_row(self.df_attack, new_row)

    def get_confidence_intervals(self) -> Dict:
        alc_scores = self.get_alc_scores(self.df_base, self.df_attack)
        # get alc_score from alc_scores with the maximum 'alc'
        max_alc = max(alc_scores, key=lambda x: x['alc'])
        return {'base': {'prec': max_alc['base_prec'],
                         'ci_low': max_alc['base_ci_low'],
                         'ci_high': max_alc['base_ci_high'],
                         'n': max_alc['base_n']},
                'attack': {'prec': max_alc['attack_prec'],
                           'ci_low': max_alc['attack_ci_low'],
                           'ci_high': max_alc['attack_ci_high'],
                           'n': max_alc['attack_n']}}

    def get_alc_scores(self, df_base: pd.DataFrame,
                             df_attack: pd.DataFrame,
                             ) -> List[Dict]:
        '''
        df_base and df_attack are the dataframes containing only the set of predictions
        of interest (i.e. already grouped in some way).
        '''
        score_info = []
        alc = AnonymityLossCoefficient()
        # sort df_base by base_confidence descending
        df_base = df_base.sort_values(by='base_confidence', ascending=False)
        atk_confs = sorted(df_attack['attack_confidence'].unique(), reverse=True)
        # limit atk_confs to 10 values, because there can be very many
        atk_confs = select_evenly_distributed_values(atk_confs)
        for atk_conf in atk_confs:
            df_atk_conf = df_attack[df_attack['attack_confidence'] >= atk_conf]
            num_predictions = len(df_atk_conf)
            df_base_conf = df_base.head(num_predictions)
            # df_atk_conf and df_base_conf are the rows that pertain to the specific
            # prediction quality (confidence) of interest
            base_prec = df_base_conf['prediction'].mean()
            base_recall = len(df_base_conf) / len(df_base)
            attack_prec = df_atk_conf['prediction'].mean()
            attack_recall = len(df_atk_conf) / len(df_attack)
            alc_score = alc.alc(p_base=base_prec, c_base=base_recall, p_attack=attack_prec, c_attack=attack_recall)
            base_ci = None
            attack_ci = None
            base_low, base_high = self.compute_precision_interval(n = len(df_base_conf),
                                                              precision = base_prec)
            base_ci = base_high - base_low
            attack_low, attack_high = self.compute_precision_interval(n = len(df_atk_conf),
                                                              precision = attack_prec)
            attack_ci = attack_high - attack_low
            score_info.append({
                'base_prec': base_prec,
                'base_recall': base_recall,
                'attack_prec': attack_prec,
                'attack_recall': attack_recall,
                'alc': alc_score,
                'base_ci': base_ci,
                'base_ci_low': base_low,
                'base_ci_high': base_high,
                'base_n': len(df_base_conf),
                'attack_ci': attack_ci,
                'attack_ci_low': attack_low,
                'attack_ci_high': attack_high,
                'attack_n': len(df_atk_conf),
            })
        return score_info

    def compute_precision_interval(self, n: int,
                                   precision: float) -> Tuple[float, float]:
        '''
        If n and precision are provided, use those values.
        returns precision, lower_bound, upper_bound, n
        '''
        if self.measure == "wilson_score_interval":
            if n == 0:
                return 0.0, 0.0
            return self.compute_wilson_score_interval(n, precision, self.confidence_level)

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


class PredictionResults:
    def __init__(self, results_path: str = None,
                       strong_thresh: float = 0.5,
                       risk_thresh: float = 0.7,
                       attack_name: str = '',
                       ci_type: str = 'wilson_score_interval',
                       ci_confidence: float = 0.95) -> None:
        self.strong_thresh = strong_thresh
        self.risk_thresh = risk_thresh
        self.results_path = results_path
        self.attack_name = attack_name
        self.ci_confidence = ci_confidence
        self.ci_type = ci_type
        self.summary_path_csv = None
        if self.results_path is not None:
            os.makedirs(self.results_path, exist_ok=True)
        self.all_known_columns = []
        self.all_secret_columns = []
        self.results = []
        self.ci = ConfidenceInterval()
        self.ci_secret = ''
        self.ci_known_columns = []
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
            score_info = self.ci.get_alc_scores(base_group, attack_group)
            for score in score_info:
                rows.append({
                    'secret_column': secret_col,
                    'base_prec': score['base_prec'],
                    'base_recall': score['base_recall'],
                    'attack_prec': score['attack_prec'],
                    'attack_recall': score['attack_recall'],
                    'alc': score['alc'],
                    'base_count': base_count,
                    'attack_count': attack_count,
                    'base_ci': score['base_ci'],
                    'base_ci_low': score['base_ci_low'],
                    'base_ci_high': score['base_ci_high'],
                    'base_n': score['base_n'],
                    'attack_ci': score['attack_ci'],
                    'attack_ci_low': score['attack_ci_low'],
                    'attack_ci_high': score['attack_ci_high'],
                    'attack_n': score['attack_n'],
                })
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
            score_info = self.ci.get_alc_scores(base_group, attack_group)
            for score in score_info:
                rows.append({
                    'secret_column': secret_col,
                    'known_columns': known_columns,
                    'num_known_columns': num_known_columns,
                    'base_prec': score['base_prec'],
                    'base_recall': score['base_recall'],
                    'attack_prec': score['attack_prec'],
                    'attack_recall': score['attack_recall'],
                    'alc': score['alc'],
                    'base_count': base_count,
                    'attack_count': attack_count,
                    'base_ci': score['base_ci'],
                    'base_ci_low': score['base_ci_low'],
                    'base_ci_high': score['base_ci_high'],
                    'base_n': score['base_n'],
                    'attack_ci': score['attack_ci'],
                    'attack_ci_low': score['attack_ci_low'],
                    'attack_ci_high': score['attack_ci_high'],
                    'attack_n': score['attack_n'],
                })
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
                          with_text: bool = True,
                          with_plot: bool = True) -> None:
        if self.results_path is None:
            raise ValueError("summarize_results called without a results_path")
        df = self.get_results_df()
        self.save_to_csv(df, 'summary_raw.csv')
        df_secret_known = self.alc_per_secret_and_known_df()
        self.save_to_csv(df_secret_known, 'summary_secret_known.csv')
        df_secret = self.alc_per_secret_df()
        self.save_to_csv(df_secret, 'summary_secret.csv')
        if with_text:
            text_summary = make_text_summary(df_secret_known,
                                                self.strong_thresh,
                                                self.risk_thresh,
                                                self.all_secret_columns,
                                                self.all_known_columns,
                                                self.attack_name)
            self.save_to_text(text_summary, 'summary.txt')
        if with_plot:
            plot_alc(df_secret_known,
                        self.strong_thresh,
                        self.risk_thresh,
                        self.attack_name,
                        os.path.join(self.results_path, 'alc_plot.png'))
            plot_alc_prec(df_secret_known,
                        self.strong_thresh,
                        self.risk_thresh,
                        self.attack_name,
                        os.path.join(self.results_path, 'alc_prec_plot.png'))
            plot_alc_best(df_secret_known,
                        self.strong_thresh,
                        self.risk_thresh,
                        self.attack_name,
                        os.path.join(self.results_path, 'alc_plot_best.png'))
            plot_alc_prec_best(df_secret_known,
                        self.strong_thresh,
                        self.risk_thresh,
                        self.attack_name,
                        os.path.join(self.results_path, 'alc_prec_plot_best.png'))

    def add_base_result(self,
                   known_columns: List[str],
                   secret_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   base_confidence: float = None,
                   attack_confidence: float = None,
                   ) -> None:
        self.add_result('base', known_columns, secret_col, predicted_value, true_value, base_confidence, attack_confidence)

    def add_attack_result(self,
                   known_columns: List[str],
                   secret_col: str,
                   predicted_value: Any,
                   true_value: Any,
                   base_confidence: float = None,
                   attack_confidence: float = None,
                   ) -> None:
        self.add_result('attack', known_columns, secret_col, predicted_value, true_value, base_confidence, attack_confidence)

    def check_for_ci_reset(self, known_columns: List[str], secret_col: str) -> None:
        if self.ci_secret != secret_col or self.ci_known_columns != known_columns:
            self.ci_secret = secret_col
            self.ci_known_columns = known_columns
            self.ci.reset()
    
    def get_ci(self) -> Dict:
        return self.ci.get_confidence_intervals()

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
        self.check_for_ci_reset(known_columns, secret_col)
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
        if base_confidence is None:
            confidence = attack_confidence
        self.ci.add_prediction(prediction, confidence, predict_type)

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
        # _prc_abs_weight is the weight given to the absolute PRC difference
        self._prc_abs_weight: float = 0.5
        # _recall_adjust_min_intercept is the recall value below which precision
        # has no effect on the PRC
        self._recall_adjust_min_intercept: float = 1/10000
        # Higher _recall_adjust_strength leads to lower recall adjustment
        self._recall_adjust_strength: float = 3.0

    def set_param(self, param: str, value: float) -> None:
        if param == 'prc_abs_weight':
            self._prc_abs_weight = value
        if param == 'recall_adjust_min_intercept':
            self._recall_adjust_min_intercept = value
        if param == 'recall_adjust_strength':
            self._recall_adjust_strength = value

    def get_param(self, param: str) -> Optional[float]:
        if param == 'prc_abs_weight':
            return self._prc_abs_weight
        if param == 'recall_adjust_min_intercept':
            return self._recall_adjust_min_intercept
        if param == 'recall_adjust_strength':
            return self._recall_adjust_strength
        return None

    def _recall_adjust(self, recall: float) -> float:
        adjust = (np.log10(recall) / np.log10(self._recall_adjust_min_intercept)) ** self._recall_adjust_strength
        return 1 - adjust

    def _prc_improve_absolute(self, prc_base: float, prc_attack: float) -> float:
        return prc_attack - prc_base

    def _prc_improve_relative(self, prc_base: float, prc_attack: float) -> float:
        return (prc_attack - prc_base) / (1.00001 - prc_base)

    def _prc_improve(self, prc_base: float, prc_attack: float) -> float:
        prc_rel = self._prc_improve_relative(prc_base, prc_attack)
        prc_abs = self._prc_improve_absolute(prc_base, prc_attack)
        prc_improve = (self._prc_abs_weight * prc_abs) + ((1 - self._prc_abs_weight) * prc_rel)
        return prc_improve

    def prc(self, prec: float, recall: float) -> float:
        ''' Generates the precision-recall-coefficient, PRC.
            prev is the precision of the attack, and recall is the recall.
        '''
        if recall <= self._recall_adjust_min_intercept:
            return recall
        Rmin = self._recall_adjust_min_intercept
        alpha = self._recall_adjust_strength
        R = recall
        P = prec
        return (1 - ((np.log10(R) / np.log10(Rmin)) ** alpha)) * P

    def alc(self,
            p_base: Optional[float] = None,
            c_base: Optional[float] = None,
            p_attack: Optional[float] = None,
            c_attack: Optional[float] = None,
            prc_base: Optional[float] = None,
            prc_attack: Optional[float] = None
            ) -> Optional[float]:
        ''' alc can be called with either p_x and c_x, or prc_x
        '''
        if prc_base is None and p_base is not None and c_base is not None:
            # Adjust the precision based on the recall to make the
            # precision-recall-coefficient prc
            prc_base = self.prc(p_base, c_base)
        if prc_attack is None and p_attack is not None and c_attack is not None:
            prc_attack = self.prc(p_attack, c_attack)
        if prc_base is not None and prc_attack is not None:
            return self._prc_improve(prc_base, prc_attack)
        return None

    # The following aren't necessary for the AnonymityLossCoefficient, but are just
    # for testing
    def prec_from_prc_recall(self, prc: float, recall: float) -> float:
        ''' Given a PRC and recall, return the precision.
        '''
        Rmin = self._recall_adjust_min_intercept
        alpha = self._recall_adjust_strength
        R = recall
        PRC = prc
        return PRC / (1 - (np.log10(R) / np.log10(Rmin)) ** alpha)

    def recall_from_prc_prec(self, prc: float, prec: float) -> float:
        ''' Given a PRC and precision, return the recall.
        '''
        Rmin = self._recall_adjust_min_intercept
        alpha = self._recall_adjust_strength
        P = prec
        PRC = prc
        return 10 ** (np.log10(Rmin) * (1 - PRC / P) ** (1 / alpha))

    def prcatk_from_prcbase_alc(self, prc_base: float, alc: float) -> float:
        ''' Given a PRC and anonymity loss coefficient, return the PRC of the attack.
        '''
        prc_atk = ((2 * alc) - (2 * alc) * prc_base + (2 * prc_base) - (prc_base ** 2)) / (2 - prc_base)
        return prc_atk


def select_evenly_distributed_values(sorted_list):
    '''
    This limits the number of values in the list to 10 values, evenly distributed
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
