import pandas as pd
from typing import Optional, List, Any, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, log_loss
import logging
import numpy as np

# The following to suppress warnings from loky about CPU count
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

class BaselinePredictor:
    '''
    '''
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.model = None
        self.known_columns = None
        self.secret_column = None
        self.encoder = None
        self.categorical_columns = []
        self.continuous_columns = []
        self.selected_model_class = None
        self.selected_model_params = None

    def select_model(
        self,
        df: pd.DataFrame,
        known_columns: List[str],
        secret_column: str,
        column_classifications: Dict[str, str],
        random_state: Optional[int] = None
    ) -> None:
        self.known_columns = known_columns
        self.secret_column = secret_column

        self.categorical_columns = [col for col in self.known_columns if column_classifications.get(col) == 'categorical']
        self.continuous_columns = [col for col in self.known_columns if column_classifications.get(col) == 'continuous']

        # Detect categorical features with 1-1 correlation to target and reclassify as continuous
        self._detect_and_reclassify_correlated_categoricals(df)

        if self.categorical_columns:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat = self.encoder.fit_transform(df[self.categorical_columns])
        else:
            self.encoder = None
            X_cat = np.empty((len(df), 0))

        X_cont = df[self.continuous_columns].values if self.continuous_columns else np.empty((len(df), 0))
        X = np.hstack([X_cont, X_cat])
        y = df[self.secret_column].values.ravel()
        classes = np.unique(y)

        models = [
            ("RandomForest", RandomForestClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=10,
                max_features='sqrt',
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                oob_score=True
            )),
            ("ExtraTrees", ExtraTreesClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=10,
                max_features='sqrt',
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced'
            )),
            ("HistGB", HistGradientBoostingClassifier(
                random_state=random_state,
                max_iter=100,
                max_depth=10
            )),
            ("LogisticRegression", LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=random_state
            )),
        ]
        if len(classes) < 2:
            best_model_name, best_model = models[0]
            self.selected_model = best_model
            return best_model_name

        def log_loss_with_labels(y_true, y_pred_proba, **kwargs):
            # Only score if all classes are present and y_pred_proba is 2D
            if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != len(classes):
                return np.nan
            return log_loss(y_true, y_pred_proba, labels=classes)

        log_loss_scorer = make_scorer(
            log_loss_with_labels,
            greater_is_better=False,
            needs_proba=True
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        best_score = -np.inf
        best_model_name = None
        best_model_class = None
        best_model_params = None

        for name, model in models:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=log_loss_scorer)
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) == 0:
                    continue  # Skip models with all nan scores
                mean_score = valid_scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_class = type(model)
                    best_model_params = model.get_params()
                    best_model_name = name
            except Exception:
                continue  # Skip models that error out

        # Fallback: If no model was selected, use the first model in the list
        if best_model_class is None:
            best_model_name, default_model = models[0]
            best_model_class = type(default_model)
            best_model_params = default_model.get_params()
            self.logger.info(f"No model selected based on validation scores. Using default model: {best_model_name}")
        else:
            self.logger.info(f"Selected model: {best_model_name} with score: {best_score:.4f}")

        self.selected_model_class = best_model_class
        self.selected_model_params = best_model_params
        return best_model_name

    def _detect_and_reclassify_correlated_categoricals(self, df: pd.DataFrame) -> None:
        """
        Detect categorical features that should be treated as continuous and reclassify them.
        
        Detects several cases:
        1. 1-1 correlation with target (perfect mapping)
        2. Monotonic relationship with target
        3. High correlation when treated as continuous vs categorical
        """
        if not self.categorical_columns or self.secret_column not in df.columns:
            return
            
        target_values = df[self.secret_column]
        columns_to_reclassify = []
        
        for cat_col in self.categorical_columns[:]:  # Create a copy to iterate over
            if cat_col not in df.columns:
                continue
                
            feature_values = df[cat_col]
            
            # Case 1: 1-1 correlation between categorical feature and target
            if self._has_one_to_one_correlation(feature_values, target_values):
                columns_to_reclassify.append((cat_col, "1-1 correlation with target"))
                continue
                    
            # Case 2: Monotonic relationship with target when treated as continuous
            if self._has_monotonic_relationship(feature_values, target_values):
                columns_to_reclassify.append((cat_col, "monotonic relationship with target"))
                continue
                
            # Case 3: Better correlation as continuous than categorical
            if self._better_as_continuous(feature_values, target_values):
                columns_to_reclassify.append((cat_col, "stronger continuous correlation"))
                continue
                
        # Reclassify identified columns as continuous
        for col, reason in columns_to_reclassify:
            self.logger.info(f"Reclassifying column '{col}' as continuous because: {reason}")
            self.categorical_columns.remove(col)
            if col not in self.continuous_columns:
                self.continuous_columns.append(col)
                
    def _has_one_to_one_correlation(self, feature_series: pd.Series, target_series: pd.Series) -> bool:
        """Check if there's a 1-1 correlation between feature and target."""
        # Create a mapping from feature values to target values
        feature_to_target = {}
        target_to_feature = {}
        
        for feat_val, target_val in zip(feature_series, target_series):
            # Check feature -> target mapping
            if feat_val in feature_to_target:
                if feature_to_target[feat_val] != target_val:
                    return False  # Feature value maps to multiple target values
            else:
                feature_to_target[feat_val] = target_val
                
            # Check target -> feature mapping  
            if target_val in target_to_feature:
                if target_to_feature[target_val] != feat_val:
                    return False  # Target value maps to multiple feature values
            else:
                target_to_feature[target_val] = feat_val
                
        # Ensure we have the same number of unique values
        return len(set(feature_series)) == len(set(target_series))
        
    def _has_monotonic_relationship(self, feature_series: pd.Series, target_series: pd.Series) -> bool:
        """Check if feature has monotonic relationship with target."""
        try:
            # Group by feature value and compute target means
            grouped = pd.DataFrame({'feature': feature_series, 'target': target_series}).groupby('feature')['target']
            
            if target_series.dtype == 'object' or target_series.dtype.name == 'category':
                # For categorical targets, use mode
                feature_target_means = grouped.agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            else:
                feature_target_means = grouped.mean()
                
            if len(feature_target_means) < 3:  # Need at least 3 points to assess monotonicity
                return False
                
            # Sort by feature value
            sorted_means = feature_target_means.sort_index()
            values = sorted_means.values
            
            # Check if monotonically increasing or decreasing
            is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
            is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
            
            return is_increasing or is_decreasing
        except:
            return False
            
    def _better_as_continuous(self, feature_series: pd.Series, target_series: pd.Series) -> bool:
        """Check if feature has better predictive power as continuous vs categorical."""
        try:
            # Only apply to high-cardinality features (>5 unique values)
            if feature_series.nunique() <= 5:
                return False
                
            # Convert target to numeric if needed for correlation computation
            if target_series.dtype == 'object' or target_series.dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_numeric = le.fit_transform(target_series.astype(str))
            else:
                target_numeric = target_series
                
            # Compute correlation treating feature as continuous
            continuous_corr = abs(np.corrcoef(feature_series, target_numeric)[0, 1])
            
            # Compute "categorical effectiveness" using variance explained by grouping
            grouped_var = pd.DataFrame({'feature': feature_series, 'target': target_numeric}).groupby('feature')['target'].var()
            total_var = target_numeric.var()
            within_group_var = grouped_var.mean()
            categorical_effectiveness = 1 - (within_group_var / total_var) if total_var > 0 else 0
            
            # Prefer continuous if correlation is strong (>0.4) and stronger than categorical
            return continuous_corr > 0.4 and continuous_corr > categorical_effectiveness
        except:
            return False

    def build_model(
        self,
        df: pd.DataFrame,
        random_state: Optional[int] = None
    ) -> None:
        if not hasattr(self, "selected_model_class") or self.selected_model_class is None:
            raise ValueError("No model has been selected. Call select_model() first.")

        model_class = self.selected_model_class
        model_params = self.selected_model_params.copy()
        if random_state is not None and "random_state" in model_params:
            model_params["random_state"] = random_state
        self.model = model_class(**model_params)

        if self.categorical_columns:
            if self.encoder is None:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                X_cat = self.encoder.fit_transform(df[self.categorical_columns])
            else:
                X_cat = self.encoder.transform(df[self.categorical_columns])
        else:
            X_cat = np.empty((len(df), 0))
        X_cont = df[self.continuous_columns].values if self.continuous_columns else np.empty((len(df), 0))
        X = np.hstack([X_cont, X_cat])

        y = df[self.secret_column].values.ravel()
        if y.dtype == object:
            try:
                y = y.astype(int)
            except Exception:
                y = y.astype(str)

        self.model.fit(X, y)

    def predict(self, df_row: pd.DataFrame) -> Tuple[Any, float]:
        if self.model is None:
            raise ValueError("Model has not been built yet")

        model_name = type(self.model).__name__

        if self.categorical_columns:
            X_cat = self.encoder.transform(df_row[self.categorical_columns])
        else:
            X_cat = np.empty((len(df_row), 0))
        X_cont = df_row[self.continuous_columns].values if self.continuous_columns else np.empty((len(df_row), 0))
        X = np.hstack([X_cont, X_cat])

        prediction = self.model.predict(X)[0]
        if hasattr(self.model, "predict_proba"):
            base_confidence = self.model.predict_proba(X)[0]
            base_confidence = base_confidence[self.model.classes_.tolist().index(prediction)]
        else:
            base_confidence = None
        return prediction, base_confidence