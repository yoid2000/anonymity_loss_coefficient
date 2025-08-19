import pandas as pd
from typing import Optional, List, Any, Tuple, Dict, TYPE_CHECKING
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import logging
import numpy as np

if TYPE_CHECKING:
    from .score_interval import ScoreInterval

# The following to suppress warnings from loky about CPU count
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

class OneToOnePredictor:
    """
    A predictor for features that have a strong correspondence with the target.
    For each feature value, uses the most frequent target value as the prediction.
    """
    def __init__(self, df: pd.DataFrame, feature: str, target: str) -> None:
        """
        Initialize the predictor with a mapping from feature values to most frequent target values.
        
        Args:
            df: DataFrame containing the feature and target columns
            feature: Name of the feature column
            target: Name of the target column
        """
        self.feature_name = feature
        self.target_name = target
        
        # Create the mapping from feature values to most frequent target values
        self.mapping = {}
        grouped = df.groupby(feature)[target]
        
        for feature_value, target_values in grouped:
            # Find the most frequent target value for this feature value
            most_frequent_target = target_values.mode()
            if len(most_frequent_target) > 0:
                self.mapping[feature_value] = most_frequent_target.iloc[0]
            else:
                # Fallback if mode is empty (shouldn't happen but safety first)
                self.mapping[feature_value] = target_values.iloc[0]
        
    def predict(self, feature_value: Any) -> Any:
        """
        Predict the target value for a given feature value using the most frequent mapping.
        
        Args:
            feature_value: A value from the feature column
            
        Returns:
            The most frequent target value for this feature value
            
        Raises:
            KeyError: If the feature value is not found in the mapping
        """
        if feature_value not in self.mapping:
            raise KeyError(f"Feature value '{feature_value}' not found in mapping")
        
        return self.mapping[feature_value]

class BaselinePredictor:
    '''
    '''
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.model = None
        self.known_columns = None
        self.secret_column = None
        self.encoder = None
        self.onehot_columns = []
        self.non_onehot_columns = []
        self.selected_model = None
        self.selected_model_class = None
        self.selected_model_params = None
        self.selected_model_name = None
        self.selected_model_prc = None
        self.otop = None
        self.df_pred_conf = None
        self.si = None

    def select_model(
        self,
        df: pd.DataFrame,
        known_columns: List[str],
        secret_column: str,
        column_classifications: Dict[str, str],
        si: "ScoreInterval",
        random_state: Optional[int] = None
    ) -> Tuple[str, float]:
        self.known_columns = known_columns
        self.secret_column = secret_column
        self.si = si

        self.onehot_columns = [col for col in self.known_columns if column_classifications.get(col) == 'categorical']
        self.non_onehot_columns = [col for col in self.known_columns if column_classifications.get(col) == 'continuous']

        self.otop = self._detect_and_reclassify_correlated_categoricals(df)
        
        # Create df_train and df_test
        df_modified = df.copy()
        
        # Determine test size
        if len(df_modified) < 6000:
            test_size = 0.5
            self.logger.info(f"Dataset has {len(df_modified)} rows (<6000), using 50% for test set")
        else:
            test_size = min(3000 / len(df_modified), 0.5)
            self.logger.info(f"Dataset has {len(df_modified)} rows, using test size of {test_size:.3f} (target: 3000 rows)")
        
        # Create stratified train/test split
        from sklearn.model_selection import train_test_split
        try:
            df_train, df_test = train_test_split(
                df_modified,
                test_size=test_size,
                stratify=df_modified[secret_column],
                random_state=random_state
            )
            self.logger.info(f"Created stratified split: {len(df_train)} train, {len(df_test)} test")
        except ValueError as e:
            # Fallback to regular split if stratification fails (e.g., too few samples per class)
            self.logger.warning(f"Stratified split failed ({e}), using regular split")
            df_train, df_test = train_test_split(
                df_modified,
                test_size=test_size,
                random_state=random_state
            )
            self.logger.info(f"Created regular split: {len(df_train)} train, {len(df_test)} test")
        
        return self._select_best_model(df_train, df_test, random_state)

    def build_model(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        random_state: Optional[int] = None
    ) -> None:
        if self.known_columns is None or self.secret_column is None:
            raise ValueError("Must call select_model first")
        
        self._build_model_from_stored_config(df_train, df_test, random_state)

    def _select_best_model(self, df_train: pd.DataFrame, df_test: pd.DataFrame, random_state: Optional[int]) -> Tuple[str, float]:
        """Select the best model based on PRC scores."""
        models = [
            ("RandomForest", RandomForestClassifier(
                random_state=random_state,
                n_estimators=100,
            )),
            ("ExtraTrees", ExtraTreesClassifier(
                random_state=random_state,
                n_estimators=100,
            )),
            ("HistGB", HistGradientBoostingClassifier(
                random_state=random_state,
                max_iter=500,
            )),
            ("LogisticRegression", LogisticRegression(
                max_iter=5000,
                random_state=random_state
            )),
        ]
        
        best_prc = -1
        best_model_name = None
        best_df_pred_conf = None
        
        # Test OneToOnePredictor if available
        if self.otop is not None:
            try:
                df_pred_conf = self._build_otop_predictions(df_test)
                prc_dict = self.si.compute_best_prc(df=df_pred_conf)
                prc_score = prc_dict['prc']
                self.logger.info(f"OneToOnePredictor PRC score: {prc_score:.4f}")
                
                if prc_score > best_prc:
                    best_prc = prc_score
                    best_model_name = "OneToOnePredictor"
                    best_df_pred_conf = df_pred_conf
            except Exception as e:
                self.logger.warning(f"OneToOnePredictor failed: {e}")
        
        # Test ML models
        self.logger.info("Determine best ML model:")
        for model_name, model in models:
            try:
                self.logger.info(f"   Testing model: {model_name}")
                df_pred_conf = self._build_ml_model_predictions(df_train, df_test, model)
                prc_dict = self.si.compute_best_prc(df=df_pred_conf)
                prc_score = prc_dict['prc']
                self.logger.info(f"    {model_name} PRC score: {prc_score:.4f}")
                for key, value in prc_dict.items():
                    if key != 'prc':
                        self.logger.info(f"          {key}: {value}")
                
                if prc_score > best_prc:
                    best_prc = prc_score
                    best_model_name = model_name
                    best_df_pred_conf = df_pred_conf
                    # Store model info for later use
                    self.selected_model = model
                    self.selected_model_class = type(model)
                    self.selected_model_params = model.get_params()
                    
            except Exception as e:
                self.logger.warning(f"{model_name} failed: {e}")
                continue
        
        if best_model_name is None:
            raise ValueError("No model could be successfully trained")
        
        # Store results
        self.selected_model_name = best_model_name
        self.df_pred_conf = best_df_pred_conf
        self.selected_model_prc = best_prc
        
        if best_model_name != "OneToOnePredictor":
            self.otop = None  # Clear if ML model was selected
        
        self.logger.info(f"Selected model: {best_model_name} with PRC score: {best_prc:.4f}")
        return best_model_name, best_prc
    
    def _build_model_from_stored_config(self, df_train: pd.DataFrame, df_test: pd.DataFrame, random_state: Optional[int]) -> None:
        """Build model using previously stored configuration."""
        if self.selected_model_name == "OneToOnePredictor":
            if self.otop is None:
                raise ValueError("OneToOnePredictor was selected but otop is None")
            self.df_pred_conf = self._build_otop_predictions(df_test)
        else:
            # Rebuild ML model with new training data
            model_class = self.selected_model_class
            model_params = self.selected_model_params.copy()
            if random_state is not None and "random_state" in model_params:
                model_params["random_state"] = random_state
            
            model = model_class(**model_params)
            self.df_pred_conf = self._build_ml_model_predictions(df_train, df_test, model)
    
    def _build_otop_predictions(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Build predictions using OneToOnePredictor."""
        predictions = []
        for _, row in df_test.iterrows():
            feature_value = row[self.otop.feature_name]
            try:
                predicted_value = self.otop.predict(feature_value)
                true_value = row[self.secret_column]
                prediction = (predicted_value == true_value)
                confidence = 1.0
                
                predictions.append({
                    'predicted_value': predicted_value,
                    'prediction': prediction,
                    'confidence': confidence
                })
            except KeyError:
                # Skip rows with unknown feature values
                continue
        
        return pd.DataFrame(predictions)
    
    def _build_ml_model_predictions(self, df_train: pd.DataFrame, df_test: pd.DataFrame, model) -> pd.DataFrame:
        """Build predictions using ML model."""
        # Prepare training data
        X_train, y_train = self._prepare_features_and_target(df_train)
        model.fit(X_train, y_train)
        
        # Prepare test data and make predictions
        X_test, y_test = self._prepare_features_and_target(df_test)
        predicted_values = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            confidences = []
            for i, pred_val in enumerate(predicted_values):
                if pred_val in model.classes_:
                    class_idx = model.classes_.tolist().index(pred_val)
                    confidences.append(probabilities[i][class_idx])
                else:
                    confidences.append(0.0)
        else:
            confidences = [0.5] * len(predicted_values)  # Default confidence for models without predict_proba
        
        predictions = []
        for i, (pred_val, true_val, conf) in enumerate(zip(predicted_values, y_test, confidences)):
            predictions.append({
                'predicted_value': pred_val,
                'prediction': (pred_val == true_val),
                'confidence': conf
            })
        
        return pd.DataFrame(predictions)
    
    def _prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector."""
        # Handle categorical features with one-hot encoding
        if self.onehot_columns:
            if self.encoder is None:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                X_cat = self.encoder.fit_transform(df[self.onehot_columns])
            else:
                X_cat = self.encoder.transform(df[self.onehot_columns])
        else:
            X_cat = np.empty((len(df), 0))
        
        # Handle continuous features
        X_cont = df[self.non_onehot_columns].values if self.non_onehot_columns else np.empty((len(df), 0))
        
        # Combine features
        X = np.hstack([X_cont, X_cat])
        
        # Prepare target
        y = df[self.secret_column].values.ravel()
        if y.dtype == object:
            try:
                y = y.astype(int)
            except Exception:
                y = y.astype(str)
        
        return X, y

    def _detect_and_reclassify_correlated_categoricals(self, df_train: pd.DataFrame) -> Optional[OneToOnePredictor]:
        """
        Detect categorical features that should be treated as continuous and reclassify them.
        
        Simplified version that only checks for near-perfect correlation.
        
        Returns:
            OneToOnePredictor if a near-perfect relationship is found, None otherwise
        """
        if not self.onehot_columns or not self.secret_column or self.secret_column not in df_train.columns:
            return None
            
        target_values = df_train[self.secret_column]
        otop_candidates = []  # Store (column, correlation_ratio) pairs
        
        for cat_col in self.onehot_columns[:]:  # Create a copy to iterate over
            if cat_col not in df_train.columns:
                continue
                
            feature_values = df_train[cat_col]
            
            # Check for near-perfect correlation between categorical feature and target (95%+)
            correlation_ratio = self._calculate_correlation_ratio(feature_values, target_values)
            if correlation_ratio >= 0.95:
                self.logger.info(f"Reclassifying column '{cat_col}' as OneToOnePredictor due to near-perfect correlation ({correlation_ratio:.6f})")
                otop_candidates.append((cat_col, correlation_ratio))
                # Remove from onehot_columns and add to non_onehot_columns
                self.onehot_columns.remove(cat_col)
                if cat_col not in self.non_onehot_columns:
                    self.non_onehot_columns.append(cat_col)
                    
        # Select the best one-to-one predictor candidate (highest correlation ratio)
        if otop_candidates:
            best_col, best_ratio = max(otop_candidates, key=lambda x: x[1])
            otop = OneToOnePredictor(df_train, feature=best_col, target=self.secret_column)
            self.logger.info(f"Selected OneToOnePredictor for column '{best_col}' with correlation ratio: {best_ratio:.6f}")
            return otop
                
        return None

    def _calculate_correlation_ratio(self, feature_series: pd.Series, target_series: pd.Series) -> float:
        """
        Calculate the correlation ratio between feature and target.
        
        Returns the ratio of rows where the feature-target mapping is consistent
        with the most common mapping for each feature value.
        """
        if len(feature_series) == 0:
            return 0.0
            
        # Create a mapping from each feature value to its most common target value
        feature_to_mode_target = {}
        df_temp = pd.DataFrame({'feature': feature_series, 'target': target_series})
        
        for feat_val in feature_series.unique():
            target_vals = df_temp[df_temp['feature'] == feat_val]['target']
            mode_target = target_vals.mode()
            if len(mode_target) > 0:
                feature_to_mode_target[feat_val] = mode_target.iloc[0]
            else:
                feature_to_mode_target[feat_val] = target_vals.iloc[0]
        
        # Count how many rows match the expected mapping
        matches = 0
        for feat_val, target_val in zip(feature_series, target_series):
            if feat_val in feature_to_mode_target and feature_to_mode_target[feat_val] == target_val:
                matches += 1
                
        return matches / len(feature_series)

    def predict(self, index: int) -> Tuple[Any, float]:
        """
        Predict using the stored prediction data structure.
        
        Args:
            index: Index into the df_pred_conf DataFrame
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        if self.df_pred_conf is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        if index < 0 or index >= len(self.df_pred_conf):
            raise IndexError(f"Index {index} out of range for prediction data of length {len(self.df_pred_conf)}")
        
        row = self.df_pred_conf.iloc[index]
        return row['predicted_value'], row['confidence']