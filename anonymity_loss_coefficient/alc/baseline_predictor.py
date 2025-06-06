import pandas as pd
from typing import Optional, List, Any, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class BaselinePredictor:
    '''
    '''
    def __init__(self) -> None:
        self.model = None
        self.known_columns = None
        self.secret_column = None
        self.encoder = None
        self.categorical_columns = []
        self.continuous_columns = []

    def build_model(
        self,
        df: pd.DataFrame,
        known_columns: Optional[List[str]] = None,
        secret_column: Optional[str] = None,
        column_classifications: Optional[Dict[str, str]] = None,
        random_state: Optional[int] = None
    ) -> None:
        # Note that, despite the Optional parameters, known_columns, secret_column, and column_classifications
        # are always provided on the first call.
        if known_columns is not None:
            self.known_columns = known_columns
        if secret_column is not None:
            self.secret_column = secret_column

        # Identify categorical and continuous columns
        if column_classifications is not None:
            self.categorical_columns = [col for col in self.known_columns if column_classifications.get(col) == 'categorical']
            self.continuous_columns = [col for col in self.known_columns if column_classifications.get(col) == 'continuous']

        # One-hot encode categorical columns
        if self.categorical_columns:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat = self.encoder.fit_transform(df[self.categorical_columns])
        else:
            X_cat = np.empty((len(df), 0))

        # Use continuous columns as-is
        X_cont = df[self.continuous_columns].values if self.continuous_columns else np.empty((len(df), 0))

        # Concatenate features
        X = np.hstack([X_cont, X_cat])
        y = df[self.secret_column].values.ravel()  # Convert to 1D array

        # Build and train the model
        try:
            self.model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=10,
                max_features='sqrt',
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                oob_score=True
            )
        except Exception as e:
            raise ValueError(f"Error building RandomForestClassifier {e}") from e

        y = y.astype(int)  # Convert to integer if necessary
        self.model.fit(X, y)

    def predict(self, df_row: pd.DataFrame) -> Tuple[Any, float]:
        if self.model is None:
            raise ValueError("Model has not been built yet")

        # Prepare features for prediction
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