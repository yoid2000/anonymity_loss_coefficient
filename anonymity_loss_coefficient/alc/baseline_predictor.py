
import pandas as pd
from typing import Optional, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier


class BaselinePredictor:
    '''
    '''
    def __init__(self) -> None:
        self.model = None
        self.known_columns = None
        self.secret_col = None

    def build_model(self, df: pd.DataFrame, known_columns: Optional[List[str]] = None, secret_col: Optional[str] = None,  random_state: Optional[int] = None) -> None:
        if known_columns is not None:
            self.known_columns = known_columns
        if secret_col is not None:
            self.secret_col = secret_col
        X = df[list(self.known_columns)]
        y = df[self.secret_col]
        y = y.values.ravel()  # Convert to 1D array

        # Build and train the model
        try:
            self.model = RandomForestClassifier(random_state=random_state)
        except Exception as e:
            # raise error
            raise ValueError(f"Error building RandomForestClassifier {e}") from e

        y = y.astype(int)  # Convert to integer if necessary
        self.model.fit(X, y)

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
