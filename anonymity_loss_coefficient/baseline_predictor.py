
import pandas as pd
from typing import Optional, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier


class BaselinePredictor:
    '''
    '''
    def __init__(self, df_orig: pd.DataFrame) -> None:
        self.df_orig = df_orig
        self.model = None

    def build_model(self, known_columns: List[str], secret_col: str,  random_state: Optional[int] = None) -> None:
        X = self.df_orig[list(known_columns)]
        y = self.df_orig[secret_col]
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
