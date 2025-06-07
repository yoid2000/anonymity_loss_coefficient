import pandas as pd
from typing import Optional, List, Any, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# The following to suppress warnings from loky about CPU count
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

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
        self.selected_model = None

    def select_model(
        self,
        df: pd.DataFrame,
        known_columns: List[str],
        secret_column: str,
        column_classifications: Dict[str, str],
        random_state: Optional[int] = str
    ) -> None:
        self.known_columns = known_columns
        self.secret_column = secret_column

        # Identify categorical and continuous columns
        self.categorical_columns = [col for col in self.known_columns if column_classifications.get(col) == 'categorical']
        self.continuous_columns = [col for col in self.known_columns if column_classifications.get(col) == 'continuous']

        # One-hot encode categorical columns for all models except CatBoost and LGBM (if available)
        if self.categorical_columns:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat = self.encoder.fit_transform(df[self.categorical_columns])
        else:
            self.encoder = None
            X_cat = np.empty((len(df), 0))

        X_cont = df[self.continuous_columns].values if self.continuous_columns else np.empty((len(df), 0))
        X = np.hstack([X_cont, X_cat])
        y = df[self.secret_column].values.ravel()
        if y.dtype == object:
            try:
                y = y.astype(int)
            except Exception:
                y = y.astype(str)

        # Build ensemble of models
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
        if XGBClassifier is not None:
            models.append(("XGB", XGBClassifier(
                eval_metric='logloss',
                random_state=random_state
            )))
        if LGBMClassifier is not None:
            models.append(("LGBM", LGBMClassifier(
                verbose=-1,
                random_state=random_state
            )))
        if CatBoostClassifier is not None:
            models.append(("CatBoost", CatBoostClassifier(
                logging_level="Silent",
                verbose=0,
                random_state=random_state
            )))

        best_score = -np.inf
        best_model = None
        best_model_name = None

        for name, model in models:
            # For CatBoost and LGBM, use original df with categorical columns as strings
            if name == "CatBoost" and CatBoostClassifier is not None:
                X_cb = df[self.known_columns]
                try:
                    scores = cross_val_score(model, X_cb, y, cv=3, scoring='neg_log_loss')
                except Exception:
                    continue
            elif name == "LGBM" and LGBMClassifier is not None:
                X_lgbm = df[self.known_columns]
                try:
                    scores = cross_val_score(model, X_lgbm, y, cv=3, scoring='neg_log_loss')
                except Exception:
                    continue
            else:
                try:
                    scores = cross_val_score(model, X, y, cv=3, scoring='neg_log_loss')
                except Exception:
                    continue
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_model_name = name

        self.selected_model = best_model
        return best_model_name

    def build_model(
        self,
        df: pd.DataFrame,
        random_state: Optional[int] = None
    ) -> None:
        # One-hot encode categorical columns for all models except CatBoost and LGBM
        if self.selected_model is None:
            raise ValueError("No model has been selected. Call select_model() first.")

        model_name = type(self.selected_model).__name__

        if model_name in ["CatBoostClassifier", "LGBMClassifier"]:
            X = df[self.known_columns]
        else:
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

        self.model = self.selected_model
        self.model.fit(X, y)

    def predict(self, df_row: pd.DataFrame) -> Tuple[Any, float]:
        if self.model is None:
            raise ValueError("Model has not been built yet")

        model_name = type(self.model).__name__

        if model_name in ["CatBoostClassifier", "LGBMClassifier"]:
            X = df_row[self.known_columns]
        else:
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