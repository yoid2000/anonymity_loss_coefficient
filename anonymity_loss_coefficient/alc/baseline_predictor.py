import pandas as pd
from typing import Optional, List, Any, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, log_loss
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
        random_state: Optional[int] = None
    ) -> None:
        self.known_columns = known_columns
        self.secret_column = secret_column

        self.categorical_columns = [col for col in self.known_columns if column_classifications.get(col) == 'categorical']
        self.continuous_columns = [col for col in self.known_columns if column_classifications.get(col) == 'continuous']

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
        # Add XGB, LGBM, CatBoost as before if available...
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
            try:
                if name in ["CatBoost", "LGBM"]:
                    # Use DataFrame with column names for these models
                    X_cv = df[self.known_columns]
                else:
                    X_cv = X
                scores = cross_val_score(model, X_cv, y, cv=cv, scoring=log_loss_scorer)
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) == 0:
                    continue  # Skip models with all nan scores
                mean_score = valid_scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = name
            except Exception:
                continue  # Skip models that error out

        # Fallback: If no model was selected, use the first model in the list
        if best_model is None:
            best_model_name, best_model = models[0]

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
            print(f"******************* Predicting with model: {model_name}")
            # Ensure df_row is a DataFrame with the correct columns
            if not isinstance(df_row, pd.DataFrame):
                df_row = pd.DataFrame([df_row], columns=self.known_columns)
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