import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.stats import norm

class YourClass:
    def __init__(self):
        self.model = None  # This should be initialized properly in your class

    def build_model(self, known_columns: List[str], target_col: str) -> None:
        # Your existing build_model code
        pass

    def predict(self, row: pd.Series) -> Tuple[Any, Any]:
        # Your existing predict code
        pass

    def compute_wilson_score_interval(self, predictions: List[bool], confidence_level: float = 0.95) -> Tuple[float, float, float]:
        n = len(predictions)
        if n == 0:
            return 0.0, 0.0, 0.0

        precision = np.mean(predictions)
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        center_adjusted_probability = precision + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((precision * (1 - precision) + z**2 / (4 * n)) / n)
        lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator

        return precision, lower_bound, upper_bound

    def make_predictions_until_confidence(self, data: pd.DataFrame, confidence_level: float = 0.95, tolerance: float = 0.05) -> List[bool]:
        predictions = []
        for _, row in data.iterrows():
            prediction, _ = self.predict(row)
            predictions.append(prediction == row['true_value'])

            precision, lower_bound, upper_bound = self.compute_wilson_score_interval(predictions, confidence_level)
            print(f"Precision: {precision}, Confidence Interval: [{lower_bound}, {upper_bound}]")

            # Stop making predictions if the width of the confidence interval is within the tolerance
            if (upper_bound - lower_bound) <= tolerance:
                break

        return predictions

# Usage example
your_class_instance = YourClass()
# Assume data is a DataFrame with the necessary columns
data = pd.DataFrame({
    'true_value': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    # Add other columns as needed for prediction
})
predictions = your_class_instance.make_predictions_until_confidence(data, confidence_level=0.95, tolerance=0.05)