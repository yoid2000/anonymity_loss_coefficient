import numpy as np
import pandas as pd
import random
from typing import Dict, List, Union, Any, Optional
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import gc
import logging
import traceback  # Added for debugging
import pprint    # Added for debugging
import inspect   # Added for frame inspection

# The following to suppress warnings from loky about CPU count
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Replace 8 with your number of logical cores

class DataFiles:
    def __init__(self,
                 df_original: pd.DataFrame,
                 anon: Union[pd.DataFrame, List[pd.DataFrame]],
                 disc_max: int,
                 disc_min: int,
                 disc_bins: int,
                 discretize_in_place: bool,
                 max_cntl_size: int,
                 max_cntl_percent: float,
                 logger: logging.Logger,
                 random_state: Optional[int] = None,
                 ) -> None:
        self.logger = logger
        self.disc_max = disc_max
        self.disc_min = disc_min
        self.disc_bins = disc_bins
        self.discretize_in_place = discretize_in_place
        self.max_cntl_size = max_cntl_size
        self.max_cntl_percent = max_cntl_percent
        self.random_state = random_state
        self.cntl_size = None
        # self.cntl will contain the part of the raw data used for attacks.
        # self.orig is the original data minus self.cntl
        self.orig = None
        self.cntl = None
        # self.orig_all is the original data. self.cntl and self.orig are taken from this.
        self.orig_all = df_original
        self.original_columns = df_original.columns.tolist()
        self.cntl_block_index = -1
        self.columns_for_discretization = []
        self._encoders = {}
        self.column_classification = {}

        if isinstance(anon, pd.DataFrame):
            self.anon = [anon]
        elif isinstance(anon, list) and all(isinstance(df, pd.DataFrame) for df in anon):
            self.anon = anon
        else:
            raise ValueError("anon must be either a pandas DataFrame or a list of pandas DataFrames")
        orig_rows_before_dropna = self.orig_all.shape[0]
        self.orig_all = self.orig_all.dropna()
        if self.orig_all.shape[0] == 0:
            raise ValueError("Original DataFrame is empty after dropping NaN values.")
        if orig_rows_before_dropna - self.orig_all.shape[0] > 0:
            print(f"Dropped {orig_rows_before_dropna - self.orig_all.shape[0]} rows with NaN values from original data")
        anon_rows_before_dropna = [df.shape[0] for df in self.anon]
        self.anon = [df.dropna() for df in self.anon]

        for i, df in enumerate(self.anon):
            if anon_rows_before_dropna[i] - df.shape[0] > 0:
                print(f"Dropped {anon_rows_before_dropna[i] - df.shape[0]} rows with NaN values from anonymized data at index {i}")

        self.columns_for_discretization = self._select_columns_for_discretization()
        self.new_discretized_columns = [f"{col}__discretized" for col in self.columns_for_discretization]

        # Initialize the dictionary to store discretizers
        discretizers = {}

        # Iterate over columns for discretization
        for col in self.columns_for_discretization:
            # Find the min and max values across all DataFrames
            min_val = min(df[col].min() for df in [self.orig_all] + self.anon if col in df)
            max_val = max(df[col].max() for df in [self.orig_all] + self.anon if col in df)

            # Handle cases where min_val or max_val is NaN
            if pd.isna(min_val) or pd.isna(max_val):
                raise ValueError(f"Column {col} contains only NaN values across all DataFrames.")

            # Handle cases where min_val == max_val
            if min_val == max_val:
                bin_edges = np.linspace(min_val - 1, max_val + 1, num=self.disc_bins + 1)
            else:
                bin_edges = np.linspace(min_val, max_val, num=self.disc_bins + 1)

            # Create and fit the discretizer
            discretizer = KBinsDiscretizer(n_bins=self.disc_bins, encode='ordinal', strategy='uniform')
            combined_col_data = pd.concat([df[[col]] for df in [self.orig_all] + self.anon if col in df])
            discretizer.fit(combined_col_data)

            # Manually set the bin edges
            discretizer.bin_edges_ = np.array([bin_edges])

            # Store the discretizer
            discretizers[col] = discretizer


        # Discretize the columns in df_orig_all and anon using the same bin widths
        self.orig_all = self._discretize_df(self.orig_all, discretizers)
        for i, df in enumerate(self.anon):
            self.anon[i] = self._discretize_df(df, discretizers)

        # set columns_to_encode to be all columns that are not integer and not
        # pre-discretized
        columns_to_encode = self.orig_all.select_dtypes(exclude=[np.int64, np.int32]).columns
        if self.discretize_in_place is False:
            columns_to_encode = [col for col in columns_to_encode if col not in self.columns_for_discretization]

        self._encoders = self._fit_encoders(columns_to_encode, [self.orig_all] + self.anon)

        self.orig_all = self._transform_df(self.orig_all)
        for i, df in enumerate(self.anon):
            self.anon[i] = self._transform_df(df)

        # As a final step, we want to classify all columns as categorical or continuous
        for col in self.orig_all.columns:
            if self.discretize_in_place is True or col not in self.columns_for_discretization:
                self.column_classification[col] = 'categorical'
            else:
                self.column_classification[col] = 'continuous'

        self._init_cntl_builder()
        
    def _init_cntl_builder(self) -> None:
        '''
        The self.orig dataframe is used to build the baseline model. When dealing with
        small datasets, we want self.orig to be large enough to build a decent model.
        Let's somewhat arbitrarily declare that we want no more than 10% of the data to be
        in self.cntl. Also no bigger than 1000 rows.
        '''
        if len(self.orig_all) * self.max_cntl_percent >= self.max_cntl_size:
            self.cntl_size = self.max_cntl_size
        else:
            self.cntl_size = int(len(self.orig_all) * self.max_cntl_percent) + 1

    def assign_first_cntl_block(self) -> bool:
        self.cntl_block_index = -1
        return self.assign_next_cntl_block()

    def assign_next_cntl_block(self) -> bool:
        self.cntl_block_index += 1
        row_index = self.cntl_block_index * self.cntl_size
        if row_index >= len(self.orig_all):
            self.cntl = None
            self.orig = None
            return False
        if self.orig is not None:
            # TODO probably delete this
            del self.orig
            del self.cntl
            gc.collect()
        self.cntl = self.orig_all.iloc[row_index:row_index + self.cntl_size]
        # Shuffle the control data to ensure randomness
        self.cntl = self.cntl.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.orig = self.orig_all.drop(self.orig_all.index[row_index:row_index + self.cntl_size])
        return True

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

    def _discretize_df(self, df: pd.DataFrame, discretizers: Dict[str, KBinsDiscretizer]) -> pd.DataFrame:
        for col in self.columns_for_discretization:
            if col in df.columns and col in discretizers:
                discretizer = discretizers[col]
                bin_indices = discretizer.transform(df[[col]]).astype(int).flatten()
                if self.discretize_in_place:
                    # Keep the column as integers
                    df.loc[:, col] = bin_indices
                else:
                    # Create a new column with integer values
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
                        df.loc[:, f"{col}__discretized"] = bin_indices    
        return df

    def _transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame by encoding specified columns using fitted encoders.
        
        This method applies the LabelEncoders fitted in _fit_encoders() to transform
        the values in the specified columns to integer codes.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame with encoded columns
        """
        # Create a copy to avoid modifying the original DataFrame
        df_transformed = df.copy()
        
        for col, encoder in self._encoders.items():
            if col not in df_transformed.columns:
                # Column not in this DataFrame, skip
                continue
            
            # Get the column values, handling NaN values
            column_values = df_transformed[col]
            non_null_mask = column_values.notna()
            
            if not non_null_mask.any():
                # All values are NaN, skip this column
                self.logger.warning(f"All values in column '{col}' are NaN, skipping encoding")
                continue
            
            # Transform only non-null values
            non_null_values = column_values[non_null_mask]
            
            try:
                # Apply the encoder to non-null values
                encoded_values = encoder.transform(non_null_values)
                
                # Create a new series with the same index as the original
                # Initialize with -1 for all values, then set the encoded ones
                encoded_series = pd.Series(-1, index=df_transformed.index, dtype='int64')
                
                # Convert encoded_values to int64 and assign
                if hasattr(encoded_values, 'astype'):
                    # It's a numpy array
                    encoded_series[non_null_mask] = encoded_values.astype('int64')
                else:
                    # Convert to numpy array first
                    encoded_series[non_null_mask] = np.array(encoded_values, dtype='int64')
                
                # Replace the column in the DataFrame
                df_transformed[col] = encoded_series
                
                # Debug logging
                unique_original = sorted(non_null_values.unique(), key=str) if len(non_null_values.unique()) <= 10 else f"{len(non_null_values.unique())} unique values"
                self.logger.debug(f"Transformed column '{col}': {unique_original} -> encoded values")
                
            except ValueError as e:
                # Handle unknown values that weren't in the training data
                self.logger.error(f"Error encoding column '{col}': {e}")
                
                # Try to handle unknown values by finding the closest match or using a default
                unknown_values = set(non_null_values) - set(encoder.classes_)
                if unknown_values:
                    self.logger.warning(f"Unknown values in column '{col}': {unknown_values}")
                    self.logger.warning("These values were not seen during encoder fitting")
                
                # For now, re-raise the error - in production you might want to handle this differently
                raise
        
        return df_transformed

    def decode_value(self, column: str, encoded_value: Any) -> Any:
        """
        Decode an encoded value back to its original form.
        
        Args:
            column: The column name
            encoded_value: The encoded integer value
            
        Returns:
            The original value before encoding
            
        Raises:
            ValueError: If the encoded value is invalid for the column
        """
        if column not in self._encoders:
            self.logger.debug(f"Column '{column}' not in encoders, returning as-is: {encoded_value}")
            return encoded_value
        
        encoder = self._encoders[column]
        
        try:
            # Convert encoded_value to standard Python int to ensure sklearn compatibility
            if pd.isna(encoded_value):
                # Handle NaN values - return None or the original NaN
                return None
            
            # Convert to standard Python int (handles pandas Int64, numpy types, etc.)
            if hasattr(encoded_value, 'item'):
                # Handle numpy scalars
                int_value = int(encoded_value.item())
            else:
                # Handle pandas/python types
                int_value = int(encoded_value)
            
            # Validate the encoded value is within valid range
            if int_value < 0 or int_value >= len(encoder.classes_):
                raise ValueError(f"Encoded value {int_value} is out of range for column '{column}'. "
                               f"Valid range: 0 to {len(encoder.classes_) - 1}")
            
            # Use inverse_transform with proper numpy array
            original_value = encoder.inverse_transform(np.array([int_value], dtype=np.int32))[0]
            
            self.logger.debug(f"Decoded {column}: {encoded_value} -> {original_value}")
            return original_value
            
        except (ValueError, TypeError, OverflowError) as e:
            self.logger.error(f"Error decoding value {encoded_value} for column '{column}': {e}")
            self.logger.error(f"Encoder classes: {encoder.classes_}")
            self.logger.error(f"Type of encoded_value: {type(encoded_value)}")
            raise ValueError(f"Cannot decode value {encoded_value} for column '{column}': {e}")
        except IndexError as e:
            self.logger.error(f"Index error decoding value {encoded_value} for column '{column}': {e}")
            self.logger.error(f"Encoder classes: {encoder.classes_}")
            self.logger.error(f"Valid range: 0 to {len(encoder.classes_) - 1}")
            raise ValueError(f"Encoded value {encoded_value} is out of range for column '{column}'")
            
    def _fit_encoders(self, columns_to_encode: List[str], dfs: List[pd.DataFrame]) -> Dict[str, LabelEncoder]:
        """
        Fit LabelEncoders for specified columns across all DataFrames.
        
        This method ensures that all DataFrames use the same encoding for each column
        by fitting each encoder on the combined unique values from all DataFrames.
        
        Args:
            columns_to_encode: List of column names to encode
            dfs: List of DataFrames (typically [self.orig_all] + self.anon)
            
        Returns:
            Dictionary mapping column names to fitted LabelEncoder objects
        """
        encoders = {}
        
        for col in columns_to_encode:
            # Collect all unique values from all DataFrames for this column
            all_values = []
            
            for df in dfs:
                if col in df.columns:
                    # Get unique values from this DataFrame
                    unique_vals = df[col].dropna().unique()
                    all_values.extend(unique_vals)
            
            # If no values found, skip this column
            if not all_values:
                self.logger.warning(f"No values found for column '{col}' across all DataFrames")
                continue
            
            # Get unique values across all DataFrames
            unique_values = pd.Series(all_values).unique()
            
            # Sort for consistent encoding order (helps with debugging and reproducibility)
            if unique_values.dtype == 'object':
                # For string/object columns, sort as strings
                unique_values = sorted(unique_values, key=str)
            else:
                # For numeric/boolean columns, sort normally
                unique_values = sorted(unique_values)
            
            # Create and fit the encoder
            encoder = LabelEncoder()
            encoder.fit(unique_values)
            
            encoders[col] = encoder
            
            # Debug logging
            self.logger.debug(f"Fitted encoder for column '{col}': {len(unique_values)} unique values")
            self.logger.debug(f"  Encoder classes: {encoder.classes_}")
        
        return encoders

    def check_and_fix_target_classes(self, secret_col: str) -> bool:
        all_classes = set(self.orig_all[secret_col].unique())
        if self.cntl is None:
            raise ValueError("self.cntl is None. Assign a control block before checking target classes.")
        test_classes = set(self.cntl[secret_col].unique())
        if self.orig is None:
            raise ValueError("self.orig is None. Assign a control block before checking target classes.")
        train_classes = set(self.orig[secret_col].unique())
        
        if train_classes == all_classes and test_classes == all_classes:
            return True

        # It can happen, esp. in small datasets, that the "test data" or the
        # "training data" does not contain all of the classes. When this occurs,
        # rather than raise an error, we override the choice of cntl and orig.
        # This can result in the same "victim" occasionally being attacked. This
        # is not perfect, but won't distort the results by much.
        # Perform stratified sampling to ensure all classes are in both sets
        
        try:
            self.logger.info(f"Adjusting test set (cntl) and training set (orig) to ensure all classes are present in both sets.")
            # Get indices for stratified split
            indices = self.orig_all.index.tolist()
            target_values = self.orig_all[secret_col].tolist()
            
            # Calculate test size to match len(self.cntl)
            test_size = len(self.cntl) / len(self.orig_all)
            
            # Generate a random seed for each call to ensure different splits
            random_seed = random.randint(0, 2**32 - 1)
            
            # Perform stratified split with a random seed for each call
            train_indices, test_indices = train_test_split(
                indices, 
                test_size=test_size, 
                stratify=target_values, 
                random_state=random_seed
            )
            
            # Update cntl and orig with stratified samples
            self.cntl = self.orig_all.loc[test_indices].reset_index(drop=True)
            self.orig = self.orig_all.loc[train_indices].reset_index(drop=True)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to perform stratified sampling for target classes: {e}")
            return False

    def _select_columns_for_discretization(self) -> List[str]:
        # Select all columns with floating-point dtypes
        float_cols = self.orig_all.select_dtypes(include=[np.floating]).columns.tolist()
        
        # Also check integer columns for "smooth" distributions that suggest continuity
        int_cols = self.orig_all.select_dtypes(include=[np.integer]).columns
        smooth_int_cols = []
        
        for col in int_cols:
            if self._is_smooth_distribution(col):
                smooth_int_cols.append(col)
                self.logger.debug(f"Column '{col}' classified as continuous based on smooth distribution")
        
        return float_cols + smooth_int_cols

    def _is_smooth_distribution(self, col: str) -> bool:
        """
        Determine if an integer column has a smooth distribution suggesting continuity.
        
        Uses multiple heuristics:
        1. Minimum unique values (less than disc_min, typically 10)
        2. Maximum unique values (less than disc_max, typically 50)
        3. Value density - checks if values are relatively dense in the range
        4. Distribution shape - checks for monotonic or unimodal patterns
        
        Args:
            col: Column name to analyze
            
        Returns:
            True if the distribution appears smooth/continuous
        """
        series = self.orig_all[col].dropna()
        
        if len(series) == 0:
            return False
            
        unique_vals = series.unique()
        n_unique = len(unique_vals)
        
        # Must have at least self.disc_min unique values to be considered continuous
        if n_unique < self.disc_min:
            return False
            
        # Must have fewer than disc_max unique values (otherwise would be selected anyway)
        if n_unique >= self.disc_max:
            return True  # Would be selected by the old logic anyway
        
        # Check value density: are the values relatively dense in their range?
        min_val, max_val = series.min(), series.max()
        value_range = max_val - min_val + 1
        density = n_unique / value_range
        
        # If density is very low, likely categorical (e.g., sparse IDs)
        if density < 0.3:
            return False
            
        # Check distribution shape for smoothness
        value_counts = series.value_counts().sort_index()
        frequencies = np.array(value_counts.values)
        
        # Test for monotonic decrease (common in continuous data)
        if self._is_monotonic_trend(frequencies):
            return True
            
        # Test for unimodal distribution (bell-shaped or similar)
        if self._is_unimodal(frequencies):
            return True
            
        # If we have high density and reasonable number of unique values,
        # lean towards continuous
        if density > 0.7 and n_unique >= 15:
            return True
            
        return False
    
    def _is_monotonic_trend(self, frequencies: np.ndarray) -> bool:
        """Check if frequencies show a generally monotonic trend (allowing some noise)."""
        if len(frequencies) < 5:
            return False
        
        # Simple check: is the trend generally decreasing or increasing?
        n = len(frequencies)
        decreasing = sum(frequencies[i] >= frequencies[i+1] for i in range(n-1))
        increasing = sum(frequencies[i] <= frequencies[i+1] for i in range(n-1))
        
        # Allow some noise: 70% of transitions should follow the trend
        return decreasing / (n-1) > 0.7 or increasing / (n-1) > 0.7
    
    def _is_unimodal(self, frequencies: np.ndarray) -> bool:
        """Check if the distribution is roughly unimodal (single peak)."""
        if len(frequencies) < 5:
            return False
            
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(frequencies) - 1):
            if frequencies[i] > frequencies[i-1] and frequencies[i] > frequencies[i+1]:
                peaks.append(i)
        
        # Unimodal should have 0-2 peaks (allowing for some noise)
        if len(peaks) <= 2:
            return True
            
        # Alternative: check if most of the mass is concentrated in the middle
        n = len(frequencies)
        middle_third_start = n // 3
        middle_third_end = 2 * n // 3
        middle_mass = sum(frequencies[middle_third_start:middle_third_end])
        total_mass = sum(frequencies)
        
        # If middle third contains >40% of the mass, likely unimodal
        return middle_mass / total_mass > 0.4