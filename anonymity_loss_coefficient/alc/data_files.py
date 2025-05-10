import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from .defaults import defaults

class DataFiles:
    def __init__(self,
                 df_original: pd.DataFrame,
                 anon: Union[pd.DataFrame, List[pd.DataFrame]],
                 disc_max: int = defaults['disc_max'],
                 disc_bins: int = defaults['disc_bins'],
                 discretize_in_place: bool = defaults['discretize_in_place'],
                 max_cntl_size: int = defaults['max_cntl_size'],
                 max_cntl_percent: float = defaults['max_cntl_percent'],
                 random_state: Optional[int] = None,
                 ) -> None:
        self.disc_max = disc_max
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
        self.orig_all = self.orig_all.dropna()
        self.anon = [df.dropna() for df in self.anon]

        # Find numeric columns with more than disc_max unique values in df_orig_all
        numeric_cols = self.orig_all.select_dtypes(include=[np.number]).columns
        self.columns_for_discretization = [
            col for col in numeric_cols 
            if self.orig_all[col].dtype == 'float64' or self.orig_all[col].dtype == 'float32' or self.orig_all[col].nunique() > self.disc_max
        ]
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
        self._discretize_df(self.orig_all, discretizers)
        for i, df in enumerate(self.anon):
            self._discretize_df(df, discretizers)

        # set columns_to_encode to be all columns that are not integer and not
        # pre-discretized
        columns_to_encode = self.orig_all.select_dtypes(exclude=[np.int64, np.int32]).columns
        if self.discretize_in_place is False:
            columns_to_encode = [col for col in columns_to_encode if col not in self.columns_for_discretization]
        self._encoders = self._fit_encoders(columns_to_encode, [self.orig_all] + self.anon)

        self._transform_df(self.orig_all)
        for i, df in enumerate(self.anon):
            self._transform_df(df)

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

    def _discretize_df(self, df: pd.DataFrame, discretizers: Dict[str, KBinsDiscretizer]) -> None:
        for col in self.columns_for_discretization:
            if col in df.columns and col in discretizers:
                discretizer = discretizers[col]
                bin_indices = discretizer.transform(df[[col]]).astype(int).flatten()
                if self.discretize_in_place:
                    # Keep the column as integers
                    df.loc[:, col] = bin_indices
                else:
                    # Create a new column with integer values
                    df.loc[:, f"{col}__discretized"] = bin_indices

    def _transform_df(self, df: pd.DataFrame) -> None:
        for col, encoder in self._encoders.items():
            if col not in df.columns:
                continue
            # Transform the column using the encoder and keep it as integers
            transformed_values = encoder.transform(df[col]).astype(int)
            df.loc[:, col] = transformed_values


    def decode_value(self, column: str, encoded_value: int) -> Any:
        if column not in self._encoders:
            return encoded_value
        
        encoder: LabelEncoder = self._encoders[column]

        # Pass encoded_value directly to inverse_transform
        original_value = encoder.inverse_transform(np.array([encoded_value]))

        # Return the first element if only one value was decoded
        return original_value[0] if len(original_value) == 1 else original_value


    def _fit_encoders(self, columns_to_encode: List[str], dfs: List[pd.DataFrame]) -> Dict[str, LabelEncoder]:
        encoders = {col: LabelEncoder() for col in columns_to_encode}

        for col in columns_to_encode:
            # Collect values from all DataFrames that contain the column
            values = pd.concat(
                [df[col] for df in dfs if col in df.columns]
            ).unique()
            # Fit the encoder on the unique values
            encoders[col].fit(values)

        return encoders
