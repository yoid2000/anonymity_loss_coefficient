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
                 disc_bins: int,
                 discretize_in_place: bool,
                 max_cntl_size: int,
                 max_cntl_percent: float,
                 logger: logging.Logger,
                 random_state: Optional[int] = None,
                 ) -> None:
        self.logger = logger
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
        print(f"20: {self.orig_all['GoodStudent'].unique()}")
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
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
                        df.loc[:, f"{col}__discretized"] = bin_indices    

    def _transform_df(self, df: pd.DataFrame) -> None:
        for col, encoder in self._encoders.items():
            if col not in df.columns:
                continue
            # Transform the column using the encoder and keep it as integers
            transformed_values = encoder.transform(df[col].astype(str))
            # cast to int if the column is bool or datetime
            if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
                    warnings.simplefilter("ignore", FutureWarning)
                    df[col] = df[col].astype('int64')
            df.loc[:, col] = transformed_values.astype(int)


    def decode_value(self, column: str, encoded_value: int) -> Any:
        if column not in self._encoders:
            return encoded_value
        
        encoder: LabelEncoder = self._encoders[column]

        # =================================================================
        # START DEBUGGING CODE - REMOVE AFTER ISSUE IS RESOLVED
        # =================================================================
        try:
            # Pass encoded_value directly to inverse_transform
            original_value = encoder.inverse_transform(np.array([encoded_value]))
            
            # Return the first element if only one value was decoded
            return original_value[0] if len(original_value) == 1 else original_value
            
        except ValueError as e:
            print("\n" + "="*80)
            print("DEBUGGING: LabelEncoder inverse_transform error caught!")
            print("="*80)
            print(f"Error message: {e}")
            print(f"Column: {column}")
            print(f"Encoded value attempting to decode: {encoded_value}")
            print(f"Encoded value type: {type(encoded_value)}")
            print(f"Encoder classes: {encoder.classes_}")
            print(f"Number of encoder classes: {len(encoder.classes_)}")
            print(f"Min encoder class index: {0}")
            print(f"Max encoder class index: {len(encoder.classes_) - 1}")
            print(f"Is encoded_value in valid range? {0 <= encoded_value < len(encoder.classes_)}")
            
            # Check if orig_all, cntl, and orig have this column
            print(f"\nColumn '{column}' presence:")
            print(f"  In orig_all: {column in self.orig_all.columns if hasattr(self, 'orig_all') else 'orig_all not available'}")
            if hasattr(self, 'orig_all') and column in self.orig_all.columns:
                print(f"  orig_all[{column}] unique values: {sorted(self.orig_all[column].unique())}")
                print(f"  orig_all[{column}] value counts: {self.orig_all[column].value_counts().to_dict()}")
            
            if hasattr(self, 'cntl') and self.cntl is not None:
                print(f"  In cntl: {column in self.cntl.columns}")
                if column in self.cntl.columns:
                    print(f"  cntl[{column}] unique values: {sorted(self.cntl[column].unique())}")
                    print(f"  cntl[{column}] value counts: {self.cntl[column].value_counts().to_dict()}")
            else:
                print(f"  cntl is None or not available")
            
            if hasattr(self, 'orig') and self.orig is not None:
                print(f"  In orig: {column in self.orig.columns}")
                if column in self.orig.columns:
                    print(f"  orig[{column}] unique values: {sorted(self.orig[column].unique())}")
                    print(f"  orig[{column}] value counts: {self.orig[column].value_counts().to_dict()}")
            else:
                print(f"  orig is None or not available")
            
            # Check anon dataframes
            if hasattr(self, 'anon'):
                for i, anon_df in enumerate(self.anon):
                    if column in anon_df.columns:
                        print(f"  In anon[{i}]: True")
                        print(f"  anon[{i}][{column}] unique values: {sorted(anon_df[column].unique())}")
                        print(f"  anon[{i}][{column}] value counts: {anon_df[column].value_counts().to_dict()}")
                    else:
                        print(f"  In anon[{i}]: False")
            
            # NEW: Add stack trace and variable dumps
            print("\n" + "-"*60)
            print("EXCEPTION STACK TRACE:")
            print("-"*60)
            traceback.print_exc()
            
            print("\n" + "-"*60)
            print("FULL CALL STACK (complete execution path):")
            print("-"*60)
            # Get the complete stack from the beginning of execution
            full_stack = traceback.extract_stack()
            for frame in full_stack:
                print(f"File: {frame.filename}")
                print(f"  Line {frame.lineno}: {frame.name}")
                print(f"  Code: {frame.line}")
                print()
            
            print("\n" + "-"*60)
            print("FORMATTED FULL STACK:")
            print("-"*60)
            # Alternative format - more compact
            formatted_stack = traceback.format_stack()
            for line in formatted_stack:
                print(line.rstrip())
            
            print("\n" + "-"*60)
            print("FRAME-SPECIFIC LOCAL VARIABLES:")
            print("-"*60)
            # Walk through the stack frames to find specific methods
            current_frame = inspect.currentframe()
            frame = current_frame
            frame_count = 0
            
            while frame is not None and frame_count < 20:  # Limit to avoid infinite loops
                frame_info = inspect.getframeinfo(frame)
                function_name = frame.f_code.co_name
                filename = frame_info.filename
                
                print(f"Frame {frame_count}: {function_name} in {filename}:{frame_info.lineno}")
                
                # Check if this is the _model_prediction method we're interested in
                if function_name == '_model_prediction':
                    print(f"  *** FOUND _model_prediction METHOD - CAPTURING LOCALS ***")
                    frame_locals = dict(frame.f_locals)
                    
                    # Remove large objects to avoid overwhelming output
                    filtered_locals = {}
                    for key, value in frame_locals.items():
                        if key == 'self':
                            filtered_locals[key] = f"<DataFiles/ALCManager object at {id(value)}>"
                        elif isinstance(value, pd.DataFrame):
                            filtered_locals[key] = f"<DataFrame shape={value.shape}>"
                        elif isinstance(value, np.ndarray):
                            filtered_locals[key] = f"<ndarray shape={value.shape}, dtype={value.dtype}>"
                        elif len(str(value)) > 200:
                            filtered_locals[key] = f"<{type(value).__name__}> {str(value)[:200]}..."
                        else:
                            filtered_locals[key] = value
                    
                    pprint.pprint(filtered_locals, width=80, depth=3)
                    print()
                
                # Also capture any other interesting methods
                elif function_name in ['predictor', 'run_one_attack', 'decode_value']:
                    print(f"  Capturing locals for {function_name}:")
                    frame_locals = dict(frame.f_locals)
                    
                    # Filter locals for readability
                    filtered_locals = {}
                    for key, value in frame_locals.items():
                        if key == 'self':
                            filtered_locals[key] = f"<{type(value).__name__} object>"
                        elif isinstance(value, (pd.DataFrame, pd.Series)):
                            filtered_locals[key] = f"<{type(value).__name__} shape={getattr(value, 'shape', 'unknown')}>"
                        elif isinstance(value, np.ndarray):
                            filtered_locals[key] = f"<ndarray shape={value.shape}>"
                        elif len(str(value)) > 100:
                            filtered_locals[key] = f"<{type(value).__name__}> {str(value)[:100]}..."
                        else:
                            filtered_locals[key] = value
                    
                    pprint.pprint(filtered_locals, width=80, depth=2)
                    print()
                else:
                    print(f"  (skipping locals for {function_name})")
                
                frame = frame.f_back
                frame_count += 1
            
            # Clean up frame references to avoid memory leaks
            del current_frame
            del frame
            
            print("\n" + "-"*60)
            print("LOCAL VARIABLES IN decode_value METHOD:")
            print("-"*60)
            local_vars = locals().copy()
            # Remove large objects to avoid overwhelming output
            if 'self' in local_vars:
                del local_vars['self']
            # Remove frame references we just created
            for key in ['current_frame', 'frame', 'frame_locals', 'filtered_locals']:
                if key in local_vars:
                    del local_vars[key]
            pprint.pprint(local_vars, width=80, depth=3)
            
            print("\n" + "-"*60)
            print("SELF OBJECT STATE (key attributes):")
            print("-"*60)
            self_state = {
                'disc_max': getattr(self, 'disc_max', 'Not set'),
                'disc_bins': getattr(self, 'disc_bins', 'Not set'),
                'discretize_in_place': getattr(self, 'discretize_in_place', 'Not set'),
                'cntl_size': getattr(self, 'cntl_size', 'Not set'),
                'cntl_block_index': getattr(self, 'cntl_block_index', 'Not set'),
                'columns_for_discretization': getattr(self, 'columns_for_discretization', 'Not set'),
                'new_discretized_columns': getattr(self, 'new_discretized_columns', 'Not set'),
                'original_columns': getattr(self, 'original_columns', 'Not set'),
                'encoders_keys': list(self._encoders.keys()) if hasattr(self, '_encoders') else 'Not set',
                'column_classification_keys': list(self.column_classification.keys()) if hasattr(self, 'column_classification') else 'Not set',
                'orig_all_shape': self.orig_all.shape if hasattr(self, 'orig_all') else 'Not set',
                'orig_shape': self.orig.shape if hasattr(self, 'orig') and self.orig is not None else 'None or Not set',
                'cntl_shape': self.cntl.shape if hasattr(self, 'cntl') and self.cntl is not None else 'None or Not set',
                'anon_count': len(self.anon) if hasattr(self, 'anon') else 'Not set',
            }
            pprint.pprint(self_state, width=80)
            
            print("="*80)
            print("HALTING EXECUTION DUE TO DEBUGGING")
            print("="*80)
            
            # Halt execution
            import sys
            sys.exit(1)
        # =================================================================
        # END DEBUGGING CODE - REMOVE AFTER ISSUE IS RESOLVED
        # =================================================================


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
            self.logger.info(f"Adjusting cntl and orig to ensure all classes are present in both sets.")
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