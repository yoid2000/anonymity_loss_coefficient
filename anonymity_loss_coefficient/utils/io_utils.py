import pandas as pd
import os
from typing import Optional
from charset_normalizer import from_path

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and detects its encoding using charset_normalizer.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the contents of the CSV file.
    """
    result = from_path(file_path).best()
    encoding = result.encoding
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except MemoryError:
        # Propagate MemoryError without printing/logging
        raise RuntimeError(f"MemoryError while reading {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path} with encoding {encoding}: {e}")

def read_parquet(file_path: str) -> pd.DataFrame:
    """
    Reads a Parquet file.

    Args:
        file_path (str): Path to the Parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the contents of the Parquet file.
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except MemoryError:
        # Propagate MemoryError without printing/logging
        raise RuntimeError(f"MemoryError while reading {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")

def read_table(file_path: str) -> pd.DataFrame:
    """
    Reads a file (CSV or Parquet) and returns a DataFrame.

    Args:
        file_path (str): Path to the file.

    Returns:
        pd.DataFrame: DataFrame containing the contents of the file.
    """
    if file_path.endswith('.csv'):
        return read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return read_parquet(file_path)
    else:
        raise ValueError(f"Error: Unsupported file format for {file_path}")

def prepare_anon_list(dir_path: str,
                      secret_column: Optional[str] = None,
                      known_columns: Optional[list[str]] = None,
                      force_feather_save = False,
                      prevent_feather_save = False) -> list[pd.DataFrame]:
    """
    Reads in the CSV and Parquet files from dir_path and returns them as a list of dataframes.

    If secret_column or known_columns are provided, then only return dataframes that contain the
    secret column and/or at least one of the known columns.

    Checks to make sure there is enough memory to store all of the dataframes in memory, and if not,
    it stores them as feather objects.


    Args:
        dir_path (str): Path to the directory containing the CSV and Parquet files.
        secret_column (Optional[str]): Name of the secret column to filter by.
        known_columns (Optional[list[str]]): List of known columns to filter by.
    Returns:
        list[pd.DataFrame]: List of DataFrames containing the contents of the files.
    """

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        # throw an exception
        raise ValueError(f"Error: {dir_path} does not exist or is not a directory")
    anon = []
    for file in os.listdir(dir_path):
        if file.endswith('.csv') or file.endswith('.parquet'):
            file_path = os.path.join(dir_path, file)
            df = read_table(file_path)
            # Check if df_candidates has the secret column and at least one known column
            if secret_column is not None and secret_column not in df.columns:
                print(f"Skipping {file_path} because it does not contain the secret column '{secret_column}'")
                del df
                continue
            if known_columns is not None and not any(col in df.columns for col in known_columns):
                print(f"Skipping {file_path} because it does not contain any of the known columns {known_columns}")
                del df
                continue
            anon.append(df)
    return anon