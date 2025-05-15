import pandas as pd
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