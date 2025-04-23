import logging
import sys

def setup_logging(log_file_path:str, stream_level:int=logging.INFO, file_level:int=logging.DEBUG) -> logging.Logger:
    """
    Configures logging to send logs to both stdout and a log file with different log levels.

    Args:
        log_file_path (str): Path to the log file.
        stream_level (int): Logging level for the StreamHandler (e.g., logging.INFO).
        file_level (int): Logging level for the FileHandler (e.g., logging.DEBUG).
    """
    # Create a logger
    logger = logging.getLogger("anonymity_loss_coefficient")
    logger.setLevel(logging.DEBUG)  # Set the logger's overall level to the lowest level needed

    # Create a formatter
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    stream_formatter = logging.Formatter(
        "%(message)s"
    )

    # Create a StreamHandler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stream_level)  # Set the log level for the StreamHandler
    stream_handler.setFormatter(stream_formatter)

    # Create a FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(file_level)  # Set the log level for the FileHandler
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger