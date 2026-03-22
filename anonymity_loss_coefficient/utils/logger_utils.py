import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from uuid import uuid4

def setup_logging(log_file_path:str, stream_level:int=logging.INFO, file_level:int=logging.INFO) -> logging.Logger:
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
    file_handler = RotatingFileHandler(log_file_path, maxBytes=500_000_000, backupCount=1)
    file_handler.setLevel(file_level)  # Set the log level for the FileHandler
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def setup_null_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Creates a logger that safely drops all log records.

    Args:
        name (Optional[str]): Logger name. If None, a unique name is generated.
    """
    if name is None:
        name = f"anonymity_loss_coefficient.null.{uuid4()}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger
