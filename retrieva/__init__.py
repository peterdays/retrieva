import logging
import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def configure_logger(log_name: str = ''):
    """
    Configure logger to file and to stdout
    Args:
        folder_suffix (str): Folder suffix to add (e.g. indicator of what the
        logs represent)
    """

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s - %(funcName)s() \n>>>> %(message)s\n")

    # Print info logs to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    # stop propagting to root logger - removes duplicates
    logger.propagate = False

    return logger


LOGGER = configure_logger(__name__)
