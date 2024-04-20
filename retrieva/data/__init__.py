import os

from retrieva import ROOT_PATH

from .filecrawler import get_file_paths

__all__ = [
    "get_file_paths"
]


DATA_PATH = os.path.join(ROOT_PATH, "artifacts/sagemaker_documentation_small")
