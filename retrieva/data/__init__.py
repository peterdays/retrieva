import os

from retrieva import ROOT_PATH

from .filecrawler import get_file_paths

__all__ = [
    "get_file_paths"
]

def add_root(path):
    # util function
    return os.path.join(ROOT_PATH, path)
