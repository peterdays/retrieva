import os

import pandas as pd


def get_file_paths(files_path: str) -> pd.DataFrame:
    paths = []
    for dirpath, dirnames, filenames in os.walk(files_path):
        aux = [os.path.join(dirpath, filename) for filename in filenames]
        if aux:
            paths.extend(aux)


    files_df = pd.DataFrame({"filepath": paths})
    files_df["filename"] = [
        os.path.basename(f).split(".")[0] for f in files_df["filepath"]
    ]

    return files_df
