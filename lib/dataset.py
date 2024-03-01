import pandas as pd

from .constants import PROJECT_DIR

__all__ = ['load_train_data', 'load_test_data']


def load_train_data() -> pd.DataFrame:
    file_path = PROJECT_DIR / 'data' / 'train.tsv'
    data = pd.read_table(file_path)
    return data


def load_test_data() -> pd.DataFrame:
    file_path = PROJECT_DIR / 'data' / 'test.tsv'
    data = pd.read_table(file_path)
    return data
