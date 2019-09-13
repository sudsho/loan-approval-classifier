"""Data loading + train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import basic_clean, TARGET


def load_dataset(path):
    df = pd.read_csv(path)
    df = basic_clean(df)
    return df


def split(df, test_size=0.2, random_state=42, stratify=True):
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    strat = y if stratify else None
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
