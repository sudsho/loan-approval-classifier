import os
import pandas as pd

from src.data import load_dataset, split


HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '..', 'data', 'train.csv')


def test_load_dataset_drops_loan_id():
    df = load_dataset(DATA)
    assert 'Loan_ID' not in df.columns
    assert 'Loan_Status' in df.columns


def test_split_shapes_match():
    df = load_dataset(DATA)
    X_tr, X_te, y_tr, y_te = split(df, test_size=0.2, random_state=0)
    assert len(X_tr) + len(X_te) == len(df)
    assert len(y_tr) == len(X_tr)
    assert len(y_te) == len(X_te)
