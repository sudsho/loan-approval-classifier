import numpy as np
import pandas as pd

from src.preprocess import (
    basic_clean,
    impute_missing,
    encode_categoricals,
    encode_target,
)


def _sample():
    return pd.DataFrame({
        'Loan_ID': ['LP001', 'LP002', 'LP003'],
        'Gender': ['Male', None, 'Female'],
        'Married': ['Yes', 'No', None],
        'Dependents': ['0', '1', '2'],
        'Education': ['Graduate', 'Not Graduate', 'Graduate'],
        'Self_Employed': ['No', 'Yes', None],
        'ApplicantIncome': [5000, 4000, np.nan],
        'CoapplicantIncome': [0, 1500, 2000],
        'LoanAmount': [120, np.nan, 100],
        'Loan_Amount_Term': [360, 360, 180],
        'Credit_History': [1, 0, np.nan],
        'Property_Area': ['Urban', 'Rural', 'Semiurban'],
        'Loan_Status': ['Y', 'N', 'Y'],
    })


def test_basic_clean_drops_loan_id():
    df = basic_clean(_sample())
    assert 'Loan_ID' not in df.columns


def test_impute_missing_no_nans_in_known_cols():
    df = impute_missing(basic_clean(_sample()))
    for c in ['Gender', 'Married', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History']:
        assert df[c].isnull().sum() == 0


def test_encode_categoricals_returns_ints():
    df = impute_missing(basic_clean(_sample()))
    enc, encoders = encode_categoricals(df)
    for c in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        assert enc[c].dtype.kind in 'iu'
    assert set(encoders.keys()) >= {'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'}


def test_encode_target():
    y = pd.Series(['Y', 'N', 'Y', 'N'])
    enc = encode_target(y)
    assert enc.tolist() == [1, 0, 1, 0]
