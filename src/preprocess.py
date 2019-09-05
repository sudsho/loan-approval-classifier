"""Preprocessing for loan approval data."""
import pandas as pd


NUMERIC_COLS = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
]

CATEGORICAL_COLS = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
]


def load_csv(path):
    return pd.read_csv(path)


def basic_clean(df):
    # drop loan id, it's just an identifier
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    return df
