"""Preprocessing for loan approval data."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

TARGET = 'Loan_Status'

# Canonical feature column order used at both train and predict time. The model
# is fit on columns in this exact order, so serving must build its frame the
# same way (sklearn checks feature name order at predict time).
FEATURE_ORDER = CATEGORICAL_COLS + NUMERIC_COLS


def load_csv(path):
    return pd.read_csv(path)


def basic_clean(df):
    # drop loan id, it's just an identifier
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    # Dependents has '3+' which we keep as a string-category, but normalise
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].astype(str).str.strip()
    return df


def impute_missing(df):
    df = df.copy()
    # Credit_History is numeric in the csv but really a 0/1 flag, treat as categorical for fill
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        if c == 'Credit_History':
            df[c] = df[c].fillna(df[c].mode()[0])
        else:
            df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])
    return df


def encode_categoricals(df, encoders=None):
    df = df.copy()
    fitted = {} if encoders is None else dict(encoders)
    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            continue
        if encoders and c in encoders:
            le = encoders[c]
            # fall back to most frequent for unseen labels
            most_common = le.classes_[0]
            df[c] = df[c].astype(str).apply(
                lambda x: x if x in set(le.classes_) else most_common
            )
            df[c] = le.transform(df[c])
        else:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            fitted[c] = le
    return df, fitted


def encode_target(y):
    # Y -> 1, N -> 0
    return (y == 'Y').astype(int)
