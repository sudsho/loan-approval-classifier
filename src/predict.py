"""Inference helpers."""
import joblib
import pandas as pd

from src.preprocess import (
    impute_missing,
    encode_categoricals,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)


FEATURE_ORDER = CATEGORICAL_COLS + NUMERIC_COLS


def load_artifacts(model_path, preproc_path):
    model = joblib.load(model_path)
    encoders = joblib.load(preproc_path)
    return model, encoders


def to_dataframe(record):
    """record is a dict from form input."""
    row = {}
    for c in FEATURE_ORDER:
        row[c] = record.get(c)
    df = pd.DataFrame([row])
    # numeric coercion - form inputs come as strings
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def predict_one(record, model, encoders):
    df = to_dataframe(record)
    df = impute_missing(df)
    df, _ = encode_categoricals(df, encoders=encoders)
    pred = model.predict(df)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(df)[0, 1])
    return int(pred), proba
