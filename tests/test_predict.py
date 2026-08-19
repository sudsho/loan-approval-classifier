"""Round-trip test: a model trained the normal way must be usable by the
serving path. Guards against a train/predict feature-order mismatch."""
import pandas as pd

from src.preprocess import (
    basic_clean,
    impute_missing,
    encode_categoricals,
    encode_target,
    TARGET,
    FEATURE_ORDER,
)
from src.model import get_model
from src.predict import predict_one


def _train_tiny():
    df = pd.DataFrame({
        'Loan_ID': ['LP%03d' % i for i in range(12)],
        'Gender': ['Male', 'Female'] * 6,
        'Married': ['Yes', 'No'] * 6,
        'Dependents': ['0', '1', '2', '3+'] * 3,
        'Education': ['Graduate', 'Not Graduate'] * 6,
        'Self_Employed': ['No', 'Yes'] * 6,
        'ApplicantIncome': [6000, 2000] * 6,
        'CoapplicantIncome': [1500, 0] * 6,
        'LoanAmount': [120, 200] * 6,
        'Loan_Amount_Term': [360] * 12,
        'Credit_History': [1, 0] * 6,
        'Property_Area': ['Urban', 'Rural'] * 6,
        # approval tracks credit history so the tiny model can learn something
        'Loan_Status': ['Y', 'N'] * 6,
    })
    df = basic_clean(df)
    X = impute_missing(df.drop(columns=[TARGET]))
    X_enc, encoders = encode_categoricals(X)
    X_enc = X_enc[FEATURE_ORDER]
    y = encode_target(df[TARGET])
    model = get_model('random_forest', {'n_estimators': 20, 'random_state': 0})
    model.fit(X_enc, y)
    return model, encoders


def test_predict_one_round_trip():
    model, encoders = _train_tiny()
    applicant = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0',
        'Education': 'Graduate', 'Self_Employed': 'No',
        'ApplicantIncome': '6000', 'CoapplicantIncome': '1500',
        'LoanAmount': '120', 'Loan_Amount_Term': '360',
        'Credit_History': '1', 'Property_Area': 'Urban',
    }
    label, proba = predict_one(applicant, model, encoders)
    assert label in (0, 1)
    assert proba is not None
    assert 0.0 <= proba <= 1.0
