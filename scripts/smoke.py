"""Offline smoke test for the loan approval classifier.

Runs the whole thing end to end with no network and no external downloads:

  1. load data (bundled data/train.csv if present, else synthesize a
     loan-approval-schema frame with a learnable approval label)
  2. build features, train the classifier, print accuracy AND roc auc
  3. save artifacts (model + label encoders)
  4. exercise the predict path two ways:
       a) direct call to src.predict.predict_one on a sample applicant
       b) Flask test client POST to /predict
     asserting a valid label and probability in both cases

Run it with:  python scripts/smoke.py   (or: make smoke)
"""
import os
import sys

# make the repo root importable when run as `python scripts/smoke.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.preprocess import (
    basic_clean,
    impute_missing,
    encode_categoricals,
    encode_target,
    TARGET,
    FEATURE_ORDER,
)
from src.model import get_model


BUNDLED_CSV = os.path.join(ROOT, 'data', 'train.csv')
MODEL_PATH = os.path.join(ROOT, 'artifacts', 'model.pkl')
PREPROC_PATH = os.path.join(ROOT, 'artifacts', 'preprocessor.pkl')


def synthesize(n=600, seed=42):
    """Build a loan-approval-schema frame with a learnable Loan_Status.

    Used only if the bundled CSV is missing, so the smoke stays offline no
    matter what. Approval leans on credit history, income vs loan size, and
    education, with some label noise so the problem is not trivially separable.
    """
    rng = np.random.default_rng(seed)
    applicant_income = rng.gamma(shape=2.0, scale=2600, size=n).round().astype(int)
    coapplicant_income = (rng.gamma(shape=1.2, scale=1200, size=n)
                          * rng.integers(0, 2, size=n)).round().astype(int)
    loan_amount = (rng.gamma(shape=2.5, scale=55, size=n) + 20).round().astype(int)
    loan_term = rng.choice([120, 180, 240, 360, 360, 360], size=n)
    credit_history = rng.choice([1.0, 0.0], size=n, p=[0.84, 0.16])
    education = rng.choice(['Graduate', 'Not Graduate'], size=n, p=[0.78, 0.22])

    total_income = applicant_income + coapplicant_income
    # affordability: income relative to monthly-ish loan burden
    burden = loan_amount / np.maximum(total_income / 1000.0, 1.0)
    score = (
        2.4 * credit_history
        + 0.45 * (education == 'Graduate')
        + 0.0009 * total_income
        - 0.18 * burden
        - 1.2
    )
    prob = 1.0 / (1.0 + np.exp(-score))
    approved = (rng.random(n) < prob)

    df = pd.DataFrame({
        'Loan_ID': ['LP%06d' % i for i in range(n)],
        'Gender': rng.choice(['Male', 'Female'], size=n, p=[0.8, 0.2]),
        'Married': rng.choice(['Yes', 'No'], size=n, p=[0.65, 0.35]),
        'Dependents': rng.choice(['0', '1', '2', '3+'], size=n, p=[0.58, 0.17, 0.16, 0.09]),
        'Education': education,
        'Self_Employed': rng.choice(['No', 'Yes'], size=n, p=[0.86, 0.14]),
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': rng.choice(['Urban', 'Semiurban', 'Rural'], size=n),
        'Loan_Status': np.where(approved, 'Y', 'N'),
    })
    return df


def load_frame():
    if os.path.exists(BUNDLED_CSV):
        return pd.read_csv(BUNDLED_CSV), 'bundled data/train.csv'
    return synthesize(), 'synthetic (bundled CSV not found)'


def main():
    print('=== loan-approval-classifier offline smoke ===')

    df, source = load_frame()
    df = basic_clean(df)
    if 'Credit_History' in df.columns:
        df['Credit_History'] = pd.to_numeric(df['Credit_History'], errors='coerce')
    print('data source : %s' % source)
    print('rows / cols : %d / %d' % (df.shape[0], df.shape[1]))

    # simple stratified holdout without leaning on src.data (keeps smoke self-contained)
    from sklearn.model_selection import train_test_split
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    X_train = impute_missing(X_train)
    X_test = impute_missing(X_test)
    X_train_enc, encoders = encode_categoricals(X_train)
    X_test_enc, _ = encode_categoricals(X_test, encoders=encoders)
    X_train_enc = X_train_enc[FEATURE_ORDER]
    X_test_enc = X_test_enc[FEATURE_ORDER]
    y_train_enc = encode_target(y_train)
    y_test_enc = encode_target(y_test)

    model = get_model('random_forest', {
        'n_estimators': 200, 'max_depth': 8,
        'min_samples_split': 4, 'random_state': 42,
    })
    model.fit(X_train_enc, y_train_enc)

    preds = model.predict(X_test_enc)
    proba = model.predict_proba(X_test_enc)[:, 1]
    acc = accuracy_score(y_test_enc, preds)
    auc = roc_auc_score(y_test_enc, proba)
    print('test accuracy : %.4f' % acc)
    print('test roc auc  : %.4f' % auc)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= auc <= 1.0

    # persist artifacts so the Flask app can load them
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    import joblib
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, PREPROC_PATH)
    print('saved model   : %s' % os.path.relpath(MODEL_PATH, ROOT))

    # a sample applicant with a strong profile (good credit, decent income)
    applicant = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '1',
        'Education': 'Graduate', 'Self_Employed': 'No',
        'ApplicantIncome': '6000', 'CoapplicantIncome': '1800',
        'LoanAmount': '130', 'Loan_Amount_Term': '360',
        'Credit_History': '1', 'Property_Area': 'Urban',
    }

    # (a) direct predict path
    from src.predict import predict_one
    label, prob = predict_one(applicant, model, encoders)
    assert label in (0, 1), 'predict_one returned an invalid label: %r' % label
    assert prob is None or (0.0 <= prob <= 1.0), 'invalid probability: %r' % prob
    print('predict_one   : label=%d (%s), proba=%.4f'
          % (label, 'Approved' if label == 1 else 'Rejected', prob))

    # (b) Flask serve path via the test client (no server, no network)
    os.environ['MODEL_PATH'] = MODEL_PATH
    os.environ['PREPROC_PATH'] = PREPROC_PATH
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        health = client.get('/health')
        assert health.status_code == 200, 'health check failed'
        assert health.get_json()['status'] == 'ok'
        resp = client.post('/predict', data=applicant)
        assert resp.status_code == 200, '/predict returned %d' % resp.status_code
        body = resp.get_data(as_text=True)
        assert ('Approved' in body) or ('Rejected' in body), 'no verdict in response'
    verdict = 'Approved' if b'Approved' in resp.data else 'Rejected'
    print('flask /predict : status=200, verdict=%s' % verdict)

    print('SMOKE OK')


if __name__ == '__main__':
    main()
