# loan-approval-classifier

[![Build Status](https://travis-ci.org/sudsho/loan-approval-classifier.svg?branch=master)](https://travis-ci.org/sudsho/loan-approval-classifier)

Binary classifier for the Loan Prediction problem from Analytics Vidhya. The
goal is to predict whether a loan application will be approved (`Y`) or
rejected (`N`) given applicant attributes such as income, credit history,
education, and property area.

## Dataset

- Source: Analytics Vidhya "Loan Prediction Practice Problem" (also mirrored
  on Kaggle).
- 614 rows, 12 features, target column `Loan_Status`.
- Features: Gender, Married, Dependents, Education, Self_Employed,
  ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
  Credit_History, Property_Area.

The CSV is committed under `data/train.csv` for convenience.

## EDA highlights

See `notebooks/eda.ipynb`. Key takeaways:

- Target is imbalanced, ~69% `Y` vs ~31% `N`. A naive always-`Y` baseline
  already scores ~69% accuracy, so the headline number to chase is F1 or
  balanced accuracy, not raw accuracy.
- Several columns have missing values. The worst offenders are
  `Credit_History`, `Self_Employed`, and `LoanAmount`.
- `ApplicantIncome` and `CoapplicantIncome` are heavily right-skewed, so a
  log transform helps any linear model.
- `Credit_History` is by far the strongest single predictor of approval.

## Approach

1. EDA in `notebooks/eda.ipynb`.
2. Preprocess: median impute numeric, mode impute categorical, label encode.
3. Try `LogisticRegression`, `RandomForestClassifier`,
   `GradientBoostingClassifier` from scikit-learn.
4. Train and serialize the chosen model with `joblib`.
5. Serve predictions through a small Flask web form.

## Results

5-fold cross-validation accuracy on the training data:

| model              | accuracy |
|--------------------|----------|
| LogisticRegression | ~0.80    |
| RandomForest       | ~0.79    |
| GradientBoosting   | ~0.78    |

Held-out test classification report (RandomForest, default config):

```
              precision    recall  f1-score   support
           N       0.92      0.46      0.61        37
           Y       0.79      0.98      0.87        86
    accuracy                           0.82       123
```

The classifier is conservative on rejections because the dataset is skewed
toward approvals.

## Quick start (runs offline)

No network, no downloads. The training CSV is committed under `data/train.csv`,
and the smoke falls back to a synthesized loan-approval-schema frame if that
file is ever missing, so it always has data to work with.

```bash
python scripts/smoke.py    # or: make smoke
```

Real output:

```
=== loan-approval-classifier offline smoke ===
data source : bundled data/train.csv
rows / cols : 333 / 12
test accuracy : 0.7463
test roc auc  : 0.8010
saved model   : artifacts\model.pkl
predict_one   : label=1 (Approved), proba=0.8564
flask /predict : status=200, verdict=Approved
SMOKE OK
```

The smoke trains the classifier (prints accuracy and ROC AUC), saves the model
and encoders under `artifacts/`, then exercises the serving path two ways: a
direct call to `src.predict.predict_one` and a Flask test-client `POST /predict`,
asserting a valid label and probability from each.

Tests:

```bash
python -m pytest -q
# 16 passed
```

## Full training and web UI (optional)

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python app.py
```

Then open `http://localhost:5000` and fill out the form. The bundled
`data/train.csv` is a 333-row sample of the Analytics Vidhya set; swap in the
full 614-row CSV for the headline numbers quoted below.

## Deploy on Heroku

```bash
heroku create my-loan-app
git push heroku master
```

The included `Procfile` runs the app under `gunicorn`, and `runtime.txt`
pins Python 3.7.

## Project layout

```
.
├── app.py                  # Flask web app
├── configs/default.yaml
├── data/train.csv
├── notebooks/eda.ipynb
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── static/style.css
├── templates/
│   ├── form.html
│   └── result.html
├── tests/
├── Procfile
├── runtime.txt
└── .travis.yml
```

## Tests

```bash
pytest -q
```

CI is set up through Travis (see `.travis.yml`).

## License

MIT
