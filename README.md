# loan-approval-classifier

Binary classifier for the Loan Prediction problem from Analytics Vidhya. The
goal is to predict whether a loan application will be approved (`Y`) or
rejected (`N`) based on applicant attributes.

## Dataset

- Source: Analytics Vidhya "Loan Prediction Practice Problem" (also mirrored
  on Kaggle).
- 614 rows, 12 features, target column `Loan_Status`.
- Features: Gender, Married, Dependents, Education, Self_Employed,
  ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
  Credit_History, Property_Area.

## Approach

1. EDA in `notebooks/eda.ipynb`
2. Preprocess: median impute numeric, mode impute categorical, label encode.
3. Try LogisticRegression, RandomForest, GradientBoosting via `sklearn`.
4. Train and serialize with `joblib`.
5. Serve through a small Flask web form.

## Quickstart

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python app.py
```

Then open `http://localhost:5000`.

## Project layout

```
.
├── app.py
├── configs/default.yaml
├── data/train.csv
├── notebooks/eda.ipynb
├── src/
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── static/style.css
├── templates/
│   ├── form.html
│   └── result.html
└── tests/
```

## License

MIT
