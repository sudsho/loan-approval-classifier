"""Flask app for loan approval prediction."""
import os
from flask import Flask, render_template, request

from src.predict import load_artifacts, predict_one


MODEL_PATH = os.environ.get('MODEL_PATH', 'artifacts/model.pkl')
PREPROC_PATH = os.environ.get('PREPROC_PATH', 'artifacts/preprocessor.pkl')


app = Flask(__name__)
_model = None
_encoders = None


def _ensure_loaded():
    global _model, _encoders
    if _model is None:
        _model, _encoders = load_artifacts(MODEL_PATH, PREPROC_PATH)


@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    _ensure_loaded()
    record = {k: request.form.get(k) for k in request.form}
    pred, proba = predict_one(record, _model, _encoders)
    label = 'Approved' if pred == 1 else 'Rejected'
    return render_template('result.html',
                           prediction=label,
                           probability=proba,
                           fields=record)


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
