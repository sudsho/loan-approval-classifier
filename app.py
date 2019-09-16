"""Flask app for loan approval prediction."""
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # TODO: load model + score
    fields = {k: request.form.get(k) for k in request.form}
    return render_template('result.html', prediction='?', fields=fields)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
