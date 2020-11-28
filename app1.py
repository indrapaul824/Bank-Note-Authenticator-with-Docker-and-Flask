from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict')
def predict_note_auth():
    """
    Bank Note Authenticator using RandomForest Classification
    ---
    Parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true

    Returns:
        The predicted outcome:
        0 - Note is not authentic
        1 - Note is authentic
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is" + str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_note_auth_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted value for the csv is" + str(list(prediction))


if __name__ == '__main__':
    app.run()