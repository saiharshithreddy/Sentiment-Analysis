import sys
from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)
# CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/api/predict')
@cross_origin()
def predict():
    # print(request.is_json)
    content = request.args.get('message')
    print(content)
    # Load the ML model
    cv = pickle.load(open("vector.pkl", "rb"))
    model = open('RCNN_model.pkl', 'rb')
    clf = joblib.load(model)

    if request.method == 'GET':
        # message = request.get_json()['message']
        data = [content]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect).tolist()
        result = str(my_prediction[0])
    return jsonify({"result": result}), 200


if __name__ == '__main__':
    app.run(debug=True)
