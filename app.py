import pickle
import flask
import pandas as pd
import json
from flask import Flask
from flask import request
from flask_apispec import ResourceMeta, Ref, doc, marshal_with, use_kwargs, FlaskApiSpec
from enhancer import FeatureEnhancer

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


# getting our trained model from a file we created earlier
model = pickle.load(open("model.pkl", "rb"))


@app.route('/predict', methods=['POST'])
def predict():
    # grabbing a set of features from the request's body
    # feature_array = request.get_json()['feature_array']
    feature_array = request.get_json()

    dataFrame = pd.DataFrame(feature_array, index=[0])

    print(dataFrame)
    # our model rates flat based on the input array
    # prediction = model.predict([feature_array]).tolist()
    prediction = model.predict(dataFrame).tolist()[0]

    # preparing a response object and storing the model's predictions
    response = {}
    response['predictions'] = prediction

    # sending our response object back as json
    return flask.jsonify(response)

if __name__ == '__main__':
    app.run()

# docs = FlaskApiSpec(app)
# docs.register()
