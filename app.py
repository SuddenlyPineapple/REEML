import os
import pickle
import flask
import pandas as pd
import json
from flask import Flask
from flask import request
from flask_apispec import ResourceMeta, Ref, doc, marshal_with, use_kwargs, FlaskApiSpec
from werkzeug.utils import secure_filename

from enhancer import FeatureEnhancer

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'cvs', 'tsv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# function for checking file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# getting our trained model from a file we created earlier for prediction purposes
model = pickle.load(open("model.pkl", "rb"))

# curl 'http://localhost:5000/predict' -d '{"sqrMeters":71, "rooms":2, "location":"Wilda"} ' -XPOST -H "Content-type: application/json"
@app.route('/predict', methods=['POST'])
def predict():
    # grabbing a set of features from the request's body
    # feature_array = request.get_json()['feature_array']
    feature_array = request.get_json()
    data_frame = pd.DataFrame(feature_array, index=[0])
    print(data_frame)

    # prediction = model.predict([feature_array]).tolist()

    # predicting estimated price - our model rates flat based on the input array
    prediction = model.predict(data_frame).tolist()[0]

    # preparing a response object and storing the model's predictions
    response = {'predictions': prediction}

    # sending our response object back as json
    return flask.jsonify(response)

# curl 'http://localhost:5000/import_csv' -F 'data=@/mnt/c/Users/Wojtek/Google Drive/Studia/Semestr 4/Technologie_Sieciowe_Lab_A_Szwabe/REEML/ceny_mieszkan_w_poznaniu.tsv' -XPOST
@app.route('/import_csv', methods=['POST'])
def import_csv():
    response = {'import_result': False}

    if 'data' not in request.files:
        return flask.jsonify(response)

    file = request.files['data']

    if file.filename == '':
        return flask.jsonify(response)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        response = {'import_result': True}

    #TODO: Retrenowanie modelu - napisac funckje

    return flask.jsonify(response)

if __name__ == '__main__':
    app.run()

# docs = FlaskApiSpec(app)
# docs.register()
