import os
import pickle
import flask
import pandas as pd
import json
from flask import Flask
from flask import request
from flask_restplus import Api, Resource, fields
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

# Project Scoped Imports
from config import UPLOAD_FOLDER
from enhancer import FeatureEnhancer
from functions import allowed_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

api = Api(app = app,
		  version = "1.0",
		  title = "REMML",
		  description = "Real Estate Estaminator using Machine Learning")

name_space = api.namespace('estaminator', description='Main APIs')

prediction_data_model = api.model('Prediction Model',
		  {
              'sqrMeters': fields.Integer(required = True,
					 description="Room size",
					 help="Cannot be blank."),
              'rooms': fields.Integer(required = True,
					 description="Amount of rooms",
					 help="Cannot be blank."),
              'location': fields.String(required = True,
					 description="Location of property",
					 help="Cannot be blank."),
          })
expectImport = api.parser().add_argument('file', type=FileStorage, location='files')

# getting our trained model from a file we created earlier for prediction purposes
model = pickle.load(open("model.pkl", "rb"))

# curl 'http://localhost:5000/estaminator/predict' -d '{"sqrMeters":71, "rooms":2, "location":"Wilda"} ' -XPOST -H "Content-type: application/json"
@name_space.route('/predict', methods=['POST'])
class Prediction(Resource):

    @api.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Internal Error'})
    @api.expect(prediction_data_model)
    def post(self):
        try:
            # grabbing a set of features from the request's body
            # feature_array = request.get_json()['feature_array']
            feature_array = request.get_json()
            data_frame = pd.DataFrame(feature_array, index=[0])
            # print(data_frame)

            # prediction = model.predict([feature_array]).tolist()

            # predicting estimated price - our model rates flat based on the input array
            prediction = model.predict(data_frame).tolist()[0]

            # preparing a response object and storing the model's predictions
            response = {'predictions': prediction}

            # sending our response object back as json
            return flask.jsonify(response)
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")


# curl 'http://localhost:5000/estaminator/import_csv' -F 'data=@/mnt/c/Users/Wojtek/Google Drive/Studia/Semestr 4/Technologie_Sieciowe_Lab_A_Szwabe/REEML/ceny_mieszkan_w_poznaniu.tsv' -XPOST
@name_space.route('/import_csv', methods=['POST'])
class DataImporter(Resource):
    @api.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Internal Error'})
    @api.expect(expectImport)
    def post(self):
        try:
            if 'file' not in request.files:
                raise Exception("Illegal file extension")
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                raise Exception("Illegal filename - filename cannot be empty!")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # return redirect(url_for('uploaded_file', filename=filename))
                # TODO: Retrenowanie modelu - napisac funckje

                return flask.jsonify({'import_result': True, 'imported_file': file.filename})
            else:
                raise Exception("File not allowed")

        except KeyError as e:
            name_space.abort(500, e, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e, status="Could not retrieve information", statusCode="400")


if __name__ == '__main__':
    app.run()
