# REEML
Estimator of the value of construction real estate in Poland using Machine Learning
It's project for studies on Poznan University of Technology and part of Allegro Machine Learning Course. 

## Run
Just run it in PyCharm IDE with app.py as entry point or in console `python app.py` <br />
### Requirements/Notes:
 - Python 3.6,
 - Install all packages from `Pipfile` or `requirements.txt`,
 - `UPLOAD_FOLDER` must be in our application folder and `PATH` must lead to `UPLOAD_FOLDER` in order to run application (it will change in future),
 - If running on Docker uncomment last line in `app.py` file (`app.run(host='0.0.0.0')`) and comment previous `app.run()` line. 
## Prediction
Prediction is estimated using `Gradient Boosting Regression` model. <br />
Experiments and tests can be found in `REEML.ipynb`

## Importing files
Accepted file extensions: `*.csv`, `*.tsv`

## Technologies
`API` is created using `Flask` <br />
`Machine Learning` using `SkLearn` (based on `Gradient Boosting Regression` algorithm) <br />
`API Specification & Docs` using `Swagger` <br />

## Future features:
- Crawler for collecting data form various websites
- Client Application using Vue & Axios in JavaScript
