# REEML
Estimator of the value of construction real estate in Poland using Machine Learning
It's project for studies on Poznan University of Technology and part of Allegro Machine Learning Course. 

## Run
Just run it in PyCharm IDE with app.py as entry point or in console `python app.py` <br />
#### Reuirements
 - Python3
 - All packages from `Pipfile`
 - `UPLOAD_FOLDER` must be in our application folder and `PATH` must lead to `UPLOAD_FOLDER` in order to run application (it will change in future)

## Prediction
Prediction is estimated using `Gradient Boosting Regression` model.

## Importing files
Accepted file extensions: `*.csv`, `*.tsv`

## Technologies
`API` is created using `Flask` <br />
`Machine Learning` using `SkLearn` (based on `Gradient Boosting Regression` algorithm) <br />
`API Specification & Docs` using `Swagger` <br />

## Future features:
- Crawler for collecting data form various websites
- Client Application using Vue & Axios in JavaScript
