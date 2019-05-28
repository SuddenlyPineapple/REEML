from datetime import datetime
import pickle
import pandas as pd
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
import glob
import config


class RealEstateEstaminator:
    trained_model = []
    le = preprocessing.LabelEncoder()
    score = 0

    def __init__(self):
        self.deploy_model()

    @classmethod
    def deploy_model(cls):

        ### Odczytanie danych z plików
        li = []
        all_files = glob.glob(config.PATH + "/*")

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0, sep='\t')
            li.append(df)

        r = pd.concat(li, axis=0, ignore_index=True)

        print("[",datetime.now().strftime("%d/%b/%y %H:%M:%S") ,"] Rows count for data model training: [", r.__len__(),"]")

        encoded_locations = cls.le.fit(r['location'])
        r['location'] = cls.le.transform(r['location'])

        train1 = r
        labels = r['price']
        train1 = r.drop(['price'], axis=1)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(train1, labels, test_size=0.10, random_state=2)

        clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')
        clf.fit(x_train, y_train)
        cls.score = clf.score(x_test, y_test)

        ### Model zostanie zapisany w pliku pkl
        pickle.dump(clf, open("model.pkl", "wb"))

        ### Mozemy go potem wczytac i wykorzystac do serwowania predykcji
        # model_p = pickle.load(open("model.pkl","rb"))
        # model_p.predict([[sqr_meters, no_of_rooms]])

        ### Przykładowa Predykcja
        # input_df = pd.DataFrame({
        #     'isNew': pd.Series([True]),
        #     'rooms': pd.Series([2]),
        #     'floor': pd.Series([0]),
        #     'location': pd.Series(cls.le.transform(['Centrum'])),
        #     'sqrMeters': pd.Series([20])
        # })
        # print(input_df)
        # print(clf.predict(input_df))

        cls.trained_model = clf
        return clf


model = RealEstateEstaminator.deploy_model()