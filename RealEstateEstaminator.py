from datetime import datetime
import pickle
import pandas as pd
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import math
from sklearn.pipeline import Pipeline
from enhancer import FeatureEnhancer
import glob
import config


class RealEstateEstaminator:
    files = {'ceny_mieszkan_w_poznaniu.tsv'}
    trained_model = []

    def __init__(self):
        self.deploy_model()

    @classmethod
    def deploy_model(cls):

        # Odczytanie danych z plików
        li = []
        all_files = glob.glob(config.PATH + "/*")

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0, sep='\t')
            li.append(df)

        r = pd.concat(li, axis=0, ignore_index=True)

        print(r)

        print("[",datetime.now().strftime("%d/%b/%y %H:%M:%S") ,"] Rows count for data model training: [", r.__len__(),"]")

        # Podzial danych na dane testowe i dane do trenowania
        r_train, r_test = sklearn.model_selection.train_test_split(r, test_size=0.2)
        # Wybranie modelu trenujcego
        model = sklearn.linear_model.LinearRegression()

        features = FeatureEnhancer()
        p = Pipeline([
            ('feature selection', features),
            ('regression', model)
        ])

        # print(p)

        # features = ['sqrMeters', 'rooms']
        label = ['price']

        X_train = r_train  # [features]
        y_train = r_train[label].values.reshape(-1, 1)
        # print(y_train)
        p.fit(X_train, y_train)

        # sqr_meters = 71
        # no_of_rooms = 2
        # model.predict([[sqr_meters, no_of_rooms]])

        # model zostanie zapisany w pliku pkl
        pickle.dump(p, open("model.pkl", "wb"))

        # mozemy go potem wczytac i wykorzystac do serwowania predykcji
        # model_p = pickle.load(open("model.pkl","rb"))
        # model_p.predict([[sqr_meters, no_of_rooms]])

        # Przykładowa Predykcja
        # input_df = pd.DataFrame({
        #     'sqrMeters': pd.Series([20]),
        #     'rooms': pd.Series([4]),
        #     'isNew': pd.Series([False]),
        #     'location': pd.Series(['Wilda'])
        # })
        #print(input_df)
        #print(p.predict(input_df))

        cls.trained_model = p
        return p



#model = RealEstateEstaminator.deploy_model()