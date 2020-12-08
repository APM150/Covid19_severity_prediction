import os, sys
sys.path.append(r'D:\dataset\CS184A-Covid19_severity_prediction\data\covid19-severity-prediction')
import load_data, pickle
import numpy as np
from Logistics_Model_Legend import Logistics
import matplotlib.pyplot as plt
from collections import defaultdict

import pandas as pd

from datetime import datetime, timedelta


if __name__ == "__main__":

    sys.path.append("../data/covid19-severity-prediction")
    sys.path.append("../data/covid19-severity-prediction/data")

    # os.chdir("../data")

    np.set_printoptions(suppress=True)

    print(os.getcwd())

    df = None





    if not "county_data.pkl" in os.listdir():

        df = load_data.load_county_level("../data/covid19-severity-prediction/data")

        with open("county_data.pkl", 'wb') as f:
            pickle.dump(df, f)
    else:
        with open("county_data.pkl", 'rb') as f:
            df = pickle.load(f)

    cases = np.array(df[(df['StateName'] == 'CA') & (df['CountyName'] == 'Orange')]['cases'].values[0])

    # cases = np.array(
    #     [288, 414, 514, 603, 671, 762, 864, 980, 1062, 1198, 1275, 1371, 1436, 1530, 1585, 1673, 1719, 1798, 1872, 1895,
    #      1936, 1959, 1967, 1969, 1979, 1991])

    x = np.arange(1, cases.shape[0] + 1, 1)

    learner = Logistics(x, cases)

    # print(learner.learn_initial_variables())

    learner.fit(maxfev=1000000)

    y_hat = learner.predict(x)

    print(y_hat)

    plt.scatter(x, cases, s=0.1)

    plt.plot(x, y_hat, c='green')

    plt.show()

    print(learner.popt)

    print(learner.mse(x, cases))

    # dataDict = defaultdict(lambda : defaultdict(list))
    #
    # for state in set(df['StateName']):
    #     for county in set(df['CountyName']):
    #         cases = np.array(df[(df[state] == 'CA') & (df[county] == 'Orange')]['cases'].values[0])
    #         deaths = np.array(df[(df[state] == 'CA') & (df[county] == 'Orange')]['deaths'].values[0])
    #
    #         x = np.arange(1, cases.shape[0] + 1, 1)
    #
    #         learner = Logistics(x, cases)
    #         learner.fit(maxfev=100000)

    # df = pd.read_csv("dpc-covid19-ita-andamento-nazionale.csv")
    #
    # df = df.loc[:, ['data', 'totale_casi']]
    #
    # FMT = '%Y-%m-%dT%H:%M:%S'
    # date = df['data']
    # df['data'] = date.map(lambda x: (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01T00:00:00", FMT)).days)
    #
    # x = np.array(df.iloc[:, 0])
    # y = np.array(df.iloc[:, 1])
    #
    #
    # learner = Logistics(x, y)
    #
    # learner.fit(maxfev=1000000)
    #
    # learner.popt
    #
    #
    # y_hat = learner.predict(x)
    #
    # print(y_hat)
    #
    # plt.scatter(x, y, s=0.1)
    #
    # plt.plot(x, y_hat, c='green')
    #
    # plt.show()
    #
    # # print(learner.popt)
    #
    # print(learner.mse(x, y))
