import sys
sys.path.append(r'D:\dataset\CS184A-Covid19_severity_prediction\data\covid19-severity-prediction')

import os, load_data, pickle, sys
import numpy as np
from Logistics_Model_Legend import Logistics
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


if __name__ == "__main__":

    print(os.getcwd())

    sys.path.append("../data/covid19-severity-prediction")
    sys.path.append("../data/covid19-severity-prediction/data")

    # os.chdir("../data")

    np.set_printoptions(suppress=True)

    print(os.getcwd())

    df = None

    if not "county_data.pkl" in os.listdir():

        df = load_data.load_county_level(r"D:\dataset\CS184A-Covid19_severity_prediction\data\covid19-severity-prediction\data")

        with open("county_data.pkl", 'wb') as f:
            pickle.dump(df, f)
    else:
        with open("county_data.pkl", 'rb') as f:
            df = pickle.load(f)

    cases = np.array(df[(df['StateName'] == 'CA') & (df['CountyName'] == 'Orange')]['cases'].values[0])

    # print(df[(df['StateName'] == 'CA') & (df['CountyName'] == 'Orange')]['cases'])

    # cases = np.array(
    #     [288, 414, 514, 603, 671, 762, 864, 980, 1062, 1198, 1275, 1371, 1436, 1530, 1585, 1673, 1719, 1798, 1872, 1895,
    #      1936, 1959, 1967, 1969, 1979, 1991])
    #
    #
    # x = np.arange(1, cases.shape[0] + 1, 1)
    #
    # learner = Logistics(x, cases)
    #
    # # print(learner.learn_initial_variables())
    #
    # learner.fit(maxfev=1000000)
    #
    # y_hat = learner.predict(x)
    #
    # print(y_hat)
    #
    # plt.scatter(x, cases, s=0.1)
    #
    # plt.plot(x, y_hat, c='green')
    #
    # plt.show()
    #
    # print(learner.popt)
    #
    # print(learner.mse(x, cases))

    countyList = []
    for state in set(df['StateName']):
        for county in set(df['CountyName']):
            countyList.append((state, county))

    learnerDict = defaultdict(lambda : defaultdict(lambda : None))
    mseDict = defaultdict(lambda : defaultdict(float))

    countyList = tqdm(countyList[:1000])

    for state, county in countyList:

        ctemp = df[(df['StateName'] == state) & (df['CountyName'] == county)]['cases'].values
        # deaths = np.array(df[(df[state] == state) & (df[county] == county)]['deaths'].values[0])

        if len(ctemp):

            yhat = [0]
            x = np.arange(1, cases.shape[0] + 1, 1)
            
            for i in x:
                yi = np.array(ctemp[0])[:i]
                xi = np.arange(1, i + 1, 1)

                learner = Logistics(xi, yi)
                learner.fit(maxfev=100000)

                yhat.append(learner.predict(i + 1))

                # learnerDict[state][county] = learner
                # mseDict[state][county] = learner.mse(x, cases)

            plt.scatter(x, cases, s=0.1)
            plt.plot(x, yhat, c='green')
            plt.show()

    # print(mseDict)
    
    # yhat = [0]
    #         x = np.arange(1, cases.shape[0] + 1, 1)
            
    #         for i in x:
    #             yi = np.array(ctemp[0])[:i]
    #             xi = np.arange(1, i + 1, 1)

    #             learner = Logistics(xi, yi)
    #             learner.fit(maxfev=100000)

    #             yhat.append(learner.predict(i + 1))

    #             # learnerDict[state][county] = learner
    #             # mseDict[state][county] = learner.mse(x, cases)

    #         plt.scatter(x, cases, s=0.1)
    #         plt.plot(x, yhat, c='green')
    #         plt.show()