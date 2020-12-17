import matplotlib.pyplot as plt
from SI_Logistics_Model import Logistics
from tqdm import tqdm
import numpy as np


def get_data(df, state, county, data_type):
    return df[(df['StateName'] == state) & (df['CountyName'] == county)][data_type].values


def get_prediction(y, batch_size=0, **kwargs):
    y_predict = []
    y_increase = []
    params = []
    x = np.arange(1, y.shape[0] + 1, 1)
    
    for i in tqdm(x, **kwargs):
        start = i - batch_size if i > batch_size else 0
        
        learner = Logistics(x[start:i], y[start:i])
        learner.fit(maxfev=1000000)
        y_hat = learner.predict(i + 1)
        
        y_predict.append(y_hat)
        y_increase.append(y_hat - y[i - 1])
        params.append(learner.popt)
    
    return y_predict, y_increase, params


def plot_data(cases, cases_predict, deaths=None, deaths_predict=None):
    f, ax = plt.subplots(2, 1, figsize=(16, 8))
    
    x = np.arange(2, cases.shape[0] + 2, 1)
    ax[0].plot(x, cases, c="g", label="Case Increasement")
    ax[0].plot(x, cases_predict, c="r", label="Case Increasement Prediction")
    ax[0].set_title("Cases and Prediction Over Time")
    ax[0].legend()
    
    if not (deaths is None or deaths_predict is None):
        x = np.arange(2, deaths.shape[0] + 2, 1)
        ax[1].plot(x, deaths, c="g", label="Death Increasement")
        ax[1].plot(x, deaths_predict, c="r", label="Death Increasement Prediction")
        ax[1].set_title("Deaths and Prediction Over Time")
        ax[1].legend()
    
    plt.show()
