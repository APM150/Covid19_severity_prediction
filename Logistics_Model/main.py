import os, pickle, sys
import numpy as np
import load_data
import matplotlib.pyplot as plt
from Logistics_Model_Legend import Logistics
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# sys.path.append("../data/covid19-severity-prediction")
# sys.path.append("../data/covid19-severity-prediction/data")
os.chdir("../data")
np.set_printoptions(suppress=True)


def data_loader():
    
    if not "county_data.pkl" in os.listdir():
        
        df = load_data.load_county_level("../data/covid19-severity-prediction/data")
        
        with open("county_data.pkl", 'wb') as f:
            pickle.dump(df, f)
    
    else:
        with open("county_data.pkl", 'rb') as f:
            df = pickle.load(f)
    
    return df

def get_data(df, state, county, data_type):
    return df[(df['StateName'] == state) & (df['CountyName'] == county)][data_type].values


def get_prediction(y):
    
    y_hats = []
    mse = []
    params = []
    x = np.arange(1, y.shape[0] + 1, 1)
    
    for i in x:
        
        learner=Logistics(x[:i],y[:i])
        learner.fit(maxfev=1000000)
        y_hats.append(learner.predict(i + 1) - learner.predict(i))
        mse.append(mean_squared_error(y[:i], y_hats))
        params.append(learner.popt)
    
    return y_hats, mse, params
    

def plot_data(cases, cases_predict, deaths = None, deaths_predict = None):
    
    f, ax = plt.subplots(2, 1, figsize=(16, 8))
    
    x = np.arange(2, cases.shape[0] + 2, 1)
    ax[0].plot(x, cases, c = "g", label = "Case Increasement")
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
    
    
if __name__ == "__main__":

    # df = data_loader()
    #
    # cases = get_data(df, "CA", "Orange", "cases")
    # deaths = get_data(df, "CA", "Orange", "deaths")
    #
    # c_hat, c_mse, c_params = get_prediction(cases - deaths)
    #
    # d_hat, d_mse, d_params = get_prediction(deaths)
    #
    # # print
    # case_increase = np.append([0], [cases[i] - cases[i - 1] for i in range(1, len(cases))])
    # death_increase = np.append([0], [deaths[i] - deaths[i - 1] for i in range(1, len(deaths))])
    #
    # plot_data(case_increase, c_hat, death_increase, d_hat)
    #
    # print("Cases Increase MSE:", mean_squared_error(case_increase, c_hat))
    # print("Deaths Increase MSE:",mean_squared_error(death_increase, d_hat))
    
    
    df = data_loader()

    samples = shuffle(df.head(1610), n_samples= 600, random_state = 1)
    
    print(samples)

    countyList = []
    for state in set(samples['StateName']):
        for county in set(samples['CountyName']):
            countyList.append((state, county))
            
            print(state, county)
    
    case_mse = []
    death_mse = []
    
    for state, county in tqdm(countyList[:60]):
        
        cases = get_data(samples, state, county, "cases")
        deaths = get_data(samples, state, county, "deaths")
        
        if len(cases) and len(deaths):
            cases = np.array(cases[0])
            deaths = np.array(deaths[0])
            
            
            c_hat, c_mse, c_params = get_prediction(cases - deaths)
            d_hat, d_mse, d_params = get_prediction(deaths)

            case_increase = np.append([0], [cases[i] - cases[i - 1] for i in range(1, len(cases))])
            death_increase = np.append([0], [deaths[i] - deaths[i - 1] for i in range(1, len(deaths))])
            
            case_mse.append(mean_squared_error(case_increase, c_hat))
            death_mse.append(mean_squared_error(death_increase, d_hat))
    
    
    
    print(f'Average MSE of Case Increasement: {np.mean(case_mse)}')
    print(f'Average MSE of Death Increasement: {np.mean(death_mse)}')