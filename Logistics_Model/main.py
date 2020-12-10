import os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from Logistics_Model_Legend import Logistics
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool

# sys.path.append("../data/covid19-severity-prediction")
# sys.path.append("../data/covid19-severity-prediction/data")

os.chdir("../data")
np.set_printoptions(suppress=True)


# print(os.getcwd())


def get_data(df, state, county, data_type):
    return df[(df['StateName'] == state) & (df['CountyName'] == county)][data_type].values


def get_prediction(y, **kwargs):
    y_hats = []
    mse = []
    params = []
    x = np.arange(1, y.shape[0] + 1, 1)
    
    for i in tqdm(x, **kwargs):
        learner = Logistics(x[:i], y[:i])
        learner.fit(maxfev=1000000)
        y_hats.append(learner.predict(i + 1) - y[i - 1])
        # y_hats.append(learner.predict(i + 1) - learner.predict(i))
        # print(y_hats)
        mse.append(mean_squared_error(y[:i], y_hats))
        params.append(learner.popt)
    
    return y_hats, mse, params


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


if __name__ == "__main__":
    
    with open("sample_testing_data.pkl", "rb") as f:
        df = pickle.load(f)
    
    # df = df.head(5)
    # print(df)
    case_mse = []
    death_mse = []
    
    for state, county in tqdm(df[["StateName", "CountyName"]].values, desc="Main Progress", position=0, ncols=100):
        
        cases = get_data(df, state, county, "cases")
        deaths = get_data(df, state, county, "deaths")
        
        if len(cases) and len(deaths):
            cases = np.array(cases[0])
            deaths = np.array(deaths[0])
            
            pool = Pool()
            
            case_result = pool.apply_async(get_prediction, args=(cases - deaths,),
                                           kwds={'desc': "Case Progress", 'position': 1, "leave": False, "ncols": 100})
            death_result = pool.apply_async(get_prediction, args=(deaths,),
                                            kwds={'desc': "Death Progress", 'position': 2, "leave": False,
                                                  "ncols": 100})
            
            pool.close()
            pool.join()
            
            c_hat, c_mse, c_params = case_result.get()
            d_hat, d_mse, d_params = death_result.get()
            
            # c_hat, c_mse, c_params = get_prediction(cases - deaths)
            # d_hat, d_mse, d_params = get_prediction(deaths)
            
            case_increase = np.append([0], [cases[i] - cases[i - 1] for i in range(1, len(cases))])
            death_increase = np.append([0], [deaths[i] - deaths[i - 1] for i in range(1, len(deaths))])
            
            case_mse.append(mean_squared_error(case_increase, c_hat))
            death_mse.append(mean_squared_error(death_increase, d_hat))
    
    print(f'Average MSE of Case Increasement: {np.mean(case_mse)}')
    print(f'Average MSE of Death Increasement: {np.mean(death_mse)}')
