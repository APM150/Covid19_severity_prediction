import os, pickle, sys
import numpy as np
from logistics_utils import get_data, get_prediction
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool

os.chdir("../data")
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    BATCH_SIZE = 30
    
    with open("sample_testing_data.pkl", "rb") as f:
        df = pickle.load(f)
    
    case_predict_mse = []
    death_predict_mse = []
    
    case_increase_mse = []
    death_increase_mse = []
    
    for state, county in tqdm(df[["StateName", "CountyName"]].values, desc="Main Progress", position=0, ncols=100):
        cases = get_data(df, state, county, "cases")
        deaths = get_data(df, state, county, "deaths")
        
        if len(cases) and len(deaths):
            cases = np.array(cases[0])
            deaths = np.array(deaths[0])
            cases -= deaths  # get the actual cases value
            
            pool = Pool()
            
            case_result = pool.apply_async(get_prediction, args=(cases, BATCH_SIZE),
                                           kwds={'desc': "Case Progress", 'position': 1, "leave": False, "ncols": 100})
            death_result = pool.apply_async(get_prediction, args=(deaths, BATCH_SIZE),
                                            kwds={'desc': "Death Progress", 'position': 2, "leave": False,
                                                  "ncols": 100})
            
            pool.close()
            pool.join()
            
            c_predict, c_increase, c_params = case_result.get()
            d_predict, d_increase, d_params = death_result.get()
            
            case_predict_mse.append(mean_squared_error(cases[1:], c_predict[:-1]))
            death_predict_mse.append(mean_squared_error(deaths[1:], d_predict[:-1]))
            
            case_increase = np.array([cases[i] - cases[i - 1] for i in range(1, len(cases))])
            death_increase = np.array([deaths[i] - deaths[i - 1] for i in range(1, len(deaths))])
            
            case_increase_mse.append(mean_squared_error(case_increase, c_increase[:-1]))
            death_increase_mse.append(mean_squared_error(death_increase, d_increase[:-1]))
    
    print(f'Average MSE of Total Cases Prediction: {np.mean(case_predict_mse)}')
    print(f'Average MSE of Total Deaths Prediction: {np.mean(death_predict_mse)}')
    
    print(f'Average MSE of Case Increasement Prediction: {np.mean(case_increase_mse)}')
    print(f'Average MSE of Death Increasement Prediction: {np.mean(death_increase_mse)}')
