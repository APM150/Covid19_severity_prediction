import os, pickle, sys
import numpy as np
import load_data
from sklearn.utils import shuffle


def data_loader():
    if not "county_data.pkl" in os.listdir():
        
        df = load_data.load_county_level("covid19-severity-prediction/data")
        
        with open("county_data.pkl", 'wb') as f:
            pickle.dump(df, f)
    
    else:
        with open("county_data.pkl", 'rb') as f:
            df = pickle.load(f)
    
    return df


def data_filter(df):
    FEATURES = [
        '#Hospitals',
        '#ICU_beds',
        'MedicareEnrollment,AgedTot2017',
        'DiabetesPercentage',
        'HeartDiseaseMortality',
        'StrokeMortality',
        'Smokers_Percentage',
        'RespMortalityRate2014',
        '#FTEHospitalTotal2017',
        "TotalM.D.'s,TotNon-FedandFed2017",
        '#HospParticipatinginNetwork2017', 'cases', 'deaths']
    
    return df.drop(df[df[FEATURES].isnull().any(axis=1)].index)

def data_sampler(df):
    df = shuffle(df, random_state=20)
    
    sample_test = df.iloc[0:100]
    sample_train = df.iloc[100:]
    
    with open("sample_testing_data.pkl", 'wb') as f:
        pickle.dump(sample_test, f)

    with open("sample_training_data.pkl", 'wb') as f:
        pickle.dump(sample_train, f)


if __name__ == "__main__":
    os.chdir("../data")
    np.set_printoptions(suppress=True)
    
    # print(os.getcwd())
    
    df = data_loader()
    
    df = data_filter(df)

    data_sampler(df)

