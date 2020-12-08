import torch

def form_input_tensor(df, features: list, maxload=float('+inf')):
    """
    Given data frame, and wanted features, form the N x S x M tensor
    :param df: panda data frame
    :param features: list of features wanted
    :return: N x S x M tensor, where N is number of counties, S is sequence number, M is number of features
    """
    result = []
    maxload = min(df.shape[0], maxload)
    for cityInfo in df[['countyFIPS'] + features].head(maxload).values:
        allCityInfo = []  # form the sequence
        for dayICases, dayIDeath in zip(df[(df['countyFIPS'] == cityInfo[0])]['cases'].values[0][:-1],
                                        df[(df['countyFIPS'] == cityInfo[0])]['deaths'].values[0][:-1]):
            oneCityOneDay = cityInfo[1:].tolist()
            oneCityOneDay.append(dayICases)
            oneCityOneDay.append(dayIDeath)
            allCityInfo.append(oneCityOneDay)
        result.append(allCityInfo)
    return torch.tensor(result, dtype=torch.float32)


def form_labels_tensor(df, maxload=float('+inf')):
    y = []
    maxload = min(df.shape[0], maxload)
    for totalCases, totalDeaths in zip(df['tot_cases'].head(maxload).values, df['tot_deaths'].head(maxload).values):
        y.append([totalCases, totalDeaths])
    return torch.tensor(y, dtype=torch.float32)

# Download data, execute this block once in a while
# checks if dataset is downloaded at data/covid19-severity-prediction
# updating to the latest version
# Note: this can take a while if you don't have the dataset in data/covid19-severity-prediction

import os
import sys
if 'data' not in os.listdir():
    os.mkdir('data')
os.chdir('data')
if 'covid19-severity-prediction' not in os.listdir():
    print("covid dataset not detected, downloading data to ./data/covid19-severity-prediction")
    os.system('git clone https://github.com/Yu-Group/covid19-severity-prediction.git')
os.chdir('covid19-severity-prediction')
print("updating to the latest version")
os.system('git pull')
print("update successful")
os.chdir('..')
os.chdir('..')
print("current working dir:", os.getcwd())
sys.path.append("./data/covid19-severity-prediction")