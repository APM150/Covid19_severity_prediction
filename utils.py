import torch

def form_input_tensor(df, features: list):
    """
    Given data frame, and wanted features, form the N x S x M tensor
    :param df: panda data frame
    :param features: list of features wanted
    :return: N x S x M tensor, where N is number of counties, S is sequence number, M is number of features
    """
    result = []
    for cityInfo in df[['countyFIPS'] + features].head(10).values:
        allCityInfo = []  # form the sequence
        for dayICases, dayIDeath in zip(df[(df['countyFIPS'] == cityInfo[0])]['cases'].values[0][:-1],
                                        df[(df['countyFIPS'] == cityInfo[0])]['deaths'].values[0][:-1]):
            oneCityOneDay = cityInfo[1:].tolist()
            oneCityOneDay.append(dayICases)
            oneCityOneDay.append(dayIDeath)
            allCityInfo.append(oneCityOneDay)
        result.append(allCityInfo)
    return torch.tensor(result, dtype=torch.float32)


def form_labels_tensor(df):
    y = []
    for totalCases, totalDeaths in zip(df['tot_cases'].head(10).values, df['tot_deaths'].head(10).values):
        y.append([totalCases, totalDeaths])
    return torch.tensor(y, dtype=torch.float32)

