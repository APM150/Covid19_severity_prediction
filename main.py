from utils import *
import load_data
from Project_Models import RNN_Model
import torch
import torch.nn as nn

def normalize(x):
    maxtensor = x.max(0, keepdim=True)[0]
    maxtensor[maxtensor==0] = 1e-4
    x_normed = x / maxtensor
    return x_normed, maxtensor

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_data.load_county_level('./data/covid19-severity-prediction/data')

    x = form_input_tensor(df, ['#Hospitals', '#ICU_beds', 'MedicareEnrollment,AgedTot2017', 'DiabetesPercentage'], maxload=100).to(device)
    x, xmaxtensor = normalize(x)
    print(f"#x nan: {(torch.sum(torch.isnan(x)))}")
    print("x:", x)

    y = form_labels_tensor(df).to(device)
    y, ymaxtensor = normalize(y)
    print(f"#y nan: {torch.sum(torch.isnan(y))}")
    print("y:", y * ymaxtensor)

    model = RNN_Model.RNN(x.shape[2], 128, 2).to(device)
    with torch.no_grad():
        print("Before training:", model(x) * ymaxtensor)

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    learning_rate = 1e-2
    num_epoches = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        outputs = model(x)

        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f'Epoch [{epoch + 1}/{num_epoches}], Loss: {loss.item():.4f}')
    with torch.no_grad():
        print("After training:", (model(x) * ymaxtensor)[10:20])
        print("y:", (y * ymaxtensor)[10:20])