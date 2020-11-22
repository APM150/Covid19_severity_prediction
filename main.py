from utils import *
import load_data
from Project_Models import RNN_Model
import torch
import torch.nn as nn

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_data.load_county_level()
    x = form_input_tensor(df, ['#Hospitals', '#ICU_beds']).to(device)
    print("x:", x)
    y = form_labels_tensor(df).to(device)
    print("y:", y)

    model = RNN_Model.RNN(x.shape[2], 128, 2).to(device)
    print("Before training:", model(x))

    criterion = nn.MSELoss()
    learning_rate = 1
    num_epoches = 10000000
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        outputs = model(x)

        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f'Epoch [{epoch + 1}/{num_epoches}], Loss: {loss.item():.4f}')

    print("After training:", model(x))