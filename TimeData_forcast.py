import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import tqdm
import wandb
import os

torchlmanual_seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seq_length = 7
data_dim = 5 # Feature 개수
hidden_dim  =10
output_dim = 7 # 출력차원
learning_rate = 0.01
epochs = 500
batch_size = 100

def build_dataset(data, seq_len, pred_steps):
    dataX = []
    dataY = []
    for i in range(len(data)-seq_len - pred_steps + 1):
        x = data[i:i+seq_len, :]
        y = data[i+seq_len:i+seq_len+pred_steps, [-1]].flatten()
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)
    
df = pd.read_csv("archive\ETH_1H.csv")

# Data Preprocessing
df = df[::-1]
df = df[['Open', 'High', 'Low', 'Volume', 'Close']] # 필요한 Feature만 추출함

# Train/Test set 분리
train_size = int(len(df)*0.7)
train_set = df[0:train_size]
test_set = df[train_size-seq_length:]


# Scaling 수행
scaler_x = MinMaxScaler()
scaler_x.fit(train_set.iloc[:,:-1])

train_set.iloc[:,:-1] = scaler_x.transform(train_set.iloc[:,:-1])
test_set.iloc[:,:-1] = scaler_x.transform(test_set.iloc[:,:-1])

scaler_y = MinMaxScaler()
scaler_y.fit(train_set.iloc[:,[-1]])

train_set.iloc[:, -1] = scaler_y.transform(train_set.iloc[:,[-1]])
test_set.iloc[:,-1] = scaler_y.transform(test_set.iloc[:,[-1]])

trainX, trainY = build_dataset(np.array(train_set), seq_length, output_dim)
testX, testY = build_dataset(np.array(test_set), seq_length, output_dim)

trainX_tensor = torch.FloatTensor(trainX).to(device)
trainY_tensor = torch.FloatTensor(trainY).to(device)

testX_tensor = torch.FloatTensor(testX).to(device)
testY_tensor = torch.FloatTensor(testY).to(device)

dataset = TensorDataset(trainX_tensor, trainY_tensor)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim)
        )

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:,-1])
        return x
    
LSTM = LSTM(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)

wandb.init(project="Practice - ETH_price_prediction", config={
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "seq_length": seq_length,
    "hidden_dim": hidden_dim,
    "data_dim": data_dim
})

def train_model(model, train_df, epochs=None, lr=None, verbose=10, patience=10):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    save_dir = "./checkpoint"
    os.makedirs(save_dir, exist_ok=True)

    train_hist = np.zeros(epochs)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = len(train_df)

        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples
            model.reset_hidden_state()

            outputs = model(x_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss/total_batch
        train_hist[epoch] = avg_cost

        if epoch % verbose == 0:
            print('Epoch: ', '%04d' % (epoch), 'train loss : ', '{:.4f}'.format(avg_cost))
        wandb.log({"train_loss": avg_cost, "epoch": epoch})

        if (epoch+1) % 100 == 0:
            save_path = os.path.join(save_dir, f"model_path_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} to {save_path}")
            # wandb.save(save_path)
        
        # if(epoch%patience==0) & (epoch!=0):
        #     if train_hist[epoch-patience] < train_hist[epoch]:
        #         print('\n Early Stopping')
        #         break
        
    return model.eval(), train_hist

model, train_hist = train_model(LSTM, dataloader, epochs=epochs, lr=learning_rate, verbose=5, patience=20)


# Evaluation
with torch.no_grad():
    pred = []
    for pr in range(len(testX_tensor)):
        model.reset_hidden_state()

        predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
        # predicted = torch.flatten(predicted).item()
        pred.append(predicted.cpu().numpy().flatten())

    # Inverse : Scaling 했던 값들을 원래대로 돌려놓는 과정
    # pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
    pred_inverse = scaler_y.inverse_transform(np.array(pred))
    testY_inverse = scaler_y.inverse_transform(testY_tensor.cpu().numpy())

def MAE(true, pred):
    return np.mean(np.abs(true-pred))

mae_score = MAE(pred_inverse, testY_inverse)
print('MAE SCORE : ', mae_score)
wandb.log({"MAE": mae_score})

# target test
length = len(test_set)
target = np.array(test_set)[length-seq_length:]

target = torch.FloatTensor(target).to(device)
target = target.reshape([1, seq_length, data_dim]).to(device)

# out = model(target)
# pre = torch.flatten(out).item()

# pre = round(pre, 8)
# pre_inverse = scaler_y.inverse_transform(np.array(pre).reshape(-1,1))
# print(pre_inverse.reshape([1])[0])

out = model(target)
pre = out.detach().cpu().numpy().flatten()  # (7,)
pre_inverse = scaler_y.inverse_transform(pre.reshape(-1,1))
print(pre_inverse.reshape(-1))  # 7개 값 출력

plt.figure(figsize=(10,6))
plt.plot(pred_inverse[:,0], label='pred')
plt.plot(testY_inverse[:,0], label='true')
plt.legend()
plt.title('Loss Plot')
plt.show()