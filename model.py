import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):

    def __init__(self, input_size,  hidden_layer_1_size, hidden_layer_2_size, hidden_layer_3_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_layer_1_size)
        self.linear2 = nn.Linear(hidden_layer_1_size, hidden_layer_2_size)
        self.linear3 = nn.Linear(hidden_layer_2_size, hidden_layer_3_size)
        self.linear4 = (nn.Linear(hidden_layer_3_size, output_size))


    def forward(self, input):
        input = F.relu(self.linear1(input))
        input = F.relu(self.linear2(input))
        input = F.relu(self.linear3(input))
        input = self.linear4(input)

        return input

    def save(self, file_name = "model.pth"):
        model_folder = './model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        file_name = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder = './model'
        file_name = os.path.join(model_folder, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            print('Model loaded from', file_name)
        else:
            print('No saved model found, starting fresh')

class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma

        # optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: Predicted Q-value of Current state
        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):

            # 2 : Predicted Q-value of Next state
            #   ==> Q_new = Reward + Y (Gamma) * max(Q_next_state)

            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        pass
