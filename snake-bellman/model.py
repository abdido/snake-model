from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class BellmanTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.n_games = 0
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def save_model(self, filename='bellmanModel.pth'):
        """Save the current model"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon
        }, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, filename='bellmanModel.pth'):
        """Load a saved model"""
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, filename)
        
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Optionally load training progress
            if 'n_games' in checkpoint:
                self.n_games = checkpoint['n_games']
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
                
            print(f"Model loaded from {file_path}")
            print(f"Resumed from episode: {self.n_games}")
            return True
        else:
            print(f"No model found at {file_path}")
            return False

    def save_checkpoint(self, filename='bellmanCheckpoint.pth'):
        """Save training checkpoint including memory"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'memory': list(self.memory)  # Convert deque to list for saving
        }, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, filename='bellmanCheckpoint.pth'):
        """Load training checkpoint including memory"""
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, filename)
        
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.n_games = checkpoint['n_games']
            self.epsilon = checkpoint['epsilon']
            
            # Restore memory if available
            if 'memory' in checkpoint:
                self.memory = deque(checkpoint['memory'], maxlen=MAX_MEMORY)
                
            print(f"Checkpoint loaded from {file_path}")
            print(f"Resumed from episode: {self.n_games}")
            return True
        else:
            print(f"No checkpoint found at {file_path}")
            return False

