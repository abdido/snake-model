from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class MonteCarloTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.episode_memory = []
        self.n_games = 0
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005 

    def store_experience(self, state, action, reward):
        if isinstance(state, torch.Tensor):
            state = state.detach().numpy()
        elif not isinstance(state, np.ndarray):
            state = np.array(state)
        
        state = state.flatten()
        
        self.episode_memory.append({
            'state': state,
            'action': action,
            'reward': reward
        })

    def calculate_returns(self, rewards):
        """Calculate discounted returns using Monte Carlo method"""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def train_episode(self):
        if len(self.episode_memory) == 0:
            return 0

        states = []
        actions = []
        rewards = []

        for experience in self.episode_memory:
            states.append(experience['state'])
            actions.append(experience['action'])
            rewards.append(experience['reward'])

        returns = self.calculate_returns(rewards)

        try:
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.long)
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
        except ValueError as e:
            print(f"Error creating tensors: {e}")
            print(f"States shapes: {[s.shape if hasattr(s, 'shape') else len(s) for s in states]}")
            self.episode_memory = []
            return 0

        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        if returns_tensor.dim() == 0:
            returns_tensor = returns_tensor.unsqueeze(0)

        pred_q_values = self.model(states)
        target_q_values = pred_q_values.clone().detach()

        for idx in range(len(actions)):
            action_idx = actions[idx].item()
            target_q_values[idx][action_idx] = returns_tensor[idx]

        self.optimizer.zero_grad()
        loss = self.criterion(pred_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.episode_memory = []
        return loss.item()

    def train_step(self, state, action, reward, next_state, done):
        """Store experience and train if episode is done"""
        
        is_batch = isinstance(state, (list, tuple)) or (hasattr(state, 'shape') and len(state.shape) > 1 and state.shape[0] > 1)
        
        if is_batch:
            states = state if isinstance(state, (list, tuple)) else state
            actions = action if isinstance(action, (list, tuple)) else action
            rewards = reward if isinstance(reward, (list, tuple)) else reward
            dones = done if isinstance(done, (list, tuple)) else [done]
            
            for i in range(len(states)):
                if isinstance(states[i], torch.Tensor):
                    state_np = states[i].detach().numpy().flatten()
                else:
                    state_np = np.array(states[i]).flatten()
                
                if isinstance(actions[i], torch.Tensor):
                    if actions[i].dim() > 0 and actions[i].numel() > 1:
                        action_value = torch.argmax(actions[i]).item()
                    else:
                        action_value = actions[i].item()
                else:
                    if hasattr(actions[i], '__len__') and len(actions[i]) > 1:
                        action_value = np.argmax(actions[i])
                    else:
                        action_value = int(actions[i])
                
                if isinstance(rewards[i], torch.Tensor):
                    reward_value = rewards[i].item() if rewards[i].numel() == 1 else rewards[i].mean().item()
                elif isinstance(rewards[i], (tuple, list)):
                    reward_value = float(rewards[i][0]) if len(rewards[i]) > 0 else 0.0
                else:
                    reward_value = float(rewards[i])
                
                self.store_experience(state_np, action_value, reward_value)
                
                if i < len(dones) and dones[i]:
                    return self.train_episode()
            
            return 0
        
        else:
            if isinstance(state, torch.Tensor):
                state_np = state.detach().numpy().flatten()
            else:
                state_np = np.array(state).flatten()

            if isinstance(action, torch.Tensor):
                if action.dim() > 0 and action.numel() > 1:
                    action_value = torch.argmax(action).item()
                else:
                    action_value = action.item()
            else:
                if hasattr(action, '__len__') and len(action) > 1:
                    action_value = np.argmax(action)
                else:
                    action_value = int(action)

            if isinstance(reward, torch.Tensor):
                reward_value = reward.item() if reward.numel() == 1 else reward.mean().item()
            elif isinstance(reward, (tuple, list)):
                reward_value = float(reward[0]) if len(reward) > 0 else 0.0
            else:
                reward_value = float(reward)

            self.store_experience(state_np, action_value, reward_value)

            if done:
                return self.train_episode()

            return 0
    
    def save_model(self, filename='monteCarloModel.pth'):
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

    def load_model(self, filename='monteCarloModel.pth'):
        """Load a saved model"""
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, filename)
        
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
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

    def save_checkpoint(self, filename='monteCarloCheckpoint.pth'):
        """Save training checkpoint including memory"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'memory': list(self.memory) 
        }, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, filename='monteCarloCheckpoint.pth'):
        """Load training checkpoint including memory"""
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, filename)
        
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.n_games = checkpoint['n_games']
            self.epsilon = checkpoint['epsilon']
            
            if 'memory' in checkpoint:
                self.memory = deque(checkpoint['memory'], maxlen=MAX_MEMORY)
                
            print(f"Checkpoint loaded from {file_path}")
            print(f"Resumed from episode: {self.n_games}")
            return True
        else:
            print(f"No checkpoint found at {file_path}")
            return False