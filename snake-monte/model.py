import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from collections import deque

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

class MonteCarloTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Store episode data
        self.episode_memory = []
        
    def store_experience(self, state, action, reward):
        """Store experience for current episode"""
        self.episode_memory.append({
            'state': state,
            'action': action,
            'reward': reward
        })
    
    def calculate_returns(self, rewards):
        """Calculate discounted returns"""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def train_episode(self):
        """Train on complete episode using Monte Carlo"""
        if len(self.episode_memory) == 0:
            return 0
        
        # Extract data from episode
        states = []
        actions = []
        rewards = []
        
        for experience in self.episode_memory:
            states.append(experience['state'])
            actions.append(experience['action'])
            rewards.append(experience['reward'])
        
        # Calculate returns
        returns = self.calculate_returns(rewards)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        returns_tensor = torch.tensor(returns, dtype=torch.float)
        
        # Handle single vs batch
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            returns_tensor = torch.unsqueeze(returns_tensor, 0)
        
        # Get current Q-value predictions
        pred_q_values = self.model(states)
        target_q_values = pred_q_values.clone()
        
        # Update Q-values for taken actions with actual returns
        for idx in range(len(actions)):
            # Find which action was taken (convert one-hot to index)
            if len(actions[idx].shape) > 0:  # One-hot encoded
                action_idx = torch.argmax(actions[idx]).item()
            else:  # Already an index
                action_idx = actions[idx].item()
            
            target_q_values[idx][action_idx] = returns_tensor[idx]
        
        # Training step
        self.optimizer.zero_grad()
        loss = self.criterion(target_q_values, pred_q_values)
        loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        self.episode_memory = []
        
        return loss.item()
    
    def train_step(self, state, action, reward, next_state, done):
        """Modified train_step for compatibility with existing code"""
        # Convert inputs to proper format
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        # Handle single sample
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        
        # For Monte Carlo, we need the full episode
        # This is a simplified version that still works step by step
        # but uses Monte Carlo concept
        
        # Store current transition
        self.store_experience(state.squeeze().numpy(), action.squeeze().numpy(), reward.item())
        
        # If episode is done, train on the full episode
        if done[0]:
            return self.train_episode()
        
        return 0  # No training until episode is done

class HybridTrainer:
    """Hybrid approach: Short memory (Bellman) + Long memory (Monte Carlo)"""
    
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), learning_rate=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Episode memory for Monte Carlo
        self.episode_memory = []
        
    def train_step(self, state, action, reward, next_state, done):
        """Immediate training using Bellman equation (for compatibility)"""
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        
        # Store for Monte Carlo training later
        self.episode_memory.append({
            'state': state.squeeze().numpy(),
            'action': action.squeeze().numpy(), 
            'reward': reward.item()
        })
        
        # Traditional Q-learning for immediate feedback
        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Handle action indexing properly
            if len(action[idx].shape) > 0:  # One-hot or multi-dim
                if action[idx].dim() == 0:  # Single value
                    action_idx = action[idx].item()
                else:  # One-hot or array
                    action_idx = torch.argmax(action[idx]).item()
            else:
                action_idx = action[idx].item()
            
            target[idx][action_idx] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_long_memory(self, memory):
        """Train using Monte Carlo on stored episodes"""
        if len(memory) == 0:
            return
        
        # Sample from memory
        batch = memory
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate Monte Carlo returns for each trajectory
        returns_list = []
        current_return = 0
        
        # Process in reverse order to calculate returns
        for i in reversed(range(len(rewards))):
            if dones[i]:  # End of episode
                current_return = rewards[i]
            else:
                current_return = rewards[i] + self.gamma * current_return
            returns_list.insert(0, current_return)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        returns_tensor = torch.tensor(returns_list, dtype=torch.float)
        
        # Training
        pred_q_values = self.model(states)
        target_q_values = pred_q_values.clone()
        
        for idx in range(len(actions)):
            # Handle action indexing
            if len(actions[idx].shape) > 0:
                action_idx = torch.argmax(actions[idx]).item()
            else:
                action_idx = actions[idx].item()
            
            target_q_values[idx][action_idx] = returns_tensor[idx]
        
        self.optimizer.zero_grad()
        loss = self.criterion(target_q_values, pred_q_values)
        loss.backward()
        self.optimizer.step()

# Fixed QTrainer (original with proper indexing)
class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # FIX: Proper action indexing
            if action[idx].dim() == 0:  # Single scalar
                action_idx = action[idx].item()
            else:  # Array/tensor with multiple elements
                action_idx = torch.argmax(action[idx]).item()
            
            target[idx][action_idx] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Debug function
def debug_tensor_shapes(state, action, reward, next_state, done):
    """Debug function to check tensor shapes"""
    print("=== DEBUGGING TENSOR SHAPES ===")
    print(f"state: {np.array(state).shape} - {type(state)}")
    print(f"action: {np.array(action).shape} - {type(action)} - {action}")
    print(f"reward: {reward} - {type(reward)}")
    print(f"next_state: {np.array(next_state).shape} - {type(next_state)}")
    print(f"done: {done} - {type(done)}")
    print("================================")
