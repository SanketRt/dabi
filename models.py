import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """Improved Deep Q-Network with normalization"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        
        # Dueling DQN architecture with batch normalization
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller initialization
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle batch size 1 for batch norm
        if x.size(0) == 1:
            self.eval()
            features = self.feature_layer(x)
            self.train()
        else:
            features = self.feature_layer(x)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        # Bound Q-values to prevent explosion
        q_values = torch.tanh(q_values) * 10.0  # Bound to [-10, 10]
        
        return q_values

class PrioritizedReplayBuffer:
    """Improved Prioritized Experience Replay Buffer with TD-error clipping"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None, None, None, None, None, None, None
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        # Clip priorities to prevent extreme sampling
        priorities = np.clip(priorities, 0.01, 10.0)
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.clip(weights, 0.1, 1.0)  # Prevent extreme weights
        
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # Clip TD-errors to prevent extreme priorities
            clipped_priority = np.clip(priority, 0.01, 10.0)
            self.priorities[idx] = max(clipped_priority, 1e-6)
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return len(self.buffer)