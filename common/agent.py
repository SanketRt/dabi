import torch
import torch.nn as nn
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomAgent:
    """Random agent for benchmarking"""
    def __init__(self, env):
        self.env = env
    
    def act(self, state, valid_actions, training=True):
        """Randomly select from valid actions"""
        return random.choice(valid_actions)
    
    def remember(self, state, action, reward, next_state, done):
        """No learning for random agent"""
        pass
    
    def save_model(self, filename):
        """No model to save"""
        pass
    
    def load_model(self, filename):
        """No model to load"""
        pass