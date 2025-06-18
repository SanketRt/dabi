import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import DQN, PrioritizedReplayBuffer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    """Enhanced DQN Agent with stable training"""
    def __init__(self, env, lr=5e-4, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # State and action sizes
        sample_state = env.get_state()
        self.state_size = len(sample_state)
        self.action_size = env.n_edges
        
        print(f"State size: {self.state_size}, Action size: {self.action_size}")
        
        # Network architecture
        hidden_size = 256  # Fixed size for stability
        
        # Neural networks
        self.q_network = DQN(self.state_size, hidden_size, self.action_size).to(device)
        self.target_network = DQN(self.state_size, hidden_size, self.action_size).to(device)
        self.update_target_network()
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        # Experience replay
        self.memory = PrioritizedReplayBuffer(20000, alpha=0.6)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.win_rates = []
        self.episode_lengths = []
        self.q_values = []
        
    def update_target_network(self, tau=1.0):
        """Soft update of target network"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def act(self, state, valid_actions, training=True):
        """Action selection with epsilon-greedy"""
        if training and random.random() <= self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            
            # Track Q-values for monitoring
            self.q_values.append(q_values.max().item())
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            for i in range(self.action_size):
                if i not in valid_actions:
                    masked_q_values[0, i] = -float('inf')
            
            return masked_q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        """Training with stabilized loss"""
        if len(self.memory) < batch_size * 4:
            return 0
        
        batch_data = self.memory.sample(batch_size, beta=0.4)
        if batch_data[0] is None:
            return 0
        
        states, actions, rewards, next_states, dones, indices, weights = batch_data
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN with clipped targets
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            # Clip target Q-values to prevent explosion
            next_q_values = torch.clamp(next_q_values, -10.0, 10.0)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            target_q_values = torch.clamp(target_q_values, -10.0, 10.0)
        
        # Compute TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Huber loss (more stable than MSE for outliers)
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (weights * loss).mean()
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes=2000, target_update_freq=50, evaluation_freq=100):
        """Training loop with improved stability"""
        best_win_rate = 0
        episode_rewards = []
        recent_wins = 0
        
        for episode in tqdm(range(episodes), desc="Training"):
            state = self.env.reset()
            total_reward = 0
            step_count = 0
            
            while not self.env.done:
                valid_actions = self.env.get_valid_actions()
                action = self.act(state, valid_actions, training=True)
                next_state, reward, done = self.env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Train more frequently with smaller batches
                if len(self.memory) > 1000 and step_count % 2 == 0:
                    loss = self.replay(batch_size=32)
                    if loss > 0:
                        self.losses.append(loss)
            
            # Soft update target network
            if episode % 10 == 0:
                self.update_target_network(tau=0.01)
            
            # Hard update less frequently
            if episode % target_update_freq == 0:
                self.update_target_network(tau=1.0)
            
            # Track metrics
            episode_rewards.append(total_reward)
            self.rewards.append(total_reward)
            self.episode_lengths.append(step_count)
            
            # Check win
            if self.env.scores[0] > self.env.scores[1]:
                recent_wins += 1
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update learning rate
            self.scheduler.step()
            
            # Periodic evaluation and reporting
            if (episode + 1) % evaluation_freq == 0:
                win_rate = recent_wins / evaluation_freq
                self.win_rates.append(win_rate)
                
                avg_reward = np.mean(episode_rewards[-evaluation_freq:])
                avg_length = np.mean(self.episode_lengths[-evaluation_freq:])
                avg_loss = np.mean(self.losses[-1000:]) if self.losses else 0
                avg_q = np.mean(self.q_values[-1000:]) if self.q_values else 0
                
                print(f"\nEpisode {episode + 1}:")
                print(f"  Win Rate: {win_rate:.1%}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Episode Length: {avg_length:.1f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Avg Q-value: {avg_q:.2f}")
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Save best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self.save_model(f"best_model_grid{self.env.grid_size}.pth")
                
                recent_wins = 0
                
                # Clear old Q-values to prevent memory issues
                if len(self.q_values) > 10000:
                    self.q_values = self.q_values[-10000:]
    
    def evaluate(self, num_games=100, render_first=5):
        """Evaluate the trained agent"""
        wins = 0
        draws = 0
        total_rewards = []
        game_lengths = []
        
        self.q_network.eval()
        
        for i in range(num_games):
            state = self.env.reset()
            total_reward = 0
            moves = 0
            
            while not self.env.done:
                valid_actions = self.env.get_valid_actions()
                action = self.act(state, valid_actions, training=False)
                
                if render_first and i < render_first:
                    print(f"Game {i+1}, Move {moves+1}: Player {self.env.current_player+1} -> Action {action}")
                
                state, reward, done = self.env.step(action)
                total_reward += reward
                moves += 1
            
            total_rewards.append(total_reward)
            game_lengths.append(moves)
            
            if render_first and i < render_first:
                print(f"Game {i+1} finished: P1={self.env.scores[0]}, P2={self.env.scores[1]}, Moves={moves}")
                print(f"Total reward: {total_reward:.2f}")
                print("-" * 40)
            
            if self.env.scores[0] > self.env.scores[1]:
                wins += 1
            elif self.env.scores[0] == self.env.scores[1]:
                draws += 1
        
        self.q_network.train()
        
        win_rate = wins / num_games
        draw_rate = draws / num_games
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(game_lengths)
        
        return {
            'win_rate': win_rate,
            'draw_rate': draw_rate, 
            'avg_reward': avg_reward,
            'avg_game_length': avg_length,
            'reward_std': np.std(total_rewards)
        }
    
    def save_model(self, filename):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'grid_size': self.env.grid_size,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filename)
    
    def load_model(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(filename, map_location=device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def plot_training_progress(self):
        """Enhanced training progress visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Rewards
        if self.rewards:
            window = 100
            smoothed_rewards = [np.mean(self.rewards[max(0, i-window):i+1]) 
                              for i in range(len(self.rewards))]
            axes[0,0].plot(smoothed_rewards)
            axes[0,0].set_title('Average Reward (100-episode window)')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Reward')
            axes[0,0].grid(True)
        
        # Win rates
        if self.win_rates:
            episodes = np.arange(100, len(self.win_rates) * 100 + 1, 100)
            axes[0,1].plot(episodes, self.win_rates)
            axes[0,1].set_title('Win Rate Over Time')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Win Rate')
            axes[0,1].grid(True)
            axes[0,1].set_ylim(0, 1)
        
        # Losses (log scale)
        if self.losses:
            window = 100
            smoothed_losses = [np.mean(self.losses[max(0, i-window):i+1]) 
                              for i in range(len(self.losses))]
            axes[0,2].semilogy(smoothed_losses)
            axes[0,2].set_title('Training Loss (100-step window, log scale)')
            axes[0,2].set_xlabel('Training Step')
            axes[0,2].set_ylabel('Loss')
            axes[0,2].grid(True)
        
        # Episode lengths
        if self.episode_lengths:
            window = 100
            smoothed_lengths = [np.mean(self.episode_lengths[max(0, i-window):i+1]) 
                               for i in range(len(self.episode_lengths))]
            axes[1,0].plot(smoothed_lengths)
            axes[1,0].set_title('Episode Length (100-episode window)')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Moves per Game')
            axes[1,0].grid(True)
        
        # Q-values
        if self.q_values:
            window = 1000
            smoothed_q = [np.mean(self.q_values[max(0, i-window):i+1]) 
                         for i in range(0, len(self.q_values), 100)]
            axes[1,1].plot(smoothed_q)
            axes[1,1].set_title('Average Q-values (1000-step window)')
            axes[1,1].set_xlabel('Steps (x100)')
            axes[1,1].set_ylabel('Q-value')
            axes[1,1].grid(True)
        
        # Loss distribution
        if len(self.losses) > 1000:
            recent_losses = self.losses[-1000:]
            axes[1,2].hist(recent_losses, bins=50, alpha=0.7)
            axes[1,2].set_title('Recent Loss Distribution (last 1000 steps)')
            axes[1,2].set_xlabel('Loss')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].grid(True)
            axes[1,2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'training_progress_grid{self.env.grid_size}.png', dpi=300, bbox_inches='tight')
        plt.show()
