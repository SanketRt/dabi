import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from common.environment import DotsAndBoxesEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, env: DotsAndBoxesEnv, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                 k_epochs=4, ent_coef=0.01, vf_coef=0.5, gae_lambda=0.95):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.gae_lambda = gae_lambda

        self.state_dim = len(env.get_state())
        self.action_dim = env.n_edges

        self.policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, valid_actions):
        state = torch.FloatTensor(state).to(device)
        action_probs, state_value = self.policy(state)
        mask = torch.zeros(self.action_dim, device=device)
        mask[valid_actions] = 1.0
        probs = action_probs * mask
        probs = probs / probs.sum()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), state_value

    def compute_gae(self, rewards, values, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            advantage = delta + self.gamma * self.gae_lambda * mask * advantage
            advantages.insert(0, advantage)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, memory):
        states = torch.stack(memory['states']).to(device)
        actions = torch.tensor(memory['actions']).to(device)
        old_logprobs = torch.stack(memory['logprobs']).detach().to(device)
        returns = torch.tensor(memory['returns']).to(device)
        advantages = torch.tensor(memory['advantages']).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            action_probs, state_values = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = (new_logprobs - old_logprobs).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((returns - state_values.squeeze()) ** 2).mean()

            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def train(self, episodes=2000):
        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': []}
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                valid_actions = self.env.get_valid_actions()
                action, logprob, entropy, value = self.select_action(state, valid_actions)
                next_state, reward, done = self.env.step(action)

                memory['states'].append(torch.FloatTensor(state))
                memory['actions'].append(action)
                memory['logprobs'].append(logprob.detach())
                memory['rewards'].append(reward)
                memory['dones'].append(done)
                memory['values'].append(value.item())

                state = next_state

            _, _, _, last_value = self.select_action(state, [0])
            memory['values'].append(last_value.item())

            advantages, returns = self.compute_gae(memory['rewards'], memory['values'], memory['dones'])
            memory['advantages'] = advantages
            memory['returns'] = returns
            self.update(memory)

            for key in ['states', 'actions', 'logprobs', 'rewards', 'dones', 'values']:
                memory[key] = []

    def evaluate(self, num_games=100):
        wins, draws, rewards, lengths = 0, 0, [], []
        for _ in range(num_games):
            state = self.env.reset()
            done = False
            total_reward, moves = 0, 0
            while not done:
                valid_actions = self.env.get_valid_actions()
                action, _, _, _ = self.select_action(state, valid_actions)
                state, reward, done = self.env.step(action)
                total_reward += reward
                moves += 1
            rewards.append(total_reward)
            lengths.append(moves)
            if self.env.scores[0] > self.env.scores[1]:
                wins += 1
            elif self.env.scores[0] == self.env.scores[1]:
                draws += 1
        return {'win_rate': wins / num_games, 'draw_rate': draws / num_games,'avg_reward': np.mean(rewards), 'reward_std': np.std(rewards),'avg_game_length': np.mean(lengths)}
    def act(self, state, valid_actions, training=False):
        """
        Select action for benchmarking/tournament.
        Ignores exploration flag (PPO is always “greedy” w.r.t. its policy).
        """
        # reuse select_action, but drop extras
        action, logprob, entropy, value = self.select_action(state, valid_actions)
        return action
