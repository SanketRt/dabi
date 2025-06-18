# Dots and Boxes AI Agent

An advanced reinforcement learning project that implements and compares two deep learning agents—**DQN** and **PPO**—for playing the classic Dots and Boxes game.

## See It In Action

![DQN Agent vs Random Agent](assets/dqn_vs_random.gif)

*Watch our trained DQN agent (Player 1) compete against a random agent (Player 2).*

---

## Project Overview

This repository provides two independent agents for Dots and Boxes:

* **Deep Q-Network (DQN)**: Uses a dueling architecture with prioritized experience replay and epsilon-greedy exploration.
* **Proximal Policy Optimization (PPO)**: Implements an actor-critic network with generalized advantage estimation (GAE) and clipped surrogate objectives.

Both agents share a common environment, utility functions, GUI, and benchmarking tools. You can train, evaluate, and benchmark them side by side.

## Key Features

* **Dueling DQN Architecture**: Value and advantage streams with batch normalization and dropout
* **Prioritized Experience Replay**: Focus learning on significant transitions
* **Actor-Critic with PPO**: Stable policy gradient updates using clipping and multiple epochs
* **Generalized Advantage Estimation**: Reduces bias-variance trade-off in advantage computation
* **Multi-Grid Support**: Configurable grid sizes from 2×2 up to 5×5
* **Interactive GUI**: Real-time visualization of any agent matchup
* **Comprehensive Benchmarking**: Head-to-head tournaments, win/draw rates, reward distributions, game lengths
* **GIF Recording**: Automatic demo creation via built-in recorder
* **Modular Design**: Clean separation of DQN, PPO, and shared components

## Project Structure

```
dots-and-boxes-ai/
├── common/              
│   ├── environment.py
│   ├── utils.py
│   ├── benchmark.py
│   └── gui.py
├── dqn/                 
│   ├── __init__.py
│   ├── models.py        
│   ├── dqn_agent.py     
│   └── train_dqn.py     
├── ppo/                 
│   ├── __init__.py
│   ├── ppo_agent.py    
│   └── train_ppo.py     
└── tools/              
    ├── make_gif.py     
    └── watch_agent_play.py  
```

## Quick Start

### Installation

```bash
git clone https://github.com/SanketRt/dabi.git
cd dabi
pip install -r requirements.txt
```

### Training Agents

**DQN**

```bash
python -m dqn.train_dqn \
  --train \
  --grid-size 3 \
  --episodes 3000 \
  --lr 5e-4
```

**PPO**

```bash
python -m ppo.train_ppo \
  --grid-size 3 \
  --episodes 2000 \
  --lr 3e-4
```

### Evaluating Agents

```bash
# DQN evaluation
python -m dqn.train_dqn \
  --evaluate \
  --grid-size 3 \
  --model-path dqn_model_3x3.pth

# PPO evaluation
python -m ppo.train_ppo \
  --grid-size 3 \
  --model-path ppo_model_3x3.pth
```

### Benchmarking

```bash
python -m common.benchmark \
  --grid-size 3 \
  --dqn-path dqn_model_3x3.pth \
  --ppo-path ppo_model_3x3.pth \
  --num-games 200
```

### Interactive GUI Demo

```bash
python tools/watch_agent_play.py
```

### Create Demo GIFs

```bash
python tools/make_gif.py
```

Follow the on-screen prompts to select models, grid size, and recording options.

## Algorithm Details

### DQN Architecture

* **Feature Extraction**: Fully connected layers with batch normalization and dropout
* **Dueling Streams**: Separate value V(s) and advantage A(s,a) estimates
* **Q-Value Combination**: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
* **Training Mechanisms**:

  * Epsilon-greedy exploration
  * Target network updates
  * Prioritized replay buffer
  * Gradient clipping

### PPO Architecture

* **Actor-Critic Network**: Shared feature layers feeding separate policy (actor) and value (critic) heads
* **GAE**: Generalized advantage estimation for more stable advantage targets
* **Clipped Surrogate Objective**: Prevents large policy updates
* **Multiple Epochs per Batch**: Improves sample efficiency

## Reward Structure

| Action              | Reward                | Cap  | Description                           |
| ------------------- | --------------------- | ---- | ------------------------------------- |
| Box completion      | +0.5 per box          | +2.0 | Encourages box capturing              |
| Creating 3-edge box | -0.1 per box          | -0.5 | Discourages giving away opportunities |
| Regular move        | -0.01                 |      | Encourages efficient play             |
| Win/Loss outcome    | ±1.0                  |      | Final match bonus or penalty          |
| Score margin        | ±0.5×tanh(diff/total) |      | Proportional endgame bonus/penalty    |
| Invalid move        | -1.0                  |      | Prevents illegal actions              |

## Performance Results

| Matchup          | Win Rate | Avg Game Length | Notes                              |
| ---------------- | -------- | --------------- | ---------------------------------- |
| DQN vs Random    | 98.0%    | 60.0 moves      | Strong baseline performance        |
| PPO vs Random    | 96.5%    | 60.0 moves      | Comparable to DQN                  |
| PPO vs DQN       | 52.0%    | 60.0 moves      | Slight edge to PPO in head‑to‑head |
| Random vs Random | 50.5%    | 60.0 moves      | Baseline random performance        |

## Configuration

Parameters are stored in YAML files for easy tuning:

```yaml
# configs/training.yaml
training:
  episodes: 3000
  batch_size: 64
  learning_rate: 5e-4
  gamma: 0.95
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995

model:
  hidden_size: 512
  dropout_rate: 0.1

environment:
  grid_size: 3
  reward_clipping: [-3.0, 3.0]
```

## Monitoring Training

* **Progress Bars**: Real-time episode tracking via tqdm
* **TensorBoard**: Optional logging of losses and rewards
* **Checkpointing**: Best model saving based on evaluation metrics

## Game Rules

Dots and Boxes is a two-player turn-based game:

1. Players alternate drawing a line between adjacent dots on a grid.
2. Completing the fourth side of a box earns a point and an extra move.
3. The game ends when all lines are drawn.
4. The player with the most boxes wins.

## Research References

* Mnih et al. (2015). Human-level control through deep reinforcement learning.
* Wang et al. (2016). Dueling network architectures for deep reinforcement learning.
* Schaul et al. (2016). Prioritized experience replay.
* Schulman et al. (2017). Proximal policy optimization algorithms.
