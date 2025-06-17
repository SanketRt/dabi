# Dots and Boxes AI Agent

A Deep Q-Network (DQN) implementation for playing Dots and Boxes.

## Project Structure

- `environment.py`: Game environment implementation
- `models.py`: Neural network models (DQN, Replay Buffer)
- `agent.py`: DQN agent and random agent implementations
- `gui.py`: Graphical interface for watching games
- `benchmark.py`: Performance benchmarking tools
- `utils.py`: Utility functions
- `main.py`: Main training and evaluation script

## Usage

### Training
```bash
python main.py --train --episodes 3000 --grid-size 3
```

### GUI Demo
```bash
python gui.py
```

### Benchmarking
```bash
python benchmark.py
```

### Quick Demo
```bash
python benchmark.py demo
```

## Requirements

- torch
- numpy
- matplotlib
- tkinter
- tqdm
