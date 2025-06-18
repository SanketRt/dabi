#!/usr/bin/env python3
"""
Main script for training and evaluating Dots and Boxes AI agents
"""

import argparse
import os
import sys
import torch
from tqdm import trange

from common.environment import DotsAndBoxesEnv
from dqn.dqn_agent import DQNAgent 
from common.agent import RandomAgent
from ppo.ppo_agent import PPOAgent
from common.benchmark import GameBenchmark
from common.gui import DotsAndBoxesGUI
from common.utils import analyze_reward_structure, validate_environment


def train_agent(grid_size=3, episodes=3000, lr=5e-4, save_path=None):
    """Train a new DQN agent"""
    print(f"Training DQN agent on {grid_size}x{grid_size} grid for {episodes} episodes")
    analyze_reward_structure(grid_size)
    env = DotsAndBoxesEnv(grid_size=grid_size)
    agent = DQNAgent(env, lr=lr, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
    print(f"\nStarting DQN training...")
    # DQNAgent.train already uses tqdm internally
    agent.train(episodes=episodes, target_update_freq=50, evaluation_freq=100)
    if save_path is None:
        save_path = f'trained_model_grid{grid_size}.pth'
    agent.save_model(save_path)
    print(f"\nModel saved to {save_path}")
    print("\n=== Final Evaluation ===")
    results = agent.evaluate(num_games=200, render_first=3)
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Draw Rate: {results['draw_rate']:.1%}")
    print(f"Average Reward: {results['avg_reward']:.2f} ± {results['reward_std']:.2f}")
    print(f"Average Game Length: {results['avg_game_length']:.1f} moves")
    agent.plot_training_progress()
    return agent


def train_ppo_agent(grid_size=3, episodes=2000, lr=3e-4, save_path=None):
    """Train a new PPO agent with tqdm progress bar"""
    print(f"Training PPO agent on {grid_size}x{grid_size} grid for {episodes} episodes")
    analyze_reward_structure(grid_size)
    env = DotsAndBoxesEnv(grid_size=grid_size)
    agent = PPOAgent(env, lr=lr)
    print(f"\nStarting PPO training...")
    # Wrap the PPO training loop in tqdm
    for ep in trange(episodes, desc="PPO Training", unit="ep"):
        agent.train(episodes=1)
    if save_path is None:
        save_path = f'ppo_model_grid{grid_size}.pth'
    torch.save(agent.policy.state_dict(), save_path)
    print(f"\nPPO model saved to {save_path}")
    print("\n=== Final PPO Evaluation ===")
    stats = agent.evaluate(num_games=200)
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"Draw Rate: {stats['draw_rate']:.1%}")
    print(f"Average Reward: {stats['avg_reward']:.2f} ± {stats['reward_std']:.2f}")
    print(f"Average Game Length: {stats['avg_game_length']:.1f} moves")
    return agent


def run_benchmark(grid_size=3, model_path=None):
    """Run comprehensive benchmarking for DQN and random"""
    benchmark = GameBenchmark(grid_size)
    
    env = DotsAndBoxesEnv(grid_size)
    random_agent = RandomAgent(env)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading DQN agent from {model_path}")
        dqn_agent = DQNAgent(env)
        dqn_agent.load_model(model_path)
        dqn_agent.epsilon = 0
        benchmark.benchmark_vs_random(dqn_agent, num_games=200, agent_name="DQN")
        benchmark.benchmark_self_play(dqn_agent, num_games=100, agent_name="DQN")
    else:
        print("No DQN model provided, running random vs random")
    
    # Random vs Random baseline
    env2 = DotsAndBoxesEnv(grid_size)
    random_agent2 = RandomAgent(env2)
    benchmark.run_tournament(random_agent, random_agent2, num_games=200, agent1_name="Random_A", agent2_name="Random_B")
    
    benchmark.plot_results()
    benchmark.save_results()
    
    return benchmark


def run_benchmark_ppo(grid_size=3, model_path=None):
    """Run benchmarking for PPO against random and DQN"""
    benchmark = GameBenchmark(grid_size)
    
    env = DotsAndBoxesEnv(grid_size)
    random_agent = RandomAgent(env)
    
    # PPO vs Random
    if model_path and os.path.exists(model_path):
        print(f"Loading PPO agent from {model_path}")
        ppo_agent = PPOAgent(env)
        ppo_agent.policy.load_state_dict(torch.load(model_path))
        benchmark.benchmark_vs_random(ppo_agent, num_games=200, agent_name="PPO")
        benchmark.benchmark_self_play(ppo_agent, num_games=100, agent_name="PPO")
        # PPO vs DQN
        print("Loading DQN agent for cross-play")
        dqn_agent = DQNAgent(env)
        # attempt to auto-detect a DQN model
        dqn_path = f'trained_model_grid{grid_size}.pth'
        if os.path.exists(dqn_path):
            dqn_agent.load_model(dqn_path)
            dqn_agent.epsilon = 0
            benchmark.run_tournament(ppo_agent, dqn_agent, num_games=200, agent1_name="PPO", agent2_name="DQN")
    else:
        print("Error: PPO model file not found for benchmarking")
        return None
    
    # Random vs Random baseline
    env2 = DotsAndBoxesEnv(grid_size)
    random_agent2 = RandomAgent(env2)
    benchmark.run_tournament(random_agent, random_agent2, num_games=200, agent1_name="Random_A", agent2_name="Random_B")
    
    benchmark.plot_results()
    benchmark.save_results()
    
    return benchmark


def run_gui_demo(grid_size=3, model_path=None):
    """Run the GUI demonstration"""
    env = DotsAndBoxesEnv(grid_size)
    gui = DotsAndBoxesGUI(grid_size)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading DQN agent from {model_path}")
        dqn_agent = DQNAgent(env)
        dqn_agent.load_model(model_path)
        dqn_agent.epsilon = 0
        agent1 = dqn_agent
    else:
        print("Using random agent for Player 1")
        agent1 = RandomAgent(env)
    
    agent2 = RandomAgent(env)
    print("Using random agent for Player 2")
    
    gui.set_agents(agent1, agent2)
    
    print("\nGUI Controls:")
    print("- Start Game: Begin automatic gameplay")
    print("- Pause: Pause/resume the game")
    print("- Next Move: Step through moves when paused")
    print("- Reset: Reset the game board")
    print("- Speed: Adjust game speed")
    
    gui.run()


def main():
    parser = argparse.ArgumentParser(description="Dots and Boxes AI Agent")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train a new DQN agent")
    mode_group.add_argument("--evaluate", action="store_true", help="Evaluate a trained DQN agent")
    mode_group.add_argument("--gui", action="store_true", help="Run GUI demonstration")
    mode_group.add_argument("--benchmark", action="store_true", help="Run DQN benchmarking")
    mode_group.add_argument("--validate", action="store_true", help="Validate environment")
    mode_group.add_argument("--train-ppo", action="store_true", help="Train a PPO agent")
    mode_group.add_argument("--benchmark-ppo", action="store_true", help="Benchmark PPO vs other agents")

    parser.add_argument("--grid-size",type=int,default=3,choices=[2, 3, 4, 5],help="Grid size for the game (default: 3)")
    parser.add_argument("--model-path",type=str,help="Path to model file (for evaluate/gui/benchmark modes)")
    parser.add_argument("--episodes",type=int,default=3000,help="Number of training episodes (default: 3000)")
    parser.add_argument("--lr",type=float,default=5e-4,help="Learning rate (default: 5e-4)")
    parser.add_argument("--save-path",type=str,help="Path to save trained model")
    parser.add_argument("--num-games",type=int,default=100,help="Number of games for evaluation (default: 100)")
    
    args = parser.parse_args()
    
    # Auto-detect model path if not provided
    if not args.model_path and args.grid_size:
        default_paths = [
            f'best_model_grid{args.grid_size}.pth',
            f'trained_model_grid{args.grid_size}.pth',
            f'ppo_model_grid{args.grid_size}.pth'
        ]
        for path in default_paths:
            if os.path.exists(path):
                args.model_path = path
                print(f"Auto-detected model: {path}")
                break
    
    try:
        if args.train:
            train_agent(
                grid_size=args.grid_size,
                episodes=args.episodes,
                lr=args.lr,
                save_path=args.save_path
            )
        elif args.train_ppo:
            train_ppo_agent(
                grid_size=args.grid_size,
                episodes=args.episodes,
                lr=args.lr,
                save_path=args.save_path
            )
        elif args.evaluate:
            if not args.model_path:
                print("Error: --model-path required for evaluation mode")
                sys.exit(1)
            evaluate_agent(
                model_path=args.model_path,
                grid_size=args.grid_size,
                num_games=args.num_games
            )
        elif args.gui:
            run_gui_demo(
                grid_size=args.grid_size,
                model_path=args.model_path
            )
        elif args.benchmark:
            run_benchmark(
                grid_size=args.grid_size,
                model_path=args.model_path
            )
        elif args.benchmark_ppo:
            if not args.model_path:
                print("Error: --model-path required for PPO benchmarking")
                sys.exit(1)
            run_benchmark_ppo(
                grid_size=args.grid_size,
                model_path=args.model_path
            )
        elif args.validate:
            validate_environment()
            analyze_reward_structure(args.grid_size)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
