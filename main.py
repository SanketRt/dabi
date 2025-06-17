#!/usr/bin/env python3
"""
Main script for training and evaluating Dots and Boxes AI agents
"""

import argparse
import os
import sys

from environment import DotsAndBoxesEnv
from agent import DQNAgent, RandomAgent
from benchmark import GameBenchmark
from gui import DotsAndBoxesGUI
from utils import analyze_reward_structure, validate_environment

def train_agent(grid_size=3, episodes=3000, lr=5e-4, save_path=None):
    """Train a new DQN agent"""
    print(f"Training DQN agent on {grid_size}x{grid_size} grid for {episodes} episodes")
    
    # Analyze reward structure first
    analyze_reward_structure(grid_size)
    
    # Create environment and agent
    env = DotsAndBoxesEnv(grid_size=grid_size)
    agent = DQNAgent(env, lr=lr, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
    
    # Train the agent
    print(f"\nStarting training...")
    agent.train(episodes=episodes, target_update_freq=50, evaluation_freq=100)
    
    # Save the trained model
    if save_path is None:
        save_path = f'trained_model_grid{grid_size}.pth'
    
    agent.save_model(save_path)
    print(f"\nModel saved to {save_path}")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    results = agent.evaluate(num_games=200, render_first=3)
    
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Draw Rate: {results['draw_rate']:.1%}")
    print(f"Average Reward: {results['avg_reward']:.2f} ± {results['reward_std']:.2f}")
    print(f"Average Game Length: {results['avg_game_length']:.1f} moves")
    
    # Plot training progress
    agent.plot_training_progress()
    
    return agent

def evaluate_agent(model_path, grid_size=3, num_games=100):
    """Evaluate a trained agent"""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return None
    
    print(f"Evaluating agent from {model_path}")
    
    # Load the agent
    env = DotsAndBoxesEnv(grid_size=grid_size)
    agent = DQNAgent(env)
    agent.load_model(model_path)
    agent.epsilon = 0  # No exploration during evaluation
    
    # Evaluate
    results = agent.evaluate(num_games=num_games, render_first=5)
    
    print(f"\nEvaluation Results ({num_games} games):")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Draw Rate: {results['draw_rate']:.1%}")
    print(f"Average Reward: {results['avg_reward']:.2f} ± {results['reward_std']:.2f}")
    print(f"Average Game Length: {results['avg_game_length']:.1f} moves")
    
    return agent

def run_gui_demo(grid_size=3, model_path=None):
    """Run the GUI demonstration"""
    env = DotsAndBoxesEnv(grid_size)
    gui = DotsAndBoxesGUI(grid_size)
    
    # Set up agents
    if model_path and os.path.exists(model_path):
        print(f"Loading DQN agent from {model_path}")
        dqn_agent = DQNAgent(env)
        dqn_agent.load_model(model_path)
        dqn_agent.epsilon = 0  # No exploration
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

def run_benchmark(grid_size=3, model_path=None):
    """Run comprehensive benchmarking"""
    benchmark = GameBenchmark(grid_size)
    
    # Create agents
    env = DotsAndBoxesEnv(grid_size)
    random_agent = RandomAgent(env)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading DQN agent from {model_path}")
        dqn_agent = DQNAgent(env)
        dqn_agent.load_model(model_path)
        dqn_agent.epsilon = 0
        
        # Benchmark DQN vs Random
        benchmark.benchmark_vs_random(dqn_agent, num_games=200, agent_name="DQN")
        
        # DQN self-play
        benchmark.benchmark_self_play(dqn_agent, num_games=100, agent_name="DQN")
    else:
        print("No DQN model provided, running random vs random")
    
    # Random vs Random baseline
    env2 = DotsAndBoxesEnv(grid_size)
    random_agent2 = RandomAgent(env2)
    benchmark.run_tournament(random_agent, random_agent2, num_games=200, 
                           agent1_name="Random_A", agent2_name="Random_B")
    
    # Plot and save results
    benchmark.plot_results()
    benchmark.save_results()
    
    return benchmark

def main():
    parser = argparse.ArgumentParser(description="Dots and Boxes AI Agent")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train a new agent")
    mode_group.add_argument("--evaluate", action="store_true", help="Evaluate a trained agent")
    mode_group.add_argument("--gui", action="store_true", help="Run GUI demonstration")
    mode_group.add_argument("--benchmark", action="store_true", help="Run benchmarking")
    mode_group.add_argument("--validate", action="store_true", help="Validate environment")
    
    # Common arguments
    parser.add_argument("--grid-size", type=int, default=3, choices=[2, 3, 4, 5],
                       help="Grid size for the game (default: 3)")
    parser.add_argument("--model-path", type=str, 
                       help="Path to model file (for evaluate/gui/benchmark modes)")
    
    # Training arguments
    parser.add_argument("--episodes", type=int, default=3000,
                       help="Number of training episodes (default: 3000)")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate (default: 5e-4)")
    parser.add_argument("--save-path", type=str,
                       help="Path to save trained model")
    
    # Evaluation arguments
    parser.add_argument("--num-games", type=int, default=100,
                       help="Number of games for evaluation (default: 100)")
    
    args = parser.parse_args()
    
    # Auto-detect model path if not provided
    if not args.model_path and args.grid_size:
        default_paths = [
            f'best_model_grid{args.grid_size}.pth',
            f'trained_model_grid{args.grid_size}.pth',
            f'final_dab_model_grid{args.grid_size}.pth'
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