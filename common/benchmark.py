import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from collections import defaultdict
from common.environment import DotsAndBoxesEnv
from dqn.dqn_agent import DQNAgent
from common.agent import RandomAgent

class GameBenchmark:
    """Benchmark class for comparing agent performance"""
    
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.env = DotsAndBoxesEnv(grid_size)
        self.results = {}
    
    def play_game(self, agent1, agent2, render=False):
        """Play a single game between two agents"""
        self.env.reset()
        move_count = 0
        move_history = []
        
        while not self.env.done:
            current_player = self.env.current_player
            agent = agent1 if current_player == 0 else agent2
            
            state = self.env.get_state()
            valid_actions = self.env.get_valid_actions()
            action = agent.act(state, valid_actions, training=False)
            
            if render:
                print(f"Move {move_count + 1}: Player {current_player + 1} chooses action {action}")
            
            next_state, reward, done = self.env.step(action)
            move_history.append({
                'player': current_player,
                'action': action,
                'state_before': state,
                'state_after': next_state,
                'reward': reward
            })
            
            move_count += 1
        
        result = {
            'winner': 0 if self.env.scores[0] > self.env.scores[1] else (1 if self.env.scores[1] > self.env.scores[0] else -1),
            'scores': self.env.scores.copy(),
            'moves': move_count,
            'move_history': move_history
        }
        
        if render:
            print(f"Game finished: Player 1: {self.env.scores[0]}, Player 2: {self.env.scores[1]}")
            print(f"Winner: {'Player 1' if result['winner'] == 0 else 'Player 2' if result['winner'] == 1 else 'Draw'}")
            print(f"Total moves: {move_count}")
        
        return result
    
    def run_tournament(self, agent1, agent2, num_games=100, agent1_name="Agent1", agent2_name="Agent2"):
        """Run a tournament between two agents"""
        print(f"\nRunning tournament: {agent1_name} vs {agent2_name}")
        print(f"Games: {num_games}, Grid size: {self.grid_size}x{self.grid_size}")
        print("-" * 60)
        
        results = {
            'games': [],
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'agent1_scores': [],
            'agent2_scores': [],
            'game_lengths': [],
            'execution_times': []
        }
        
        for i in tqdm(range(num_games), desc="Playing games"):
            start_time = time.time()
            
            # Alternate who goes first
            if i % 2 == 0:
                game_result = self.play_game(agent1, agent2)
                p1_score, p2_score = game_result['scores']
            else:
                game_result = self.play_game(agent2, agent1)
                p2_score, p1_score = game_result['scores']  
                game_result['winner'] = 1 - game_result['winner'] if game_result['winner'] != -1 else -1
            
            execution_time = time.time() - start_time
            
            results['games'].append(game_result)
            results['agent1_scores'].append(p1_score)
            results['agent2_scores'].append(p2_score)
            results['game_lengths'].append(game_result['moves'])
            results['execution_times'].append(execution_time)
            
            if game_result['winner'] == 0:
                results['agent1_wins'] += 1
            elif game_result['winner'] == 1:
                results['agent2_wins'] += 1
            else:
                results['draws'] += 1
        
        # Calculate statistics
        results['agent1_win_rate'] = results['agent1_wins'] / num_games
        results['agent2_win_rate'] = results['agent2_wins'] / num_games
        results['draw_rate'] = results['draws'] / num_games
        results['avg_game_length'] = np.mean(results['game_lengths'])
        results['avg_execution_time'] = np.mean(results['execution_times'])
        results['avg_agent1_score'] = np.mean(results['agent1_scores'])
        results['avg_agent2_score'] = np.mean(results['agent2_scores'])
        
        self.results[f"{agent1_name}_vs_{agent2_name}"] = results
        
        # Print summary
        print(f"\nTournament Results:")
        print(f"{agent1_name} wins: {results['agent1_wins']} ({results['agent1_win_rate']:.1%})")
        print(f"{agent2_name} wins: {results['agent2_wins']} ({results['agent2_win_rate']:.1%})")
        print(f"Draws: {results['draws']} ({results['draw_rate']:.1%})")
        print(f"Average game length: {results['avg_game_length']:.1f} moves")
        print(f"Average execution time: {results['avg_execution_time']:.3f}s per game")
        print(f"Average scores - {agent1_name}: {results['avg_agent1_score']:.1f}, {agent2_name}: {results['avg_agent2_score']:.1f}")
        
        return results
    
    def benchmark_vs_random(self, agent, num_games=200, agent_name="DQN"):
        """Benchmark an agent against random play"""
        env = DotsAndBoxesEnv(self.grid_size)
        random_agent = RandomAgent(env)
        
        return self.run_tournament(agent, random_agent, num_games, agent_name, "Random")
    
    def benchmark_self_play(self, agent, num_games=100, agent_name="DQN"):
        """Benchmark an agent against itself"""
        return self.run_tournament(agent, agent, num_games, f"{agent_name}_P1", f"{agent_name}_P2")
    
    def analyze_game_patterns(self, results):
        """Analyze patterns in game results"""
        games = results['games']
        
        analysis = {
            'first_move_advantage': 0,
            'game_length_distribution': defaultdict(int),
            'score_distributions': {
                'agent1': defaultdict(int),
                'agent2': defaultdict(int)
            },
            'move_patterns': defaultdict(int)
        }
        
        # Analyze first move advantage
        first_player_wins = sum(1 for i, game in enumerate(games) 
                               if (i % 2 == 0 and game['winner'] == 0) or 
                                  (i % 2 == 1 and game['winner'] == 1))
        analysis['first_move_advantage'] = first_player_wins / len(games)
        
        # Game length distribution
        for game in games:
            analysis['game_length_distribution'][game['moves']] += 1
        
        # Score distributions
        for i, game in enumerate(games):
            if i % 2 == 0:
                p1_score, p2_score = game['scores']
            else:
                p2_score, p1_score = game['scores']
            
            analysis['score_distributions']['agent1'][p1_score] += 1
            analysis['score_distributions']['agent2'][p2_score] += 1
        
        return analysis
    
    def plot_results(self, results_dict=None):
        """Plot benchmark results"""
        if results_dict is None:
            results_dict = self.results
        
        if not results_dict:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Benchmark Results - {self.grid_size}x{self.grid_size} Grid', fontsize=16)
        
        for idx, (match_name, results) in enumerate(results_dict.items()):
            color = f'C{idx}'
            
            # Win rates
            win_rates = [results['agent1_win_rate'], results['agent2_win_rate'], results['draw_rate']]
            labels = match_name.split('_vs_') + ['Draws']
            axes[0, 0].bar([f"{labels[0]}", f"{labels[1]}", "Draws"], win_rates, 
                          color=[color, f'C{idx+1}', 'gray'], alpha=0.7, 
                          label=match_name if idx == 0 else "")
        
        axes[0, 0].set_title('Win Rates')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # Game length distribution
        for idx, (match_name, results) in enumerate(results_dict.items()):
            axes[0, 1].hist(results['game_lengths'], bins=20, alpha=0.6, 
                           label=match_name, color=f'C{idx}')
        
        axes[0, 1].set_title('Game Length Distribution')
        axes[0, 1].set_xlabel('Moves per Game')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Score distributions
        for idx, (match_name, results) in enumerate(results_dict.items()):
            axes[0, 2].scatter(results['agent1_scores'], results['agent2_scores'], 
                              alpha=0.6, label=match_name, color=f'C{idx}')
        
        axes[0, 2].set_title('Score Distributions')
        axes[0, 2].set_xlabel('Agent 1 Score')
        axes[0, 2].set_ylabel('Agent 2 Score')
        axes[0, 2].legend()
        
        # Performance over time (running win rate)
        for idx, (match_name, results) in enumerate(results_dict.items()):
            games = results['games']
            running_wins = []
            agent1_wins = 0
            
            for i, game in enumerate(games):
                if game['winner'] == 0:
                    agent1_wins += 1
                running_wins.append(agent1_wins / (i + 1))
            
            axes[1, 0].plot(running_wins, label=f"{match_name} (Agent1)", color=f'C{idx}')
        
        axes[1, 0].set_title('Running Win Rate (Agent 1)')
        axes[1, 0].set_xlabel('Game Number')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # Execution time
        for idx, (match_name, results) in enumerate(results_dict.items()):
            axes[1, 1].hist(results['execution_times'], bins=20, alpha=0.6, 
                           label=match_name, color=f'C{idx}')
        
        axes[1, 1].set_title('Execution Time Distribution')
        axes[1, 1].set_xlabel('Time per Game (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # Average scores comparison
        match_names = list(results_dict.keys())
        agent1_avg_scores = [results['avg_agent1_score'] for results in results_dict.values()]
        agent2_avg_scores = [results['avg_agent2_score'] for results in results_dict.values()]
        
        x = np.arange(len(match_names))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, agent1_avg_scores, width, label='Agent 1', alpha=0.7)
        axes[1, 2].bar(x + width/2, agent2_avg_scores, width, label='Agent 2', alpha=0.7)
        
        axes[1, 2].set_title('Average Scores')
        axes[1, 2].set_xlabel('Match')
        axes[1, 2].set_ylabel('Average Score')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels([name.replace('_vs_', ' vs ') for name in match_names], rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'benchmark_results_grid{self.grid_size}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename=None):
        """Save benchmark results to file"""
        if filename is None:
            filename = f'benchmark_results_grid{self.grid_size}.txt'
        
        with open(filename, 'w') as f:
            f.write(f"Dots and Boxes Benchmark Results\n")
            f.write(f"Grid Size: {self.grid_size}x{self.grid_size}\n")
            f.write("=" * 50 + "\n\n")
            
            for match_name, results in self.results.items():
                f.write(f"Match: {match_name.replace('_vs_', ' vs ')}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Games: {len(results['games'])}\n")
                f.write(f"Agent 1 Wins: {results['agent1_wins']} ({results['agent1_win_rate']:.1%})\n")
                f.write(f"Agent 2 Wins: {results['agent2_wins']} ({results['agent2_win_rate']:.1%})\n")
                f.write(f"Draws: {results['draws']} ({results['draw_rate']:.1%})\n")
                f.write(f"Average Game Length: {results['avg_game_length']:.1f} moves\n")
                f.write(f"Average Execution Time: {results['avg_execution_time']:.3f}s\n")
                f.write(f"Average Scores - Agent 1: {results['avg_agent1_score']:.1f}, Agent 2: {results['avg_agent2_score']:.1f}\n")
                
                # Game patterns analysis
                analysis = self.analyze_game_patterns(results)
                f.write(f"First Move Advantage: {analysis['first_move_advantage']:.1%}\n")
                f.write("\n")
        
        print(f"Results saved to {filename}")

def run_comprehensive_benchmark():
    """Run a comprehensive benchmark suite"""
    grid_size = 3
    benchmark = GameBenchmark(grid_size)
    
    print("=" * 60)
    print("COMPREHENSIVE DOTS AND BOXES BENCHMARK")
    print("=" * 60)
    
    env = DotsAndBoxesEnv(grid_size)
    random_agent = RandomAgent(env)
    
    # Check if we have a trained DQN model
    dqn_agent = None
    model_path = f'best_model_grid{grid_size}.pth'
    
    if os.path.exists(model_path):
        print(f"Loading trained DQN model from {model_path}")
        dqn_agent = DQNAgent(env)
        dqn_agent.load_model(model_path)
        dqn_agent.epsilon = 0 
    else:
        print(f"No trained model found at {model_path}")
        print("Training a new DQN agent for benchmarking...")
        dqn_agent = DQNAgent(env)
        dqn_agent.train(episodes=1000, evaluation_freq=200)
        dqn_agent.save_model(model_path)
    
    # Benchmark 1: DQN vs Random
    print("\n" + "="*40)
    print("BENCHMARK 1: DQN vs Random Agent")
    print("="*40)
    benchmark.benchmark_vs_random(dqn_agent, num_games=200, agent_name="DQN")
    
    # Benchmark 2: Random vs Random (baseline)
    print("\n" + "="*40)
    print("BENCHMARK 2: Random vs Random (Baseline)")
    print("="*40)
    env2 = DotsAndBoxesEnv(grid_size)
    random_agent2 = RandomAgent(env2)
    benchmark.run_tournament(random_agent, random_agent2, num_games=200, agent1_name="Random_A", agent2_name="Random_B")
    
    # Benchmark 3: DQN Self-play
    print("\n" + "="*40)
    print("BENCHMARK 3: DQN Self-Play")
    print("="*40)
    benchmark.benchmark_self_play(dqn_agent, num_games=100, agent_name="DQN")
    
    benchmark.plot_results()
    benchmark.save_results()
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for match_name, results in benchmark.results.items():
        print(f"\n{match_name.replace('_vs_', ' vs ')}:")
        analysis = benchmark.analyze_game_patterns(results)
        print(f"  First move advantage: {analysis['first_move_advantage']:.1%}")
        print(f"  Most common game length: {max(analysis['game_length_distribution'], key=analysis['game_length_distribution'].get)} moves")
        print(f"  Game length range: {min(results['game_lengths'])}-{max(results['game_lengths'])} moves")
    
    return benchmark

def quick_demo():
    """Quick demonstration of agents playing"""
    print("Quick Demo: Watching agents play...")
    grid_size = 3
    env = DotsAndBoxesEnv(grid_size)
    benchmark = GameBenchmark(grid_size)
    random_agent1 = RandomAgent(env)
    random_agent2 = RandomAgent(env)
    print("\nDemo Game: Random vs Random")
    print("-" * 40)
    result = benchmark.play_game(random_agent1, random_agent2, render=True)
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        quick_demo()
    else:
        run_comprehensive_benchmark()