import numpy as np

def analyze_reward_structure(grid_size=5):
    """Analyze expected rewards for the given grid size"""
    from environment import DotsAndBoxesEnv
    env = DotsAndBoxesEnv(grid_size=grid_size)
    
    print(f"=== Reward Analysis for {grid_size}x{grid_size} Grid ===")
    print(f"Total boxes: {env.n_boxes}")
    print(f"Total edges: {env.n_edges}")
    print(f"Max possible boxes per player: {env.n_boxes}")
    print()
    
    print("Bounded Reward Structure:")
    print("- Box completion: +0.5 per box (capped at +2.0)")
    print("- Creating 3-edge box: -0.1 per box (capped at -0.5)")
    print("- Not completing box: -0.01")
    print("- Win bonus: +1.0")
    print("- Loss penalty: -1.0")
    print("- Margin bonus/penalty: ±0.5 * tanh(score_diff/total_boxes)")
    print("- Invalid move: -1.0")
    print("- All rewards clipped to [-3.0, 3.0]")
    print()
    
    # Best case scenario
    best_reward = 2.0 + 1.0 + 0.5  # Max box reward + win + margin
    print(f"Best case reward: {best_reward:.1f}")
    
    # Worst case scenario  
    worst_reward = -0.01 * env.n_edges - 1.0 - 0.5  # All moves penalized + loss + margin
    worst_reward = max(worst_reward, -3.0)  # Clipped
    print(f"Worst case reward: {worst_reward:.1f}")
    
    # Expected close game
    avg_boxes = env.n_boxes // 2
    avg_moves = env.n_edges * 0.7
    expected_reward = min(avg_boxes * 0.5, 2.0) - (avg_moves - avg_boxes) * 0.01
    print(f"Expected reward (close game): {expected_reward:.1f}")
    
    return best_reward, worst_reward, expected_reward

def print_game_state(env):
    """Print a text representation of the game state"""
    print(f"Current Player: {env.current_player + 1}")
    print(f"Scores: Player 1: {env.scores[0]}, Player 2: {env.scores[1]}")
    print(f"Move: {env.move_count}")
    print()
    
    # Print horizontal edges
    for i in range(env.grid_size + 1):
        # Print dots and horizontal edges
        line = ""
        for j in range(env.grid_size):
            line += "●"
            if i < env.grid_size + 1:
                line += "━" if env.horizontal_edges[i, j] else " "
        line += "●"
        print(line)
        
        # Print vertical edges and boxes
        if i < env.grid_size:
            line = ""
            for j in range(env.grid_size + 1):
                if j < env.grid_size + 1:
                    line += "┃" if env.vertical_edges[i, j] else " "
                if j < env.grid_size:
                    if env.box_owners[i, j] == -1:
                        line += " "
                    else:
                        line += str(env.box_owners[i, j] + 1)
            print(line)
    print()

def validate_environment():
    """Test the environment implementation"""
    print("Validating Dots and Boxes Environment...")
    
    from environment import DotsAndBoxesEnv
    
    # Test different grid sizes
    for grid_size in [2, 3, 4]:
        env = DotsAndBoxesEnv(grid_size)
        
        print(f"\nTesting {grid_size}x{grid_size} grid:")
        print(f"  Boxes: {env.n_boxes}")
        print(f"  Edges: {env.n_edges}")
        print(f"  State size: {len(env.get_state())}")
        
        # Test action space
        valid_actions = env.get_valid_actions()
        assert len(valid_actions) == env.n_edges, f"Expected {env.n_edges} valid actions, got {len(valid_actions)}"
        
        # Test a few random moves
        for _ in range(min(5, len(valid_actions))):
            action = np.random.choice(valid_actions)
            state, reward, done = env.step(action)
            assert isinstance(reward, (int, float, np.floating)), "Reward must be numeric"
            assert isinstance(done, bool), "Done must be boolean"
            valid_actions = env.get_valid_actions()
        
        print(f"  ✓ Basic functionality works")
    
    print("\nEnvironment validation complete!")


if __name__ == "__main__":
    validate_environment()
    analyze_reward_structure(3)
