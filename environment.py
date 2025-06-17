import numpy as np

class DotsAndBoxesEnv:
    """Dots and Boxes game environment with improved reward structure"""
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.n_dots = (grid_size + 1) ** 2
        self.n_horizontal_edges = grid_size * (grid_size + 1)
        self.n_vertical_edges = (grid_size + 1) * grid_size
        self.n_edges = self.n_horizontal_edges + self.n_vertical_edges
        self.n_boxes = grid_size * grid_size
        self.reset()
    
    def reset(self):
        # 0: not drawn, 1: drawn
        self.horizontal_edges = np.zeros((self.grid_size + 1, self.grid_size))
        self.vertical_edges = np.zeros((self.grid_size, self.grid_size + 1))
        # -1: no owner, 0: player 1, 1: player 2
        self.box_owners = np.full((self.grid_size, self.grid_size), -1)
        self.current_player = 0
        self.scores = [0, 0]
        self.done = False
        self.move_count = 0
        return self.get_state()
    
    def get_state(self):
        """Enhanced state representation"""
        # Basic game state
        state = np.concatenate([
            self.horizontal_edges.flatten(),
            self.vertical_edges.flatten(),
            (self.box_owners.flatten() + 1) / 2,  # Normalize to [0, 0.5, 1]
            [self.current_player],
            np.array(self.scores) / max(self.n_boxes, 1),  # Normalized scores
            [self.move_count / self.n_edges],  # Game progress
        ])
        
        # Add tactical features: count boxes with 1, 2, 3 edges completed
        tactical_features = self._get_tactical_features()
        
        return np.concatenate([state, tactical_features]).astype(np.float32)
    
    def _get_tactical_features(self):
        """Get tactical information about box completion states"""
        box_edge_counts = [0, 0, 0, 0]  # 0, 1, 2, 3 edges completed
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.box_owners[i, j] == -1:  # Unowned box
                    edges_count = int(
                        self.horizontal_edges[i, j] +      # top
                        self.horizontal_edges[i + 1, j] +  # bottom
                        self.vertical_edges[i, j] +        # left
                        self.vertical_edges[i, j + 1]      # right
                    )
                    box_edge_counts[edges_count] += 1
        
        # Normalize by total boxes
        return np.array(box_edge_counts, dtype=np.float32) / max(self.n_boxes, 1)
    
    def get_valid_actions(self):
        """Get list of valid actions (undrawn edges)"""
        valid = []
        # Horizontal edges
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size):
                if self.horizontal_edges[i, j] == 0:
                    valid.append(i * self.grid_size + j)
        # Vertical edges
        offset = self.n_horizontal_edges
        for i in range(self.grid_size):
            for j in range(self.grid_size + 1):
                if self.vertical_edges[i, j] == 0:
                    valid.append(offset + i * (self.grid_size + 1) + j)
        return valid
    
    def step(self, action):
        """Execute an action with bounded reward structure"""
        if action < self.n_horizontal_edges:
            # Horizontal edge
            i = action // self.grid_size
            j = action % self.grid_size
            if self.horizontal_edges[i, j] == 1:
                return self.get_state(), -1.0, True  # Reduced invalid move penalty
            self.horizontal_edges[i, j] = 1
        else:
            # Vertical edge
            action -= self.n_horizontal_edges
            i = action // (self.grid_size + 1)
            j = action % (self.grid_size + 1)
            if self.vertical_edges[i, j] == 1:
                return self.get_state(), -1.0, True  # Reduced invalid move penalty
            self.vertical_edges[i, j] = 1
        
        self.move_count += 1
        
        # Check for completed boxes and calculate rewards
        boxes_completed = 0
        dangerous_moves = 0  # Count of 3-edge boxes created
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.box_owners[i, j] == -1:  # Unowned box
                    # Check if all 4 edges are drawn
                    edges = [
                        self.horizontal_edges[i, j],      # top
                        self.horizontal_edges[i + 1, j],  # bottom
                        self.vertical_edges[i, j],        # left
                        self.vertical_edges[i, j + 1]     # right
                    ]
                    edges_count = int(sum(edges))
                    
                    if edges_count == 4:
                        self.box_owners[i, j] = self.current_player
                        self.scores[self.current_player] += 1
                        boxes_completed += 1
                    elif edges_count == 3:
                        dangerous_moves += 1
        
        # Calculate bounded reward
        reward = 0
        
        # Primary reward: boxes completed (bounded)
        reward += min(boxes_completed * 0.5, 2.0)  # Cap at 2.0
        
        # Penalty for creating 3-edge boxes (bounded)
        if boxes_completed == 0 and dangerous_moves > 0:
            reward -= min(dangerous_moves * 0.1, 0.5)  # Cap penalty at -0.5
        
        # Small negative reward for not completing boxes
        if boxes_completed == 0:
            reward -= 0.01
            self.current_player = 1 - self.current_player
        
        # Check if game is done
        if len(self.get_valid_actions()) == 0:
            self.done = True
            # Final reward based on game outcome (bounded)
            score_diff = self.scores[0] - self.scores[1]
            if score_diff > 0:
                final_reward = 1.0 if self.current_player == 0 else -1.0
            elif score_diff < 0:
                final_reward = -1.0 if self.current_player == 0 else 1.0
            else:
                final_reward = 0  # Draw
            
            # Add small proportional reward based on margin (bounded)
            margin_bonus = np.tanh(abs(score_diff) / self.n_boxes) * 0.5
            if (score_diff > 0 and self.current_player == 0) or \
               (score_diff < 0 and self.current_player == 1):
                final_reward += margin_bonus
            else:
                final_reward -= margin_bonus
            
            reward += final_reward
        
        # Ensure reward is bounded
        reward = np.clip(reward, -3.0, 3.0)
        
        return self.get_state(), reward, self.done
    
    def action_to_coords(self, action):
        """Convert action number to edge coordinates and type"""
        if action < self.n_horizontal_edges:
            # Horizontal edge
            i = action // self.grid_size
            j = action % self.grid_size
            return 'horizontal', i, j
        else:
            # Vertical edge
            action -= self.n_horizontal_edges
            i = action // (self.grid_size + 1)
            j = action % (self.grid_size + 1)
            return 'vertical', i, j
    
    def copy(self):
        """Create a copy of the current environment state"""
        new_env = DotsAndBoxesEnv(self.grid_size)
        new_env.horizontal_edges = self.horizontal_edges.copy()
        new_env.vertical_edges = self.vertical_edges.copy()
        new_env.box_owners = self.box_owners.copy()
        new_env.current_player = self.current_player
        new_env.scores = self.scores.copy()
        new_env.done = self.done
        new_env.move_count = self.move_count
        return new_env