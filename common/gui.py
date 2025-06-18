import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
from common.environment import DotsAndBoxesEnv

class DotsAndBoxesGUI:
    """GUI for visualizing Dots and Boxes games"""
    
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.env = DotsAndBoxesEnv(grid_size)
        self.cell_size = 80
        self.dot_radius = 4
        self.edge_width = 4
        
        # Colors
        self.colors = {
            'background': '#f0f0f0',
            'dot': '#333333',
            'edge_undrawn': '#cccccc',
            'edge_drawn': '#333333',
            'player1_box': '#ff6b6b',
            'player2_box': '#4ecdc4',
            'highlight': '#ffd93d'
        }
        
        self.setup_gui()
        self.agents = [None, None]  # Will be set by the game runner
        self.game_running = False
        self.game_speed = 500  # milliseconds between moves
        
    def setup_gui(self):
        """Initialize the GUI"""
        self.root = tk.Tk()
        self.root.title(f"Dots and Boxes ({self.grid_size}x{self.grid_size})")
        self.root.configure(bg=self.colors['background'])
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=(0, 10))
        
        # Game info
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.score_label = ttk.Label(info_frame, text="Player 1: 0  |  Player 2: 0", font=("Arial", 12, "bold"))
        self.score_label.pack()
        
        self.turn_label = ttk.Label(info_frame, text="Player 1's Turn", font=("Arial", 10))
        self.turn_label.pack()
        
        self.move_label = ttk.Label(info_frame, text="Move: 0", font=("Arial", 10))
        self.move_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT)
        
        self.start_button = ttk.Button(button_frame, text="Start Game", command=self.start_game)
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_game, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=2)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_game)
        self.reset_button.pack(side=tk.LEFT, padx=2)
        
        self.step_button = ttk.Button(button_frame, text="Next Move", command=self.step_game, state=tk.DISABLED)
        self.step_button.pack(side=tk.LEFT, padx=2)
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(speed_frame, text="Speed:").pack()
        self.speed_var = tk.StringVar(value="Normal")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, 
                                  values=["Very Slow", "Slow", "Normal", "Fast", "Very Fast"],
                                  state="readonly", width=10)
        speed_combo.pack()
        speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)
        
        # Canvas for the game board
        canvas_width = (self.grid_size + 1) * self.cell_size
        canvas_height = (self.grid_size + 1) * self.cell_size
        
        self.canvas = tk.Canvas(main_frame, width=canvas_width, height=canvas_height, 
                               bg='white', highlightthickness=1, highlightbackground='gray')
        self.canvas.pack()
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready to start", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        self.draw_board()
        
    def on_speed_change(self, event=None):
        """Handle speed control changes"""
        speed_map = {
            "Very Slow": 2000,
            "Slow": 1000,
            "Normal": 500,
            "Fast": 200,
            "Very Fast": 50
        }
        self.game_speed = speed_map.get(self.speed_var.get(), 500)
    
    def draw_board(self):
        """Draw the current game board"""
        self.canvas.delete("all")
        
        # Draw dots
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size + 1):
                x = j * self.cell_size + self.cell_size // 2
                y = i * self.cell_size + self.cell_size // 2
                self.canvas.create_oval(x - self.dot_radius, y - self.dot_radius,
                                      x + self.dot_radius, y + self.dot_radius,
                                      fill=self.colors['dot'], outline=self.colors['dot'])
        
        # Draw horizontal edges
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size):
                x1 = j * self.cell_size + self.cell_size // 2 + self.dot_radius
                x2 = (j + 1) * self.cell_size + self.cell_size // 2 - self.dot_radius
                y = i * self.cell_size + self.cell_size // 2
                
                color = self.colors['edge_drawn'] if self.env.horizontal_edges[i, j] else self.colors['edge_undrawn']
                width = self.edge_width if self.env.horizontal_edges[i, j] else 2
                
                self.canvas.create_line(x1, y, x2, y, fill=color, width=width,
                                      tags=f"hedge_{i}_{j}")
        
        # Draw vertical edges
        for i in range(self.grid_size):
            for j in range(self.grid_size + 1):
                x = j * self.cell_size + self.cell_size // 2
                y1 = i * self.cell_size + self.cell_size // 2 + self.dot_radius
                y2 = (i + 1) * self.cell_size + self.cell_size // 2 - self.dot_radius
                
                color = self.colors['edge_drawn'] if self.env.vertical_edges[i, j] else self.colors['edge_undrawn']
                width = self.edge_width if self.env.vertical_edges[i, j] else 2
                
                self.canvas.create_line(x, y1, x, y2, fill=color, width=width,
                                      tags=f"vedge_{i}_{j}")
        
        # Draw boxes
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.env.box_owners[i, j] != -1:
                    x1 = j * self.cell_size + self.cell_size // 2 + self.dot_radius
                    y1 = i * self.cell_size + self.cell_size // 2 + self.dot_radius
                    x2 = (j + 1) * self.cell_size + self.cell_size // 2 - self.dot_radius
                    y2 = (i + 1) * self.cell_size + self.cell_size // 2 - self.dot_radius
                    
                    color = self.colors['player1_box'] if self.env.box_owners[i, j] == 0 else self.colors['player2_box']
                    
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='', stipple='gray25', tags=f"box_{i}_{j}")
                    
                    # Add player number in the center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    player_num = self.env.box_owners[i, j] + 1
                    self.canvas.create_text(center_x, center_y, text=str(player_num), font=("Arial", 16, "bold"), fill='white')
    
    def highlight_last_move(self, action):
        """Highlight the last move made"""
        # Remove previous highlights
        self.canvas.delete("highlight")
        
        if action is None:
            return
            
        edge_type, i, j = self.env.action_to_coords(action)
        
        if edge_type == 'horizontal':
            x1 = j * self.cell_size + self.cell_size // 2 + self.dot_radius
            x2 = (j + 1) * self.cell_size + self.cell_size // 2 - self.dot_radius
            y = i * self.cell_size + self.cell_size // 2
            
            self.canvas.create_line(x1, y, x2, y, fill=self.colors['highlight'], width=self.edge_width + 2, tags="highlight")
        else:  # vertical
            x = j * self.cell_size + self.cell_size // 2
            y1 = i * self.cell_size + self.cell_size // 2 + self.dot_radius
            y2 = (i + 1) * self.cell_size + self.cell_size // 2 - self.dot_radius
            
            self.canvas.create_line(x, y1, x, y2, fill=self.colors['highlight'], width=self.edge_width + 2, tags="highlight")
    
    def update_display(self, last_action=None):
        """Update the game display"""
        self.draw_board()
        if last_action is not None:
            self.highlight_last_move(last_action)
        
        # Update labels
        self.score_label.config(text=f"Player 1: {self.env.scores[0]}  |  Player 2: {self.env.scores[1]}")
        
        if not self.env.done:
            current_player = self.env.current_player + 1
            self.turn_label.config(text=f"Player {current_player}'s Turn")
        else:
            if self.env.scores[0] > self.env.scores[1]:
                self.turn_label.config(text="Player 1 Wins!")
            elif self.env.scores[1] > self.env.scores[0]:
                self.turn_label.config(text="Player 2 Wins!")
            else:
                self.turn_label.config(text="It's a Draw!")
        
        self.move_label.config(text=f"Move: {self.env.move_count}")
        self.root.update()
    
    def set_agents(self, agent1, agent2):
        """Set the agents for player 1 and player 2"""
        self.agents = [agent1, agent2]
    
    def start_game(self):
        """Start the game"""
        if not self.agents[0] or not self.agents[1]:
            messagebox.showerror("Error", "Please set both agents before starting the game")
            return
        
        self.game_running = True
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.step_button.config(state=tk.DISABLED)
        
        self.status_label.config(text="Game running...")
        
        # Start game in a separate thread
        self.game_thread = threading.Thread(target=self.run_game)
        self.game_thread.daemon = True
        self.game_thread.start()
    
    def pause_game(self):
        """Pause/resume the game"""
        if self.game_running:
            self.game_running = False
            self.pause_button.config(text="Resume")
            self.step_button.config(state=tk.NORMAL)
            self.status_label.config(text="Game paused")
        else:
            self.game_running = True
            self.pause_button.config(text="Pause")
            self.step_button.config(state=tk.DISABLED)
            self.status_label.config(text="Game running...")
    
    def reset_game(self):
        """Reset the game"""
        self.game_running = False
        self.env.reset()
        self.update_display()
        
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")
        self.step_button.config(state=tk.DISABLED)
        self.status_label.config(text="Game reset")
    
    def step_game(self):
        """Execute one move when paused"""
        if not self.env.done and not self.game_running:
            self.make_move()
    
    def run_game(self):
        """Main game loop"""
        last_action = None
        
        while not self.env.done and self.game_running:
            last_action = self.make_move()
            
            # Wait for the specified time or until game is paused
            start_time = time.time()
            while time.time() - start_time < self.game_speed / 1000.0:
                if not self.game_running:
                    break
                time.sleep(0.01)
        
        # Game finished
        if self.env.done:
            self.root.after(0, self.game_finished)
    
    def make_move(self):
        """Make a single move"""
        if self.env.done:
            return None
        
        current_player = self.env.current_player
        agent = self.agents[current_player]
        state = self.env.get_state()
        valid_actions = self.env.get_valid_actions()
        
        # Get action from agent
        action = agent.act(state, valid_actions, training=False)
        
        # Execute action
        next_state, reward, done = self.env.step(action)
        
        # Update display in main thread
        self.root.after(0, self.update_display, action)
        
        return action
    
    def game_finished(self):
        """Handle game completion"""
        self.game_running = False
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")
        self.step_button.config(state=tk.DISABLED)
        
        winner_text = ""
        if self.env.scores[0] > self.env.scores[1]:
            winner_text = "Player 1 wins!"
        elif self.env.scores[1] > self.env.scores[0]:
            winner_text = "Player 2 wins!"
        else:
            winner_text = "It's a draw!"
        
        self.status_label.config(text=f"Game finished - {winner_text}")
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

def demo_gui():
    """Demo function to show the GUI with random agents"""
    from common.agent import RandomAgent
    
    grid_size = 3
    gui = DotsAndBoxesGUI(grid_size)
    
    # Create random agents for demo
    env = DotsAndBoxesEnv(grid_size)
    agent1 = RandomAgent(env)
    agent2 = RandomAgent(env)
    
    gui.set_agents(agent1, agent2)
    gui.run()

if __name__ == "__main__":
    demo_gui()