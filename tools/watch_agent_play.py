#!/usr/bin/env python3
"""
Script to watch a trained DQN agent play against a random agent in the GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
from pathlib import Path
from common.gui import DotsAndBoxesGUI
from dqn.dqn_agent import DQNAgent 
from common.agent import RandomAgent
from ppo.ppo_agent import PPOAgent
from common.environment import DotsAndBoxesEnv

class AgentMatchupGUI:
    """GUI for setting up and watching agent matchups"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dots and Boxes - Agent Matchup")
        self.root.geometry("500x400")
        
        self.trained_agent = None
        self.random_agent = None
        self.env = None
        self.game_gui = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main configuration GUI"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame,text="Dots and Boxes Agent Matchup",font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Grid size selection
        ttk.Label(main_frame, text="Grid Size:").grid(row=1,column=0,sticky=tk.W, pady=5)
        self.grid_size_var = tk.StringVar(value="3")
        grid_combo = ttk.Combobox(main_frame, textvariable=self.grid_size_var, values=["2", "3", "4", "5"], state="readonly", width=10)
        grid_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Model file selection
        ttk.Label(main_frame, text="Trained Model:").grid(row=2, column=0, sticky=tk.W, pady=5)
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        self.model_path_var = tk.StringVar()
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=30)
        self.model_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        browse_button = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        browse_button.grid(row=0, column=1, padx=(5, 0))
        
        model_frame.columnconfigure(0, weight=1)
        
        # Player assignment
        ttk.Label(main_frame, text="Player Assignment:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.player_assignment_var = tk.StringVar(value="DQN vs Random")
        assignment_combo = ttk.Combobox(main_frame, textvariable=self.player_assignment_var,values=["DQN vs Random", "Random vs DQN"],state="readonly", width=15)
        assignment_combo.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Auto-detect models
        ttk.Separator(main_frame, orient='horizontal').grid(row=4, column=0, columnspan=2,sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(main_frame, text="Auto-detect Models:").grid(row=5, column=0, sticky=tk.W, pady=5)
        
        self.model_listbox = tk.Listbox(main_frame, height=6)
        self.model_listbox.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.model_listbox.bind('<Double-1>', self.on_model_select)
        
        # Refresh models button
        refresh_button = ttk.Button(main_frame, text="Refresh Models", command=self.refresh_models)
        refresh_button.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(button_frame,text="Start Game Viewer",command=self.start_game_viewer)
        self.start_button.pack(side=tk.LEFT,padx=5)
        
        self.evaluate_button = ttk.Button(button_frame,text="Quick Evaluation",command=self.quick_evaluation)
        self.evaluate_button.pack(side=tk.LEFT,padx=5)
        
        quit_button = ttk.Button(button_frame, text="Quit", command=self.root.quit)
        quit_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select a model and start game viewer")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,relief=tk.SUNKEN, padding="5")
        status_label.grid(row=9,column=0,columnspan=2,sticky=(tk.W, tk.E),pady=(10, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Auto-refresh models on startup
        self.refresh_models()
    
    def browse_model(self):
        """Browse for a model file"""
        filename = filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("PyTorch Model", "*.pth"),("All Files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def refresh_models(self):
        """Refresh the list of available models"""
        self.model_listbox.delete(0, tk.END)
        
        # Look for .pth files in current directory
        current_dir = Path(".")
        model_files = list(current_dir.glob("*.pth"))
        
        if model_files:
            for model_file in sorted(model_files):
                self.model_listbox.insert(tk.END, str(model_file))
        else:
            self.model_listbox.insert(tk.END, "No .pth files found in current directory")
    
    def on_model_select(self, event):
        """Handle model selection from listbox"""
        selection = self.model_listbox.curselection()
        if selection:
            model_name = self.model_listbox.get(selection[0])
            if model_name.endswith('.pth'):
                self.model_path_var.set(model_name)
    
    def load_agents(self):
        """Load the trained agent and create random agent"""
        try:
            grid_size = int(self.grid_size_var.get())
            model_path = self.model_path_var.get()
            
            if not model_path or not os.path.exists(model_path):
                raise ValueError("Please select a valid model file")
            
            # Create environment
            self.env = DotsAndBoxesEnv(grid_size)
            
            # Create and load trained agent
            self.status_var.set("Loading trained model...")
            self.root.update()
            
            self.trained_agent = DQNAgent(self.env)
            self.trained_agent.load_model(model_path)
            self.trained_agent.epsilon = 0.0  # No exploration during play
            
            # Create random agent
            self.random_agent = RandomAgent(self.env)
            
            self.status_var.set(f"Loaded model: {os.path.basename(model_path)}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
            return False
    
    def start_game_viewer(self):
        """Start the game viewer GUI"""
        if not self.load_agents():
            return
        
        try:
            grid_size = int(self.grid_size_var.get())
            
            # Create game GUI
            self.game_gui = DotsAndBoxesGUI(grid_size)
            
            # Set up agents based on player assignment
            assignment = self.player_assignment_var.get()
            if assignment == "DQN vs Random":
                self.game_gui.set_agents(self.trained_agent, self.random_agent)
                player_info = "Player 1: DQN Agent, Player 2: Random Agent"
            else:
                self.game_gui.set_agents(self.random_agent, self.trained_agent)
                player_info = "Player 1: Random Agent, Player 2: DQN Agent"
            
            self.status_var.set("Game viewer opened")
            
            # Add player info to the game window
            info_text = f"{player_info}\nModel: {os.path.basename(self.model_path_var.get())}"
            
            # Create info label in game GUI
            info_label = ttk.Label(self.game_gui.root, text=info_text, 
                                  font=("Arial", 9), foreground="blue")
            info_label.pack(pady=5)
            
            # Start the game GUI (this will block until the window is closed)
            self.game_gui.run()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start game viewer: {str(e)}")
            self.status_var.set("Error starting game viewer")
    
    def quick_evaluation(self):
        """Run a quick evaluation without GUI"""
        if not self.load_agents():
            return
        
        try:
            self.status_var.set("Running evaluation...")
            self.root.update()
            
            # Run evaluation
            assignment = self.player_assignment_var.get()
            if assignment == "DQN vs Random":
                results = self.trained_agent.evaluate(num_games=20, render_first=0)
                player_info = "DQN Agent (Player 1) vs Random Agent (Player 2)"
            else:
                # For Random vs DQN, we need to simulate from random agent's perspective
                # This is more complex, so we'll just switch the interpretation
                results = self.trained_agent.evaluate(num_games=20, render_first=0)
                # Invert the win rate since the trained agent is now player 2
                results['win_rate'] = 1 - results['win_rate'] - results['draw_rate']
                player_info = "Random Agent (Player 1) vs DQN Agent (Player 2)"
            
            # Show results
            result_text = f"""Evaluation Results (20 games):
{player_info}
            
Win Rate: {results['win_rate']:.1%}
Draw Rate: {results['draw_rate']:.1%}
Average Reward: {results['avg_reward']:.2f}
Average Game Length: {results['avg_game_length']:.1f} moves
Reward Std Dev: {results['reward_std']:.2f}"""
            
            messagebox.showinfo("Evaluation Results", result_text)
            self.status_var.set("Evaluation completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
            self.status_var.set("Evaluation failed")
    
    def run(self):
        """Start the main GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Dots and Boxes - Agent Matchup Viewer")
    print("=" * 40)
    
    # Check if we're in the right directory
    required_files = ['gui.py', 'agent.py', 'environment.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please run this script from the directory containing your Dots and Boxes code.")
        return
    
    try:
        app = AgentMatchupGUI()
        app.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()