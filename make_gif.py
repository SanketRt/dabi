#!/usr/bin/env python3
"""
GIF Recorder for Dots and Boxes Game
Captures the game window and creates animated GIFs
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
from datetime import datetime
from pathlib import Path
import numpy as np

# Try to import required packages
try:
    import PIL
    from PIL import Image, ImageTk, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import your modules
from gui import DotsAndBoxesGUI
from agent import DQNAgent, RandomAgent
from environment import DotsAndBoxesEnv

class GIFRecorderGUI(DotsAndBoxesGUI):
    """Extended DotsAndBoxesGUI with GIF recording capabilities"""
    
    def __init__(self, grid_size=3):
        super().__init__(grid_size)
        self.recording = False
        self.frames = []
        self.recording_thread = None
        self.output_path = "game_recording.gif"
        self.fps = 2  # Frames per second for GIF
        self.frame_duration = 500  # milliseconds between captures
        
        self.add_recording_controls()
    
    def add_recording_controls(self):
        """Add recording controls to the existing GUI"""
        # Create recording control frame
        recording_frame = ttk.LabelFrame(self.root, text="GIF Recording", padding="5")
        recording_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # Recording controls
        controls_frame = ttk.Frame(recording_frame)
        controls_frame.pack(fill=tk.X, pady=2)
        
        self.record_button = ttk.Button(controls_frame, text="Start Recording", 
                                       command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=2)
        
        self.save_button = ttk.Button(controls_frame, text="Save GIF", 
                                     command=self.save_gif, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=2)
        
        self.clear_button = ttk.Button(controls_frame, text="Clear Frames", 
                                      command=self.clear_frames)
        self.clear_button.pack(side=tk.LEFT, padx=2)
        
        # Settings frame
        settings_frame = ttk.Frame(recording_frame)
        settings_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(settings_frame, text="Capture Rate:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="2")
        fps_combo = ttk.Combobox(settings_frame, textvariable=self.fps_var,
                                values=["1", "2", "3", "4", "5"], width=5, state="readonly")
        fps_combo.pack(side=tk.LEFT, padx=2)
        fps_combo.bind("<<ComboboxSelected>>", self.on_fps_change)
        
        ttk.Label(settings_frame, text="fps").pack(side=tk.LEFT, padx=(2, 10))
        
        # Output path
        path_frame = ttk.Frame(recording_frame)
        path_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(path_frame, text="Output:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value=self.output_path)
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var, width=30)
        path_entry.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        browse_button = ttk.Button(path_frame, text="Browse", command=self.browse_output)
        browse_button.pack(side=tk.LEFT, padx=2)
        
        # Status
        self.recording_status = ttk.Label(recording_frame, text="Ready to record", 
                                         foreground="green")
        self.recording_status.pack(pady=2)
        
        # Frame count
        self.frame_count_label = ttk.Label(recording_frame, text="Frames: 0")
        self.frame_count_label.pack()
    
    def on_fps_change(self, event=None):
        """Handle FPS change"""
        self.fps = int(self.fps_var.get())
        self.frame_duration = max(100, 1000 // self.fps)  # At least 100ms between frames
    
    def browse_output(self):
        """Browse for output file"""
        filename = filedialog.asksaveasfilename(
            title="Save GIF As",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("All files", "*.*")]
        )
        if filename:
            self.path_var.set(filename)
            self.output_path = filename
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not PIL_AVAILABLE:
            messagebox.showerror("Error", "PIL (Pillow) is required for GIF recording.\n"
                               "Install with: pip install Pillow")
            return
        
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording frames"""
        self.recording = True
        self.frames = []
        self.record_button.config(text="Stop Recording", style="Accent.TButton")
        self.save_button.config(state=tk.DISABLED)
        self.recording_status.config(text="Recording...", foreground="red")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording frames"""
        self.recording = False
        self.record_button.config(text="Start Recording", style="")
        
        if self.frames:
            self.save_button.config(state=tk.NORMAL)
            self.recording_status.config(text=f"Recording stopped - {len(self.frames)} frames captured", 
                                       foreground="blue")
        else:
            self.recording_status.config(text="No frames captured", foreground="orange")
    
    def recording_loop(self):
        """Main recording loop"""
        while self.recording:
            try:
                # Capture the canvas area
                frame = self.capture_canvas()
                if frame:
                    self.frames.append(frame)
                    # Update frame count in main thread
                    self.root.after(0, self.update_frame_count)
                
                time.sleep(self.frame_duration / 1000.0)
                
            except Exception as e:
                print(f"Recording error: {e}")
                break
        
        # Update UI when recording stops
        self.root.after(0, self.stop_recording)
    
    def capture_canvas(self):
        """Capture the current canvas as an image"""
        try:
            # Get canvas coordinates relative to screen
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            # Capture screenshot of canvas area
            bbox = (x, y, x + width, y + height)
            screenshot = ImageGrab.grab(bbox)
            
            return screenshot
            
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    def update_frame_count(self):
        """Update frame count display"""
        self.frame_count_label.config(text=f"Frames: {len(self.frames)}")
    
    def clear_frames(self):
        """Clear captured frames"""
        self.frames = []
        self.update_frame_count()
        self.save_button.config(state=tk.DISABLED)
        self.recording_status.config(text="Frames cleared", foreground="green")
    
    def save_gif(self):
        """Save captured frames as GIF"""
        if not self.frames:
            messagebox.showwarning("Warning", "No frames to save!")
            return
        
        try:
            output_path = self.path_var.get()
            
            # Calculate duration between frames (in milliseconds)
            duration = 1000 // self.fps
            
            self.recording_status.config(text="Saving GIF...", foreground="blue")
            self.root.update()
            
            # Save as GIF
            self.frames[0].save(
                output_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0,  # Loop forever
                optimize=True
            )
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            self.recording_status.config(
                text=f"GIF saved: {os.path.basename(output_path)} ({file_size:.1f} MB)", 
                foreground="green"
            )
            
            # Ask if user wants to open the file
            if messagebox.askyesno("Success", f"GIF saved successfully!\n\nOpen {os.path.basename(output_path)}?"):
                self.open_file(output_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save GIF: {str(e)}")
            self.recording_status.config(text="Save failed", foreground="red")
    
    def open_file(self, filepath):
        """Open file with default system application"""
        import subprocess
        import sys
        
        try:
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.call(['open', filepath])
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(filepath)
            else:  # Linux and others
                subprocess.call(['xdg-open', filepath])
        except Exception as e:
            print(f"Could not open file: {e}")

class GIFRecorderApp:
    """Main application for GIF recording"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dots and Boxes - GIF Recorder Setup")
        self.root.geometry("600x500")
        
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
        title_label = ttk.Label(main_frame, text="Dots and Boxes - GIF Recorder", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Check dependencies - now passing main_frame as parameter
        self.check_dependencies(main_frame)
        
        # Grid size selection
        ttk.Label(main_frame, text="Grid Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.grid_size_var = tk.StringVar(value="3")
        grid_combo = ttk.Combobox(main_frame, textvariable=self.grid_size_var, 
                                 values=["2", "3", "4", "5"], state="readonly", width=10)
        grid_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Model file selection
        ttk.Label(main_frame, text="Trained Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        
        self.model_path_var = tk.StringVar()
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=40)
        self.model_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        browse_button = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        browse_button.grid(row=0, column=1, padx=(5, 0))
        
        model_frame.columnconfigure(0, weight=1)
        
        # Player assignment
        ttk.Label(main_frame, text="Player Assignment:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.player_assignment_var = tk.StringVar(value="DQN vs Random")
        assignment_combo = ttk.Combobox(main_frame, textvariable=self.player_assignment_var,
                                       values=["DQN vs Random", "Random vs DQN"], 
                                       state="readonly", width=15)
        assignment_combo.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Game speed (affects recording)
        ttk.Label(main_frame, text="Game Speed:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.speed_var = tk.StringVar(value="Slow")
        speed_combo = ttk.Combobox(main_frame, textvariable=self.speed_var,
                                  values=["Very Slow", "Slow", "Normal", "Fast"], 
                                  state="readonly", width=15)
        speed_combo.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Auto-detect models
        ttk.Separator(main_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, 
                                                           sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(main_frame, text="Available Models:").grid(row=7, column=0, sticky=tk.W, pady=5)
        
        self.model_listbox = tk.Listbox(main_frame, height=8)
        self.model_listbox.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.model_listbox.bind('<Double-1>', self.on_model_select)
        
        # Refresh models button
        refresh_button = ttk.Button(main_frame, text="Refresh Models", command=self.refresh_models)
        refresh_button.grid(row=9, column=0, columnspan=2, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Launch Game Recorder", 
                                      command=self.start_game_recorder, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        quit_button = ttk.Button(button_frame, text="Quit", command=self.root.quit)
        quit_button.pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = """Instructions:
1. Select your trained model and game settings
2. Click 'Launch Game Recorder' to open the game window
3. In the game window, use the recording controls:
   • Click 'Start Recording' before starting the game
   • Play the game (Start Game button)
   • Click 'Stop Recording' when done
   • Click 'Save GIF' to create your animated GIF

Tips:
• Use slower game speeds for better GIF quality
• Keep recordings short (< 100 frames) for smaller file sizes
• The GIF will loop automatically when played"""
        
        instructions_text = tk.Text(main_frame, height=12, wrap=tk.WORD, font=("Arial", 9))
        instructions_text.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        instructions_text.insert(tk.END, instructions)
        instructions_text.config(state=tk.DISABLED, bg=self.root.cget('bg'))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Auto-refresh models on startup
        self.refresh_models()
    
    def check_dependencies(self, parent_frame):
        """Check and display dependency status"""
        dep_frame = ttk.LabelFrame(parent_frame, text="Dependencies", padding="5")
        dep_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # PIL check
        pil_status = "✓ Installed" if PIL_AVAILABLE else "✗ Missing"
        pil_color = "green" if PIL_AVAILABLE else "red"
        ttk.Label(dep_frame, text=f"PIL (Pillow): {pil_status}", foreground=pil_color).pack(anchor=tk.W)
        
        if not PIL_AVAILABLE:
            ttk.Label(dep_frame, text="Install with: pip install Pillow",font=("Arial", 8), foreground="blue").pack(anchor=tk.W, padx=20)
    
    def browse_model(self):
        """Browse for a model file"""
        filename = filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("PyTorch Model","*.pth"), ("All Files", "*.*")]
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
    
    def start_game_recorder(self):
        """Start the game recorder"""
        if not PIL_AVAILABLE:
            messagebox.showerror("Error", "PIL (Pillow) is required for GIF recording.\n" "Install with: pip install Pillow")
            return
        
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        try:
            grid_size = int(self.grid_size_var.get())
            
            # Create environment and agents
            env = DotsAndBoxesEnv(grid_size)
            
            # Load trained agent
            trained_agent = DQNAgent(env)
            trained_agent.load_model(model_path)
            trained_agent.epsilon = 0.0  # No exploration
            
            # Create random agent
            random_agent = RandomAgent(env)
            
            # Create GIF recorder GUI
            self.game_gui = GIFRecorderGUI(grid_size)
            
            # Set game speed
            speed_map = {
                "Very Slow": "Very Slow",
                "Slow": "Slow", 
                "Normal": "Normal",
                "Fast": "Fast"
            }
            self.game_gui.speed_var.set(speed_map[self.speed_var.get()])
            self.game_gui.on_speed_change()
            
            # Set agents
            assignment = self.player_assignment_var.get()
            if assignment == "DQN vs Random":
                self.game_gui.set_agents(trained_agent, random_agent)
                player_info = "Player 1: DQN Agent, Player 2: Random Agent"
            else:
                self.game_gui.set_agents(random_agent, trained_agent)
                player_info = "Player 1: Random Agent, Player 2: DQN Agent"
            
            # Add info to game window
            info_text = f"{player_info}\nModel: {os.path.basename(model_path)}"
            info_label = ttk.Label(self.game_gui.root, text=info_text, 
                                  font=("Arial", 9), foreground="blue")
            info_label.pack(after=self.game_gui.canvas, pady=5)
            
            # Set default output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"dotsandboxes_{grid_size}x{grid_size}_{assignment.replace(' ', '_').lower()}_{timestamp}.gif"
            self.game_gui.path_var.set(default_name)
            self.game_gui.output_path = default_name
            
            print(f"Game recorder launched!")
            print(f"Model: {os.path.basename(model_path)}")
            print(f"Setup: {player_info}")
            
            # Run the game GUI
            self.game_gui.run()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start game recorder: {str(e)}")
    
    def run(self):
        """Start the main GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Dots and Boxes - GIF Recorder")
    print("=" * 40)
    
    # Check if we're in the right directory
    required_files = ['gui.py', 'agent.py', 'environment.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please run this script from the directory containing your Dots and Boxes code.")
        return
    
    if not PIL_AVAILABLE:
        print("Warning: PIL (Pillow) is not installed.")
        print("To create GIFs, install it with: pip install Pillow")
        print()
    
    try:
        app = GIFRecorderApp()
        app.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()