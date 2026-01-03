"""
Simulation Logger for Belief Propagation Model
Handles all file logging for simulation runs.
"""

from datetime import datetime
from typing import Optional


class SimulationLogger:
    """Full logging of simulation to file."""
    
    def __init__(self, log_dir: str = "logs"):
        import os
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"simulation_{timestamp}.log")
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self._write_header()
        
    def _write_header(self):
        """Write log header."""
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write("AGENTIC BELIEF PROPAGATION SIMULATION LOG\n")
        self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()
        
    def log_config(self, grid_size: int, rounds: int, iterations: int, model: str, seed: Optional[int]):
        """Log configuration."""
        self.log_file.write("CONFIGURATION\n")
        self.log_file.write("-" * 40 + "\n")
        self.log_file.write(f"Grid size: {grid_size}x{grid_size} ({grid_size**2} agents)\n")
        self.log_file.write(f"Conversation rounds: {rounds}\n")
        self.log_file.write(f"Simulation iterations: {iterations}\n")
        self.log_file.write(f"Model: {model}\n")
        if seed is not None:
            self.log_file.write(f"Random seed: {seed}\n")
        else:
            self.log_file.write(f"Random seed: None (non-reproducible)\n")
        self.log_file.write("\n")
        self.log_file.flush()
        
    def log_system_prompts(self, persuader_prompt: str, defender_prompt: str):
        """Log system prompts used for conversations."""
        self.log_file.write("SYSTEM PROMPTS\n")
        self.log_file.write("-" * 40 + "\n")
        self.log_file.write("PERSUADER PROMPT:\n")
        self.log_file.write(f"{persuader_prompt}\n\n")
        self.log_file.write("DEFENDER PROMPT:\n")
        self.log_file.write(f"{defender_prompt}\n\n")
        self.log_file.write("-" * 40 + "\n\n")
        self.log_file.flush()
        
    def log_beliefs(self, beliefs: list[str]):
        """Log available beliefs."""
        self.log_file.write("AVAILABLE BELIEFS\n")
        self.log_file.write("-" * 40 + "\n")
        for i, b in enumerate(beliefs):
            self.log_file.write(f"[{i}] {b}\n")
        self.log_file.write("\n")
        self.log_file.flush()
        
    def log_starting_grid(self, grid: list[list[str]], size: int):
        """Log starting grid state."""
        self.log_file.write("STARTING GRID BELIEFS\n")
        self.log_file.write("-" * 40 + "\n")
        for row in range(size):
            for col in range(size):
                agent_id = row * size + col
                belief = grid[row][col]
                self.log_file.write(f"Agent {agent_id} @ ({row},{col}):\n")
                self.log_file.write(f"  {belief}\n\n")
        self.log_file.write("\n")
        self.log_file.flush()
        
    def log_iteration_start(self, iteration: int, total: int, 
                           persuader_id: int, persuader_pos: tuple,
                           defender_id: int, defender_pos: tuple,
                           persuader_belief: str, defender_belief: str):
        """Log start of an iteration."""
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"ITERATION {iteration}/{total}\n")
        self.log_file.write(f"Time: {datetime.now().isoformat()}\n")
        self.log_file.write("=" * 80 + "\n\n")
        
        self.log_file.write(f"PERSUADER: Agent {persuader_id} @ {persuader_pos}\n")
        self.log_file.write(f"Belief: {persuader_belief}\n\n")
        
        self.log_file.write(f"DEFENDER: Agent {defender_id} @ {defender_pos}\n")
        self.log_file.write(f"Belief: {defender_belief}\n\n")
        
        self.log_file.write("-" * 40 + "\n")
        self.log_file.write("CONVERSATION\n")
        self.log_file.write("-" * 40 + "\n\n")
        self.log_file.flush()
        
    def log_message(self, speaker: str, round_num: int, content: str):
        """Log a conversation message."""
        self.log_file.write(f"[Round {round_num}] {speaker.upper()}:\n")
        self.log_file.write(f"{content}\n\n")
        self.log_file.flush()
        
    def log_decision(self, old_belief: str, new_belief: str, change_type: str):
        """Log the defender's decision."""
        self.log_file.write("-" * 40 + "\n")
        self.log_file.write("DECISION\n")
        self.log_file.write("-" * 40 + "\n")
        self.log_file.write(f"Old belief: {old_belief}\n\n")
        self.log_file.write(f"New belief: {new_belief}\n\n")
        self.log_file.write(f"Change type: {change_type.upper()}\n\n")
        self.log_file.flush()
        
    def log_final_grid(self, grid: list[list[str]], size: int):
        """Log final grid state."""
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write("FINAL GRID BELIEFS\n")
        self.log_file.write("=" * 80 + "\n\n")
        for row in range(size):
            for col in range(size):
                agent_id = row * size + col
                belief = grid[row][col]
                self.log_file.write(f"Agent {agent_id} @ ({row},{col}):\n")
                self.log_file.write(f"  {belief}\n\n")
        self.log_file.flush()
        
    def log_summary(self, total_iterations: int, belief_changes: int):
        """Log final summary."""
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write("SIMULATION SUMMARY\n")
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Completed: {datetime.now().isoformat()}\n")
        self.log_file.write(f"Total iterations: {total_iterations}\n")
        self.log_file.write(f"Belief changes detected: {belief_changes}\n")
        self.log_file.write(f"Change rate: {belief_changes/total_iterations*100:.1f}%\n")
        self.log_file.flush()
        
    def close(self):
        """Close the log file."""
        self.log_file.close()
        
    def get_path(self) -> str:
        """Get the log file path."""
        return self.log_path

