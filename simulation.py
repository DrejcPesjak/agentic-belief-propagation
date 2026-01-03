"""
Agentic Axelrod-like Belief Propagation Model
Uses Ollama with Gemma3 for LLM-based belief dissemination on a 2D grid.
"""

import json
import random
from typing import Optional, Callable
import ollama

from chat_gui import run_conversation_with_gui
from simulation_logger import SimulationLogger


# === Configuration ===
GRID_SIZE = 3  # 3x3 grid = 9 agents
CONVERSATION_ROUNDS = 5  # 5 exchanges per interaction
SIMULATION_ITERATIONS = 40  # total interactions to simulate
MODEL = "gemma3"
BELIEFS_FILE = "money-philosophy.json"
GUI_CLOSE_DELAY = 5.0  # seconds to show final decision before closing
RANDOM_SEED = 42  # Seed for reproducible starting grid distribution (None for random)


def load_beliefs(filepath: str) -> list[str]:
    """Load beliefs from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["beliefs"]


def initialize_grid(beliefs: list[str], size: int) -> list[list[str]]:
    """Initialize MxM grid with randomly sampled beliefs (with repetition).
    
    Note: Random seed should be set before calling this function for reproducibility.
    """
    grid = []
    for _ in range(size):
        row = [random.choice(beliefs) for _ in range(size)]
        grid.append(row)
    return grid


def get_neighbors(row: int, col: int, size: int) -> list[tuple[int, int]]:
    """Get valid 4-connected neighbors (up, down, left, right) for a position."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return neighbors


def pos_to_id(row: int, col: int, size: int) -> int:
    """Convert grid position to agent ID."""
    return row * size + col


def id_to_pos(agent_id: int, size: int) -> tuple[int, int]:
    """Convert agent ID to grid position."""
    return agent_id // size, agent_id % size


def build_system_prompt(belief: str, role: str) -> str:
    """Build simple system prompt for an agent."""
    if role == "persuader":
        return f'This is your belief: "{belief}". You are the persuader. Try to convince the other person of your belief. Be concise.'
        # return f'This is your belief: "{belief}". You are the aggressive persuader. At any price try to convince the other person of your belief. Be concise.'
    else:  # defender
        return f'This is your belief: "{belief}". You are the defender. Listen to the other person and critically analyze their arguments based on your belief. Be concise.'


def create_conversation_runner(
    persuader_belief: str,
    defender_belief: str,
    rounds: int,
    logger: Optional[SimulationLogger] = None
) -> Callable:
    """
    Create a conversation function that can be passed to the GUI.
    
    Returns a function that takes on_message callback and returns (history, new_belief).
    """
    def run_conversation(on_message: Callable[[str, int, str], None]) -> tuple[list[dict], str]:
        """
        Run a conversation between persuader and defender.
        """
        persuader_system = build_system_prompt(persuader_belief, "persuader")
        defender_system = build_system_prompt(defender_belief, "defender")
        
        conversation_history = []
        
        # Persuader starts
        persuader_messages = [{"role": "system", "content": persuader_system}]
        defender_messages = [{"role": "system", "content": defender_system}]
        
        # Initial persuader message
        persuader_messages.append({
            "role": "user", 
            "content": "Start the conversation by presenting your belief and why the other person should adopt it."
        })
        
        for round_num in range(rounds):
            # Persuader speaks
            persuader_response = ollama.chat(
                model=MODEL,
                messages=persuader_messages
            )
            persuader_text = persuader_response["message"]["content"]
            conversation_history.append({
                "round": round_num + 1,
                "speaker": "persuader",
                "content": persuader_text
            })
            
            # Notify GUI
            on_message("persuader", round_num + 1, persuader_text)
            
            # Log full message
            if logger:
                logger.log_message("persuader", round_num + 1, persuader_text)
            
            # Add to persuader's history
            persuader_messages.append({"role": "assistant", "content": persuader_text})
            
            # Add to defender's history as user message
            defender_messages.append({"role": "user", "content": persuader_text})
            
            # Defender responds
            defender_response = ollama.chat(
                model=MODEL,
                messages=defender_messages
            )
            defender_text = defender_response["message"]["content"]
            conversation_history.append({
                "round": round_num + 1,
                "speaker": "defender",
                "content": defender_text
            })
            
            # Notify GUI
            on_message("defender", round_num + 1, defender_text)
            
            # Log full message
            if logger:
                logger.log_message("defender", round_num + 1, defender_text)
            
            # Add to defender's history
            defender_messages.append({"role": "assistant", "content": defender_text})
            
            # Add to persuader's history as user message for next round
            persuader_messages.append({"role": "user", "content": defender_text})
        
        # Final decision from defender
        defender_messages.append({
            "role": "user",
            "content": (
                "Based on this conversation, decide if you want to update your belief or keep it. "
                "Output ONLY your final belief as a single statement starting with 'I'. Nothing else."
                "Keep it short and core."
            )
        })
        
        final_response = ollama.chat(
            model=MODEL,
            messages=defender_messages
        )
        new_belief = final_response["message"]["content"].strip()
        
        return conversation_history, new_belief
    
    return run_conversation


def print_grid(grid: list[list[str]], beliefs: list[str], size: int) -> None:
    """Print the grid showing belief indices for readability."""
    print("\nCurrent Grid (belief indices):")
    print("-" * (size * 4 + 1))
    for row in grid:
        row_display = []
        for belief in row:
            # Find index or show * for custom beliefs
            try:
                idx = beliefs.index(belief)
                row_display.append(f" {idx} ")
            except ValueError:
                row_display.append(" * ")  # Modified belief
        print("|" + "|".join(row_display) + "|")
        print("-" * (size * 4 + 1))


def print_grid_beliefs(grid: list[list[str]], size: int, title: str) -> None:
    """Print full beliefs for all agents in grid."""
    print(f"\n{title}:")
    for row in range(size):
        for col in range(size):
            agent_id = row * size + col
            belief = grid[row][col]
            print(f"  Agent {agent_id}: {belief}")


def run_simulation():
    """Main simulation loop."""
    # Initialize logger
    logger = SimulationLogger()
    
    print("=" * 60)
    print("AGENTIC BELIEF PROPAGATION SIMULATION")
    print("=" * 60)
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE**2} agents)")
    print(f"Conversation rounds: {CONVERSATION_ROUNDS}")
    print(f"Simulation iterations: {SIMULATION_ITERATIONS}")
    print(f"Model: {MODEL}")
    print(f"Log file: {logger.get_path()}")
    print("=" * 60)
    
    # Set random seed if specified
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"Random seed: {RANDOM_SEED} (reproducible)")
    else:
        print("Random seed: None (non-reproducible)")
    
    # Log config
    logger.log_config(GRID_SIZE, CONVERSATION_ROUNDS, SIMULATION_ITERATIONS, MODEL, RANDOM_SEED)
    
    # Load beliefs
    beliefs = load_beliefs(BELIEFS_FILE)
    print(f"\nLoaded {len(beliefs)} beliefs:")
    for i, b in enumerate(beliefs):
        print(f"  [{i}] {b[:60]}...")
    
    logger.log_beliefs(beliefs)
    
    # Log system prompts (using example beliefs)
    example_persuader_prompt = build_system_prompt("...", "persuader")
    example_defender_prompt = build_system_prompt("...", "defender")
    logger.log_system_prompts(example_persuader_prompt, example_defender_prompt)
    
    # Initialize grid (seed already set above if RANDOM_SEED is not None)
    grid = initialize_grid(beliefs, GRID_SIZE)
    
    # Print and log starting grid (FULL beliefs)
    print_grid_beliefs(grid, GRID_SIZE, "Starting Grid Beliefs")
    print_grid(grid, beliefs, GRID_SIZE)
    logger.log_starting_grid(grid, GRID_SIZE)
    
    # Track changes
    belief_changes = 0
    
    # Run simulation iterations
    for iteration in range(1, SIMULATION_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{SIMULATION_ITERATIONS}")
        print("=" * 60)
        
        # Pick random agent
        agent_row = random.randint(0, GRID_SIZE - 1)
        agent_col = random.randint(0, GRID_SIZE - 1)
        agent_id = pos_to_id(agent_row, agent_col, GRID_SIZE)
        
        # Pick random neighbor
        neighbors = get_neighbors(agent_row, agent_col, GRID_SIZE)
        neighbor_row, neighbor_col = random.choice(neighbors)
        neighbor_id = pos_to_id(neighbor_row, neighbor_col, GRID_SIZE)
        
        # 50/50 who is persuader vs defender
        if random.random() < 0.5:
            persuader_pos = (agent_row, agent_col)
            defender_pos = (neighbor_row, neighbor_col)
            persuader_id, defender_id = agent_id, neighbor_id
        else:
            persuader_pos = (neighbor_row, neighbor_col)
            defender_pos = (agent_row, agent_col)
            persuader_id, defender_id = neighbor_id, agent_id
        
        persuader_belief = grid[persuader_pos[0]][persuader_pos[1]]
        defender_belief = grid[defender_pos[0]][defender_pos[1]]
        
        print(f"\nAgent {persuader_id} (persuader) @ {persuader_pos}")
        print(f"  Belief: {persuader_belief[:50]}...")
        print(f"Agent {defender_id} (defender) @ {defender_pos}")
        print(f"  Belief: {defender_belief[:50]}...")
        
        # Log iteration start
        logger.log_iteration_start(
            iteration, SIMULATION_ITERATIONS,
            persuader_id, persuader_pos,
            defender_id, defender_pos,
            persuader_belief, defender_belief
        )
        
        # Create conversation function with logger
        conversation_func = create_conversation_runner(
            persuader_belief,
            defender_belief,
            CONVERSATION_ROUNDS,
            logger
        )
        
        # Run conversation with GUI
        print(f"\n--- Conversation ({CONVERSATION_ROUNDS} rounds) ---")
        conversation, new_belief, change_type = run_conversation_with_gui(
            persuader_id=persuader_id,
            defender_id=defender_id,
            persuader_belief=persuader_belief,
            defender_belief=defender_belief,
            iteration=iteration,
            total_iterations=SIMULATION_ITERATIONS,
            conversation_func=conversation_func,
            close_delay_seconds=GUI_CLOSE_DELAY
        )
        
        # Print conversation summary (truncated for terminal)
        for entry in conversation:
            speaker = entry["speaker"].upper()
            content = entry["content"][:100].replace("\n", " ")
            print(f"[R{entry['round']}] {speaker}: {content}...")
        
        # Results
        old_belief_short = defender_belief[:40]
        new_belief_short = new_belief[:40]
        
        print(f"\n--- Defender's Decision ---")
        print(f"Old belief: {old_belief_short}...")
        print(f"New belief: {new_belief_short}...")
        
        # Log decision
        logger.log_decision(defender_belief, new_belief, change_type)
        
        # Update grid
        grid[defender_pos[0]][defender_pos[1]] = new_belief
        
        if change_type == "changed":
            belief_changes += 1
            print(">>> BELIEF UPDATED!")
        elif change_type == "similar":
            print(">>> Belief rephrased but similar")
        else:
            print(">>> Belief unchanged")
        
        print_grid(grid, beliefs, GRID_SIZE)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {SIMULATION_ITERATIONS}")
    print(f"Belief changes detected: {belief_changes}")
    
    # Print full final beliefs
    print_grid_beliefs(grid, GRID_SIZE, "Final Grid Beliefs")
    print_grid(grid, beliefs, GRID_SIZE)
    
    # Log final state
    logger.log_final_grid(grid, GRID_SIZE)
    logger.log_summary(SIMULATION_ITERATIONS, belief_changes)
    logger.close()
    
    print(f"\nFull log saved to: {logger.get_path()}")


if __name__ == "__main__":
    run_simulation()
