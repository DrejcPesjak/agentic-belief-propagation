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
from grid_layouts import Layout, Grid4Layout, Grid8Layout, RingLayout, MeshLayout, StarLayout
from network_gui import NetworkVisualizer


# === Configuration ===
# GRID_SIZE = 3  # 3x3 grid = 9 agents
N_AGENTS = 9
CONVERSATION_ROUNDS = 5  # 5 exchanges per interaction
SIMULATION_ITERATIONS = 5  # total interactions to simulate
MODEL = "gemma3"
BELIEFS_FILE = "money-philosophy.json"
GUI_CLOSE_DELAY = 5.0  # seconds to show final decision before closing
RANDOM_SEED = 42  # Seed for reproducible starting grid distribution (None for random)


def load_beliefs(filepath: str) -> list[str]:
    """Load beliefs from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["beliefs"]


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


def print_layout_grid(layout: Layout, beliefs: list[str]) -> None:
    """Print the grid showing belief indices for readability (for grid layouts)."""
    # Check if this is a grid layout with .size attribute
    if hasattr(layout, 'size'):
        size = layout.size
        print("\nCurrent Grid (belief indices):")
        print("-" * (size * 4 + 1))
        for row in range(size):
            row_display = []
            for col in range(size):
                agent_id = row * size + col
                belief = layout.get_belief(agent_id)
                # Find index or show * for custom beliefs
                try:
                    idx = beliefs.index(belief)
                    row_display.append(f" {idx} ")
                except ValueError:
                    row_display.append(" * ")  # Modified belief
            print("|" + "|".join(row_display) + "|")
            print("-" * (size * 4 + 1))
    else:
        # For non-grid layouts, just print agent beliefs in a list
        print("\nCurrent Beliefs (indices):")
        for agent_id in range(layout.get_agent_count()):
            belief = layout.get_belief(agent_id)
            try:
                idx = beliefs.index(belief)
                print(f"  Agent {agent_id}: [{idx}]")
            except ValueError:
                print(f"  Agent {agent_id}: [*]")


def print_layout_beliefs(layout: Layout, title: str) -> None:
    """Print full beliefs for all agents in layout."""
    print(f"\n{title}:")
    for agent_id in range(layout.get_agent_count()):
        belief = layout.get_belief(agent_id)
        pos_label = layout.get_position_label(agent_id)
        print(f"  Agent {agent_id} {pos_label}: {belief}")


def run_simulation():
    """Main simulation loop."""
    # Initialize logger
    logger = SimulationLogger()
    
    # Initialize layout
    # layout = Grid4Layout(GRID_SIZE)
    # layout = Grid8Layout(GRID_SIZE)
    layout = RingLayout(N_AGENTS)
    # layout = MeshLayout(GRID_SIZE)
    # layout = StarLayout(N_AGENTS)
    
    print("=" * 60)
    print("AGENTIC BELIEF PROPAGATION SIMULATION")
    print("=" * 60)
    print(f"Layout: {layout.__class__.__name__}")
    print(f"Number of agents: {N_AGENTS}")
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
    logger.log_config(N_AGENTS, CONVERSATION_ROUNDS, SIMULATION_ITERATIONS, MODEL, RANDOM_SEED)
    
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
    
    # Initialize layout with beliefs (seed already set above if RANDOM_SEED is not None)
    layout.initialize(beliefs)
    
    # Print and log starting beliefs
    print_layout_beliefs(layout, "Starting Grid Beliefs")
    print_layout_grid(layout, beliefs)
    logger.log_starting_grid_from_layout(layout)
    
    # Initialize network visualization
    network_viz = NetworkVisualizer(layout, beliefs, title="Belief Propagation Network")
    
    # Track changes
    belief_changes = 0
    
    # Run simulation iterations
    for iteration in range(1, SIMULATION_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{SIMULATION_ITERATIONS}")
        print("=" * 60)
        
        # Pick random agent
        agent_id = random.randint(0, layout.get_agent_count() - 1)
        
        # Pick random neighbor using layout
        neighbors = layout.get_neighbors(agent_id)
        neighbor_id = random.choice(neighbors)
        
        # 50/50 who is persuader vs defender
        if random.random() < 0.5:
            persuader_id, defender_id = agent_id, neighbor_id
        else:
            persuader_id, defender_id = neighbor_id, agent_id
        
        persuader_belief = layout.get_belief(persuader_id)
        defender_belief = layout.get_belief(defender_id)
        persuader_pos = layout.get_position_label(persuader_id)
        defender_pos = layout.get_position_label(defender_id)
        
        print(f"\nAgent {persuader_id} (persuader) @ {persuader_pos}")
        print(f"  Belief: {persuader_belief[:50]}...")
        print(f"Agent {defender_id} (defender) @ {defender_pos}")
        print(f"  Belief: {defender_belief[:50]}...")
        
        # Update network visualization to highlight active pair
        network_viz.set_active_conversation(persuader_id, defender_id, iteration, SIMULATION_ITERATIONS)
        
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
        
        # Update belief in layout
        layout.set_belief(defender_id, new_belief)
        
        # Update network visualization to show new belief colors
        network_viz.update_beliefs()
        
        if change_type == "changed":
            belief_changes += 1
            print(">>> BELIEF UPDATED!")
        elif change_type == "similar":
            print(">>> Belief rephrased but similar")
        else:
            print(">>> Belief unchanged")
        
        print_layout_grid(layout, beliefs)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {SIMULATION_ITERATIONS}")
    print(f"Belief changes detected: {belief_changes}")
    
    # Update network visualization to show completion
    network_viz.set_complete(SIMULATION_ITERATIONS, belief_changes)
    
    # Print full final beliefs
    print_layout_beliefs(layout, "Final Grid Beliefs")
    print_layout_grid(layout, beliefs)
    
    # Log final state
    logger.log_final_grid_from_layout(layout)
    logger.log_summary(SIMULATION_ITERATIONS, belief_changes)
    logger.close()
    
    print(f"\nFull log saved to: {logger.get_path()}")
    
    # Keep network visualization open until user closes it
    print("\nNetwork visualization window open. Close it to exit.")
    try:
        while network_viz.is_open():
            network_viz.update()
    except KeyboardInterrupt:
        pass
    finally:
        if network_viz.is_open():
            network_viz.close()


if __name__ == "__main__":
    run_simulation()
