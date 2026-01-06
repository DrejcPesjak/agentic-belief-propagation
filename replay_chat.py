#!/usr/bin/env python3
"""
Replay Chat - Replays a conversation from a simulation log file using the GUI.

Usage:
    python replay_chat.py <log_file> <conversation_number>
    
Example:
    python replay_chat.py logs/simulation_20260103_001256.log 5
"""

import sys
import re
import time
from pathlib import Path

from chat_gui import ConversationGUI


def parse_log_file(log_path: str) -> dict:
    """Parse a simulation log file and extract all iterations."""
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split into iterations
    iteration_pattern = r"={80}\nITERATION (\d+)/(\d+)\n.*?(?=={80}\nITERATION|\Z|={80}\nFINAL GRID)"
    iterations = {}
    
    for match in re.finditer(iteration_pattern, content, re.DOTALL):
        iteration_num = int(match.group(1))
        iteration_text = match.group(0)
        
        # Extract persuader info (flexible position format: "(0, 2)" or "[mesh node 5]" etc.)
        persuader_match = re.search(
            r"PERSUADER: Agent (\d+) @ .+?\nBelief: (.+?)(?=\n\nDEFENDER:)",
            iteration_text, re.DOTALL
        )
        
        # Extract defender info
        defender_match = re.search(
            r"DEFENDER: Agent (\d+) @ .+?\nBelief: (.+?)(?=\n\n-{40})",
            iteration_text, re.DOTALL
        )
        
        if not persuader_match or not defender_match:
            continue
            
        persuader_id = int(persuader_match.group(1))
        persuader_belief = persuader_match.group(2).strip()
        
        defender_id = int(defender_match.group(1))
        defender_belief = defender_match.group(2).strip()
        
        # Extract messages
        messages = []
        message_pattern = r"\[Round (\d+)\] (PERSUADER|DEFENDER):\n(.+?)(?=\[Round|\n-{40}\nDECISION)"
        
        for msg_match in re.finditer(message_pattern, iteration_text, re.DOTALL):
            round_num = int(msg_match.group(1))
            speaker = msg_match.group(2).lower()
            content = msg_match.group(3).strip()
            messages.append({
                "round": round_num,
                "speaker": speaker,
                "content": content
            })
        
        # Extract decision
        decision_match = re.search(
            r"DECISION\n-{40}\nOld belief: (.+?)\n\nNew belief: (.+?)\n\nChange type: (\w+)",
            iteration_text, re.DOTALL
        )
        
        if decision_match:
            old_belief = decision_match.group(1).strip()
            new_belief = decision_match.group(2).strip()
            change_type = decision_match.group(3).strip().lower()
        else:
            old_belief = defender_belief
            new_belief = defender_belief
            change_type = "unchanged"
        
        iterations[iteration_num] = {
            "persuader_id": persuader_id,
            "defender_id": defender_id,
            "persuader_belief": persuader_belief,
            "defender_belief": defender_belief,
            "messages": messages,
            "old_belief": old_belief,
            "new_belief": new_belief,
            "change_type": change_type
        }
    
    return iterations


def replay_conversation(iteration_data: dict, iteration_num: int, total_iterations: int, delay: float = 0.8):
    """Replay a conversation using the GUI."""
    
    # Create GUI
    gui = ConversationGUI(
        persuader_id=iteration_data["persuader_id"],
        defender_id=iteration_data["defender_id"],
        persuader_belief=iteration_data["persuader_belief"],
        defender_belief=iteration_data["defender_belief"],
        iteration=iteration_num,
        total_iterations=total_iterations
    )
    gui.create_window()
    
    # Update window title to indicate replay
    gui.root.title(f"REPLAY - Iteration {iteration_num}/{total_iterations}")
    
    messages = iteration_data["messages"]
    current_msg_idx = 0
    
    def show_next_message():
        nonlocal current_msg_idx
        
        if current_msg_idx < len(messages):
            msg = messages[current_msg_idx]
            gui.add_message(msg["speaker"], msg["round"], msg["content"])
            current_msg_idx += 1
            gui.root.after(int(delay * 1000), show_next_message)
        else:
            # Show decision after all messages
            gui.root.after(500, show_decision)
    
    def show_decision():
        gui.show_decision(iteration_data["new_belief"], iteration_data["change_type"])
        # # Schedule close after viewing decision
        # gui.root.after(5000, lambda: gui.root.quit())
        
        # Window stays open until user closes it (X button or Ctrl+C)
    
    # Start showing messages after a brief delay
    gui.root.after(500, show_next_message)
    
    # Run the GUI
    gui.run_mainloop()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("Error: Please provide log file path and conversation number.")
        print("\nExample:")
        print("  python replay_chat.py logs/simulation_20260103_001256.log 5")
        sys.exit(1)
    
    log_path = sys.argv[1]
    try:
        conversation_num = int(sys.argv[2])
    except ValueError:
        print(f"Error: Conversation number must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    # Optional delay argument
    delay = 0.8
    if len(sys.argv) >= 4:
        try:
            delay = float(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid delay '{sys.argv[3]}', using default 0.8s")
    
    # Check if file exists
    if not Path(log_path).exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"Parsing log file: {log_path}")
    iterations = parse_log_file(log_path)
    
    if not iterations:
        print("Error: No conversations found in log file.")
        sys.exit(1)
    
    print(f"Found {len(iterations)} conversations.")
    
    if conversation_num not in iterations:
        available = sorted(iterations.keys())
        print(f"Error: Conversation {conversation_num} not found.")
        print(f"Available conversations: {available[0]}-{available[-1]}")
        sys.exit(1)
    
    iteration_data = iterations[conversation_num]
    total_iterations = max(iterations.keys())
    
    print(f"\nReplaying conversation {conversation_num}:")
    print(f"  Persuader: Agent {iteration_data['persuader_id']}")
    print(f"  Defender: Agent {iteration_data['defender_id']}")
    print(f"  Messages: {len(iteration_data['messages'])}")
    print(f"  Change type: {iteration_data['change_type']}")
    print(f"  Delay: {delay}s between messages")
    print("\nLaunching GUI...")
    
    replay_conversation(iteration_data, conversation_num, total_iterations, delay)
    
    print("Replay complete.")


if __name__ == "__main__":
    main()

