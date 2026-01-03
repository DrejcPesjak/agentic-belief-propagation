"""
Modern Chat GUI for Belief Propagation Conversations
Displays live conversation between persuader and defender agents.
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import queue
from typing import Callable, Optional


class ConversationGUI:
    """GUI window for displaying agent conversations."""
    
    # Color scheme - dark modern theme
    COLORS = {
        "bg_dark": "#0f0f0f",
        "bg_panel": "#1a1a1a",
        "bg_chat": "#141414",
        "accent_persuader": "#3b82f6",  # Blue
        "accent_defender": "#8b5cf6",   # Purple
        "text_primary": "#f5f5f5",
        "text_secondary": "#a3a3a3",
        "text_muted": "#737373",
        "bubble_persuader": "#1e3a5f",
        "bubble_defender": "#2d1f4e",
        "border": "#2a2a2a",
        "decision_changed": "#22c55e",    # Green
        "decision_similar": "#f59e0b",    # Orange
        "decision_unchanged": "#ef4444",  # Red
    }
    
    def __init__(self, 
                 persuader_id: int, 
                 defender_id: int,
                 persuader_belief: str, 
                 defender_belief: str,
                 iteration: int,
                 total_iterations: int):
        self.persuader_id = persuader_id
        self.defender_id = defender_id
        self.persuader_belief = persuader_belief
        self.defender_belief = defender_belief
        self.iteration = iteration
        self.total_iterations = total_iterations
        
        self.root: Optional[tk.Tk] = None
        self.chat_frame: Optional[tk.Frame] = None
        self.chat_canvas: Optional[tk.Canvas] = None
        self.scrollable_frame: Optional[tk.Frame] = None
        self.decision_label: Optional[tk.Label] = None
        self.decision_frame: Optional[tk.Frame] = None
        
        # Message queue for thread-safe updates
        self.message_queue: queue.Queue = queue.Queue()
        self._should_close = False
        self._close_scheduled = False
        
    def create_window(self):
        """Create the main window and all widgets."""
        self.root = tk.Tk()
        self.root.title(f"Belief Propagation - Iteration {self.iteration}/{self.total_iterations}")
        self.root.configure(bg=self.COLORS["bg_dark"])
        self.root.geometry("1400x800")
        self.root.minsize(1000, 600)
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=0)  # Header
        self.root.grid_rowconfigure(1, weight=1)  # Main content
        self.root.grid_rowconfigure(2, weight=0)  # Decision footer
        
        # Fonts
        self.font_title = tkfont.Font(family="JetBrains Mono", size=11, weight="bold")
        self.font_label = tkfont.Font(family="JetBrains Mono", size=9)
        self.font_belief = tkfont.Font(family="Georgia", size=12)
        self.font_chat = tkfont.Font(family="SF Pro Display", size=11)
        self.font_decision = tkfont.Font(family="JetBrains Mono", size=11, weight="bold")
        
        # Header
        self._create_header()
        
        # Main content area
        self._create_main_content()
        
        # Decision footer
        self._create_decision_footer()
        
        # Start processing message queue
        self._process_queue()
        
    def _create_header(self):
        """Create header with iteration info."""
        header = tk.Frame(self.root, bg=self.COLORS["bg_dark"], height=50)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=20, pady=(15, 5))
        
        title = tk.Label(
            header,
            text=f"◉ ITERATION {self.iteration} OF {self.total_iterations}",
            font=self.font_title,
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg_dark"]
        )
        title.pack(anchor="w")
        
    def _create_main_content(self):
        """Create the three-column layout."""
        # Left panel - Persuader
        left_panel = self._create_belief_panel(
            "PERSUADER",
            f"Agent {self.persuader_id}",
            self.persuader_belief,
            self.COLORS["accent_persuader"]
        )
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        
        # Middle panel - Chat
        self._create_chat_panel()
        
        # Right panel - Defender
        right_panel = self._create_belief_panel(
            "DEFENDER",
            f"Agent {self.defender_id}",
            self.defender_belief,
            self.COLORS["accent_defender"]
        )
        right_panel.grid(row=1, column=2, sticky="nsew", padx=(10, 20), pady=10)
        
    def _create_belief_panel(self, role: str, agent_label: str, belief: str, accent: str) -> tk.Frame:
        """Create a belief display panel."""
        panel = tk.Frame(self.root, bg=self.COLORS["bg_panel"])
        
        # Role label with accent
        role_frame = tk.Frame(panel, bg=self.COLORS["bg_panel"])
        role_frame.pack(fill="x", padx=20, pady=(20, 5))
        
        accent_bar = tk.Frame(role_frame, bg=accent, width=4, height=20)
        accent_bar.pack(side="left", padx=(0, 10))
        
        role_label = tk.Label(
            role_frame,
            text=role,
            font=self.font_title,
            fg=accent,
            bg=self.COLORS["bg_panel"]
        )
        role_label.pack(side="left")
        
        # Agent ID
        agent_id_label = tk.Label(
            panel,
            text=agent_label,
            font=self.font_label,
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg_panel"]
        )
        agent_id_label.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Separator
        sep = tk.Frame(panel, bg=self.COLORS["border"], height=1)
        sep.pack(fill="x", padx=20, pady=5)
        
        # Belief text
        belief_label = tk.Label(
            panel,
            text=belief,
            font=self.font_belief,
            fg=self.COLORS["text_primary"],
            bg=self.COLORS["bg_panel"],
            wraplength=280,
            justify="left"
        )
        belief_label.pack(anchor="nw", padx=20, pady=20, fill="both", expand=True)
        
        return panel
    
    def _create_chat_panel(self):
        """Create the scrollable chat panel."""
        chat_container = tk.Frame(self.root, bg=self.COLORS["bg_chat"])
        chat_container.grid(row=1, column=1, sticky="nsew", padx=5, pady=10)
        
        # Header
        header_frame = tk.Frame(chat_container, bg=self.COLORS["bg_chat"])
        header_frame.pack(fill="x", padx=20, pady=(15, 10))
        
        chat_title = tk.Label(
            header_frame,
            text="CONVERSATION",
            font=self.font_title,
            fg=self.COLORS["text_secondary"],
            bg=self.COLORS["bg_chat"]
        )
        chat_title.pack(anchor="w")
        
        # Separator
        sep = tk.Frame(chat_container, bg=self.COLORS["border"], height=1)
        sep.pack(fill="x", padx=20, pady=5)
        
        # Scrollable chat area
        self.chat_canvas = tk.Canvas(
            chat_container, 
            bg=self.COLORS["bg_chat"],
            highlightthickness=0
        )
        scrollbar = tk.Scrollbar(
            chat_container, 
            orient="vertical", 
            command=self.chat_canvas.yview
        )
        
        self.scrollable_frame = tk.Frame(self.chat_canvas, bg=self.COLORS["bg_chat"])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        
        self.canvas_window = self.chat_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Make scrollable frame expand with canvas
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Pack with scrollbar
        self.chat_canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=10)
        scrollbar.pack(side="right", fill="y", pady=10, padx=(0, 5))
        
        # Bind mouse wheel to this canvas only
        self.chat_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.chat_canvas.bind("<Button-4>", self._on_mousewheel)
        self.chat_canvas.bind("<Button-5>", self._on_mousewheel)
        
    def _on_canvas_configure(self, event):
        """Handle canvas resize."""
        self.chat_canvas.itemconfig(self.canvas_window, width=event.width)
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if event.num == 4:
            self.chat_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.chat_canvas.yview_scroll(1, "units")
        else:
            self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _create_decision_footer(self):
        """Create the decision footer."""
        self.decision_frame = tk.Frame(self.root, bg=self.COLORS["bg_dark"], height=80)
        self.decision_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=(5, 20))
        
        # Separator line
        sep = tk.Frame(self.decision_frame, bg=self.COLORS["border"], height=1)
        sep.pack(fill="x", pady=(0, 15))
        
        # Decision label placeholder
        self.decision_label = tk.Label(
            self.decision_frame,
            text="⏳ Conversation in progress...",
            font=self.font_decision,
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg_dark"]
        )
        self.decision_label.pack(anchor="w")
        
    def _process_queue(self):
        """Process messages from the queue (runs in main thread)."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == "message":
                    speaker, round_num, content = data
                    self._add_message_impl(speaker, round_num, content)
                elif msg_type == "decision":
                    new_belief, change_type = data
                    self._show_decision_impl(new_belief, change_type)
                elif msg_type == "close":
                    delay_ms = data
                    self._schedule_close(delay_ms)
                    
        except queue.Empty:
            pass
        
        # Schedule next queue check
        if self.root and not self._should_close:
            self.root.after(50, self._process_queue)
    
    def _add_message_impl(self, speaker: str, round_num: int, content: str):
        """Actually add the message (must be called from main thread)."""
        if not self.scrollable_frame:
            return
            
        is_persuader = speaker == "persuader"
        
        # Message container
        msg_container = tk.Frame(self.scrollable_frame, bg=self.COLORS["bg_chat"])
        msg_container.pack(fill="x", pady=8, padx=10)
        
        # Round indicator
        round_label = tk.Label(
            msg_container,
            text=f"Round {round_num}",
            font=tkfont.Font(size=8),
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg_chat"]
        )
        
        # Message bubble
        bubble_color = self.COLORS["bubble_persuader"] if is_persuader else self.COLORS["bubble_defender"]
        accent = self.COLORS["accent_persuader"] if is_persuader else self.COLORS["accent_defender"]
        
        bubble = tk.Frame(msg_container, bg=bubble_color, padx=15, pady=12)
        
        # Speaker label inside bubble
        speaker_label = tk.Label(
            bubble,
            text=f"{'PERSUADER' if is_persuader else 'DEFENDER'}",
            font=tkfont.Font(size=9, weight="bold"),
            fg=accent,
            bg=bubble_color
        )
        speaker_label.pack(anchor="w")
        
        # Message text
        msg_label = tk.Label(
            bubble,
            text=content,
            font=self.font_chat,
            fg=self.COLORS["text_primary"],
            bg=bubble_color,
            wraplength=400,
            justify="left"
        )
        msg_label.pack(anchor="w", pady=(5, 0))
        
        # Position based on speaker
        if is_persuader:
            round_label.pack(anchor="w", pady=(0, 3))
            bubble.pack(anchor="w")
        else:
            round_label.pack(anchor="e", pady=(0, 3))
            bubble.pack(anchor="e")
        
        # Scroll to bottom
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)
        
    def _show_decision_impl(self, new_belief: str, change_type: str):
        """Actually show the decision (must be called from main thread)."""
        if not self.decision_label:
            return
            
        # Determine color based on change type
        if change_type == "changed":
            color = self.COLORS["decision_changed"]
            prefix = "✓ BELIEF UPDATED"
        elif change_type == "similar":
            color = self.COLORS["decision_similar"]
            prefix = "◐ BELIEF REPHRASED"
        else:
            color = self.COLORS["decision_unchanged"]
            prefix = "✗ BELIEF UNCHANGED"
        
        # Update decision label
        self.decision_label.configure(
            text=f"{prefix}: {new_belief[:100]}{'...' if len(new_belief) > 100 else ''}",
            fg=color
        )
        
    def _schedule_close(self, delay_ms: int):
        """Schedule window close."""
        if not self._close_scheduled:
            self._close_scheduled = True
            self.root.after(delay_ms, self._do_close)
            
    def _do_close(self):
        """Actually close the window."""
        self._should_close = True
        if self.root:
            self.root.quit()
        
    def add_message(self, speaker: str, round_num: int, content: str):
        """Add a message to the chat (thread-safe)."""
        self.message_queue.put(("message", (speaker, round_num, content)))
        
    def show_decision(self, new_belief: str, change_type: str):
        """Show the final decision (thread-safe)."""
        self.message_queue.put(("decision", (new_belief, change_type)))
        
    def schedule_close(self, delay_ms: int = 3000):
        """Schedule window close (thread-safe)."""
        self.message_queue.put(("close", delay_ms))
        
    def run_mainloop(self):
        """Run the tkinter mainloop (must be in main thread)."""
        self.root.mainloop()
        # Cleanup after mainloop exits
        if self.root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
        self.root = None


def run_conversation_with_gui(
    persuader_id: int,
    defender_id: int,
    persuader_belief: str,
    defender_belief: str,
    iteration: int,
    total_iterations: int,
    conversation_func: Callable,
    close_delay_seconds: float = 3.0
) -> tuple[list[dict], str, str]:
    """
    Run a conversation with live GUI display.
    
    Args:
        persuader_id: ID of persuader agent
        defender_id: ID of defender agent
        persuader_belief: Persuader's belief
        defender_belief: Defender's belief
        iteration: Current iteration number
        total_iterations: Total iterations
        conversation_func: Function(on_message_callback) -> (history, new_belief)
        close_delay_seconds: How long to show decision before closing
        
    Returns:
        Tuple of (conversation_history, new_belief, change_type)
    """
    # Create GUI
    gui = ConversationGUI(
        persuader_id, defender_id,
        persuader_belief, defender_belief,
        iteration, total_iterations
    )
    gui.create_window()
    
    # Result container
    result = {"history": None, "new_belief": None, "error": None}
    
    def worker():
        """Background worker for LLM conversation."""
        try:
            history, new_belief = conversation_func(gui.add_message)
            result["history"] = history
            result["new_belief"] = new_belief
            
            # Determine change type
            old_clean = defender_belief.lower().strip()
            new_clean = new_belief.lower().strip()
            
            if new_clean == old_clean:
                change_type = "unchanged"
            elif new_belief[:20].lower() == defender_belief[:20].lower():
                change_type = "similar"
            else:
                change_type = "changed"
            
            result["change_type"] = change_type
            
            # Show decision and schedule close
            gui.show_decision(new_belief, change_type)
            gui.schedule_close(int(close_delay_seconds * 1000))
            
        except Exception as e:
            result["error"] = str(e)
            gui.schedule_close(1000)
    
    # Start worker thread
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    
    # Run GUI mainloop (blocks until window closes)
    gui.run_mainloop()
    
    # Wait for worker to finish
    worker_thread.join(timeout=1.0)
    
    if result["error"]:
        raise RuntimeError(result["error"])
    
    return result["history"], result["new_belief"], result["change_type"]


# Test the GUI
if __name__ == "__main__":
    import time
    
    def mock_conversation(on_message):
        """Mock conversation for testing."""
        messages = [
            ("persuader", 1, "Let me share why saving aggressively has transformed my life. Having that safety buffer means I never have to make decisions out of fear or desperation."),
            ("defender", 1, "I understand the appeal of security, but aren't you missing out on life while you're busy saving? Money sitting in a bank account isn't creating memories."),
            ("persuader", 2, "But here's the thing - that saved money IS creating something valuable: freedom. When unexpected opportunities or problems arise, I can handle them without stress."),
            ("defender", 2, "Freedom is subjective though. I feel more free spending on experiences with loved ones NOW, rather than hoarding for some uncertain future that may never come."),
            ("persuader", 3, "The future always comes. And those who prepared for it have options. Those who didn't have to scramble. Which sounds more freeing to you?"),
            ("defender", 3, "Fair point, but there's a middle ground. I'm not saying zero savings, but prioritizing experiences over excessive accumulation seems healthier."),
        ]
        
        history = []
        for speaker, round_num, content in messages:
            on_message(speaker, round_num, content)
            history.append({"round": round_num, "speaker": speaker, "content": content})
            time.sleep(0.8)
        
        new_belief = "I believe in balanced approach - saving for security while still investing in meaningful experiences and relationships."
        return history, new_belief
    
    # Run test
    history, new_belief, change_type = run_conversation_with_gui(
        persuader_id=5,
        defender_id=8,
        persuader_belief="I save aggressively because having a safety buffer is the fastest way to feel free and make better decisions.",
        defender_belief="I spend on experiences and people now because life is short and money is pointless if you never use it.",
        iteration=1,
        total_iterations=20,
        conversation_func=mock_conversation,
        close_delay_seconds=3.0
    )
    
    print(f"\nConversation complete!")
    print(f"Change type: {change_type}")
    print(f"New belief: {new_belief}")
