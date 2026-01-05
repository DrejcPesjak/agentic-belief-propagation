"""
Network Visualization GUI for Belief Propagation Model
Displays the network topology and highlights active conversations.
"""

import tkinter as tk
from tkinter import font as tkfont
import math
from typing import Optional, List, Tuple
from grid_layouts import Layout, Grid4Layout, Grid8Layout, RingLayout, MeshLayout, StarLayout


# Color palette for beliefs (up to 10 distinct colors)
BELIEF_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
]

# Styling
CANVAS_BG = "#1a1a2e"
NODE_RADIUS = 25
NODE_OUTLINE = "#ffffff"
NODE_OUTLINE_WIDTH = 2
EDGE_COLOR = "#404060"
EDGE_WIDTH = 2
ACTIVE_EDGE_COLOR = "#FFD700"
ACTIVE_EDGE_WIDTH = 4
ACTIVE_NODE_OUTLINE = "#FFD700"
ACTIVE_NODE_OUTLINE_WIDTH = 4
LABEL_COLOR = "#ffffff"
MODIFIED_BELIEF_COLOR = "#808080"  # Gray for beliefs not in original set


class NetworkVisualizer:
    """Visualizes network topology and active conversations."""
    
    def __init__(self, layout: Layout, beliefs: List[str], title: str = "Network Topology"):
        self.layout = layout
        self.beliefs = beliefs
        self.n_agents = layout.get_agent_count()
        
        # Node positions (will be calculated based on layout type)
        self.positions: List[Tuple[float, float]] = []
        
        # Active agents
        self.active_persuader: Optional[int] = None
        self.active_defender: Optional[int] = None
        
        # Tkinter setup
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=CANVAS_BG)
        
        # Calculate canvas size based on layout
        self.canvas_width = 500
        self.canvas_height = 500
        self.padding = 60
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=CANVAS_BG,
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Info label at bottom
        self.info_var = tk.StringVar(value="Waiting for conversation...")
        self.info_label = tk.Label(
            self.root,
            textvariable=self.info_var,
            bg=CANVAS_BG,
            fg="#ffffff",
            font=("Helvetica", 11)
        )
        self.info_label.pack(pady=(0, 10))
        
        # Legend frame
        self._create_legend()
        
        # Calculate positions and draw initial state
        self._calculate_positions()
        self._draw_network()
        
        # Don't block - just update
        self.root.update()
    
    def _create_legend(self):
        """Create a legend showing belief colors."""
        legend_frame = tk.Frame(self.root, bg=CANVAS_BG)
        legend_frame.pack(pady=(0, 10))
        
        tk.Label(
            legend_frame,
            text="Beliefs: ",
            bg=CANVAS_BG,
            fg="#ffffff",
            font=("Helvetica", 9)
        ).pack(side=tk.LEFT)
        
        for i, belief in enumerate(self.beliefs):
            color = BELIEF_COLORS[i % len(BELIEF_COLORS)]
            # Small colored square
            square = tk.Canvas(legend_frame, width=15, height=15, bg=CANVAS_BG, highlightthickness=0)
            square.create_rectangle(2, 2, 13, 13, fill=color, outline="")
            square.pack(side=tk.LEFT, padx=2)
            
            tk.Label(
                legend_frame,
                text=f"[{i}]",
                bg=CANVAS_BG,
                fg="#aaaaaa",
                font=("Helvetica", 8)
            ).pack(side=tk.LEFT, padx=(0, 5))
    
    def _calculate_positions(self):
        """Calculate node positions based on layout type."""
        cx = self.canvas_width / 2
        cy = self.canvas_height / 2
        usable_width = self.canvas_width - 2 * self.padding
        usable_height = self.canvas_height - 2 * self.padding
        
        if isinstance(self.layout, (Grid4Layout, Grid8Layout)):
            # Grid layout - arrange in a grid
            size = self.layout.size
            cell_w = usable_width / size
            cell_h = usable_height / size
            
            for agent_id in range(self.n_agents):
                row = agent_id // size
                col = agent_id % size
                x = self.padding + col * cell_w + cell_w / 2
                y = self.padding + row * cell_h + cell_h / 2
                self.positions.append((x, y))
                
        elif isinstance(self.layout, RingLayout):
            # Ring layout - arrange in a circle
            radius = min(usable_width, usable_height) / 2 - NODE_RADIUS
            for agent_id in range(self.n_agents):
                angle = 2 * math.pi * agent_id / self.n_agents - math.pi / 2  # Start from top
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                self.positions.append((x, y))
                
        elif isinstance(self.layout, MeshLayout):
            # Mesh layout - arrange in a circle (connections make it look like a mesh)
            radius = min(usable_width, usable_height) / 2 - NODE_RADIUS
            for agent_id in range(self.n_agents):
                angle = 2 * math.pi * agent_id / self.n_agents - math.pi / 2
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                self.positions.append((x, y))
                
        elif isinstance(self.layout, StarLayout):
            # Star layout - hub in center, spokes around
            radius = min(usable_width, usable_height) / 2 - NODE_RADIUS
            # Hub at center
            self.positions.append((cx, cy))
            # Spokes around
            for agent_id in range(1, self.n_agents):
                angle = 2 * math.pi * (agent_id - 1) / (self.n_agents - 1) - math.pi / 2
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                self.positions.append((x, y))
        else:
            # Fallback: arrange in a circle
            radius = min(usable_width, usable_height) / 2 - NODE_RADIUS
            for agent_id in range(self.n_agents):
                angle = 2 * math.pi * agent_id / self.n_agents - math.pi / 2
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                self.positions.append((x, y))
    
    def _get_belief_color(self, agent_id: int) -> str:
        """Get color for an agent based on their belief."""
        belief = self.layout.get_belief(agent_id)
        try:
            idx = self.beliefs.index(belief)
            return BELIEF_COLORS[idx % len(BELIEF_COLORS)]
        except ValueError:
            return MODIFIED_BELIEF_COLOR  # Modified belief
    
    def _draw_network(self):
        """Draw the entire network (edges and nodes)."""
        self.canvas.delete("all")
        
        # Draw edges first (so nodes appear on top)
        self._draw_edges()
        
        # Draw nodes
        self._draw_nodes()
    
    def _draw_edges(self):
        """Draw all edges in the network."""
        drawn_edges = set()
        
        for agent_id in range(self.n_agents):
            neighbors = self.layout.get_neighbors(agent_id)
            x1, y1 = self.positions[agent_id]
            
            for neighbor_id in neighbors:
                # Avoid drawing the same edge twice
                edge_key = tuple(sorted([agent_id, neighbor_id]))
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                
                x2, y2 = self.positions[neighbor_id]
                
                # Check if this is the active edge
                is_active = (
                    self.active_persuader is not None and
                    self.active_defender is not None and
                    edge_key == tuple(sorted([self.active_persuader, self.active_defender]))
                )
                
                if is_active:
                    # Draw active edge with glow effect
                    self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill=ACTIVE_EDGE_COLOR,
                        width=ACTIVE_EDGE_WIDTH + 4,
                        tags="edge_glow"
                    )
                    self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill=ACTIVE_EDGE_COLOR,
                        width=ACTIVE_EDGE_WIDTH,
                        tags="edge_active"
                    )
                else:
                    self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill=EDGE_COLOR,
                        width=EDGE_WIDTH,
                        tags="edge"
                    )
    
    def _draw_nodes(self):
        """Draw all nodes."""
        for agent_id in range(self.n_agents):
            x, y = self.positions[agent_id]
            color = self._get_belief_color(agent_id)
            
            # Determine if active
            is_persuader = agent_id == self.active_persuader
            is_defender = agent_id == self.active_defender
            is_active = is_persuader or is_defender
            
            # Draw node circle
            r = NODE_RADIUS
            if is_active:
                # Draw glow/highlight
                self.canvas.create_oval(
                    x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                    fill="",
                    outline=ACTIVE_NODE_OUTLINE,
                    width=ACTIVE_NODE_OUTLINE_WIDTH,
                    tags="node_glow"
                )
            
            # Main node
            outline_color = NODE_OUTLINE
            outline_width = NODE_OUTLINE_WIDTH
            
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color,
                outline=outline_color,
                width=outline_width,
                tags="node"
            )
            
            # Agent ID label
            self.canvas.create_text(
                x, y,
                text=str(agent_id),
                fill=LABEL_COLOR,
                font=("Helvetica", 12, "bold"),
                tags="label"
            )
            
            # Role indicator (P for persuader, D for defender)
            if is_persuader:
                self.canvas.create_text(
                    x, y - r - 12,
                    text="P",
                    fill=ACTIVE_NODE_OUTLINE,
                    font=("Helvetica", 10, "bold"),
                    tags="role"
                )
            elif is_defender:
                self.canvas.create_text(
                    x, y - r - 12,
                    text="D",
                    fill=ACTIVE_NODE_OUTLINE,
                    font=("Helvetica", 10, "bold"),
                    tags="role"
                )
    
    def set_active_conversation(self, persuader_id: int, defender_id: int, iteration: int, total: int):
        """Highlight the active conversation pair."""
        self.active_persuader = persuader_id
        self.active_defender = defender_id
        
        # Update info label
        self.info_var.set(
            f"Iteration {iteration}/{total}: Agent {persuader_id} (P) → Agent {defender_id} (D)"
        )
        
        # Redraw
        self._draw_network()
        self.root.update()
    
    def clear_active(self):
        """Clear the active conversation highlight."""
        self.active_persuader = None
        self.active_defender = None
        self._draw_network()
        self.root.update()
    
    def update_beliefs(self):
        """Redraw nodes to reflect updated beliefs."""
        self._draw_network()
        self.root.update()
    
    def set_complete(self, iterations: int, changes: int):
        """Mark simulation as complete."""
        self.active_persuader = None
        self.active_defender = None
        self.info_var.set(f"Complete! {iterations} iterations, {changes} belief changes")
        self._draw_network()
        self.root.update()
    
    def update(self):
        """Process pending GUI events."""
        self.root.update()
    
    def close(self):
        """Close the visualization window."""
        self.root.destroy()
    
    def is_open(self) -> bool:
        """Check if the window is still open."""
        try:
            return self.root.winfo_exists()
        except tk.TclError:
            return False


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import time
    
    # Test with Ring layout
    test_beliefs = [
        "Belief A",
        "Belief B", 
        "Belief C",
        "Belief D",
        "Belief E",
        "Belief F",
        "Belief G"
    ]
    
    # Test different layouts
    layouts = [
        ("Grid4", Grid4Layout(3)),
        ("Grid8", Grid8Layout(3)),
        ("Ring", RingLayout(9)),
        ("Mesh", MeshLayout(6)),
        ("Star", StarLayout(7)),
    ]
    
    for name, layout in layouts:
        print(f"\nTesting {name} layout...")
        layout.initialize(test_beliefs)
        
        viz = NetworkVisualizer(layout, test_beliefs, title=f"Test: {name}")
        
        # Simulate a few iterations
        for i in range(3):
            import random
            agent = random.randint(0, layout.get_agent_count() - 1)
            neighbors = layout.get_neighbors(agent)
            if neighbors:
                neighbor = random.choice(neighbors)
                viz.set_active_conversation(agent, neighbor, i + 1, 3)
                time.sleep(1)
        
        viz.set_complete(3, 1)
        time.sleep(1)
        viz.close()
    
    print("\nAll tests complete!")

