"""
Grid Layouts for Belief Propagation Model
Different network topologies for agent interactions.

Available layouts:
- grid4: 2D grid with 4 neighbors (up, down, left, right)
- grid8: 2D grid with 8 neighbors (includes diagonals)
- ring: Circular arrangement, each agent has 2 neighbors
- mesh: Fully connected, every agent connected to every other
- star: One central node connected to all others
"""

import random
from typing import Callable
from abc import ABC, abstractmethod


class Layout(ABC):
    """Abstract base class for network layouts."""
    
    @abstractmethod
    def initialize(self, beliefs: list[str], n_agents: int) -> list[str]:
        """Initialize agents with beliefs. Returns list of beliefs indexed by agent ID."""
        pass
    
    @abstractmethod
    def get_neighbors(self, agent_id: int) -> list[int]:
        """Get list of neighbor agent IDs for a given agent."""
        pass
    
    @abstractmethod
    def get_agent_count(self) -> int:
        """Return total number of agents."""
        pass
    
    @abstractmethod
    def get_position_label(self, agent_id: int) -> str:
        """Return a human-readable position label for an agent."""
        pass


# =============================================================================
# GRID 4 NEIGHBORS (Original)
# =============================================================================

class Grid4Layout(Layout):
    """2D grid with 4 neighbors (up, down, left, right)."""
    
    def __init__(self, size: int):
        self.size = size
        self.n_agents = size * size
        self.beliefs: list[str] = []
    
    def initialize(self, beliefs: list[str], n_agents: int = None) -> list[str]:
        """Initialize grid with randomly sampled beliefs."""
        self.beliefs = [random.choice(beliefs) for _ in range(self.n_agents)]
        return self.beliefs
    
    def get_neighbors(self, agent_id: int) -> list[int]:
        """Get 4-connected neighbors (up, down, left, right)."""
        row, col = self._id_to_pos(agent_id)
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append(self._pos_to_id(nr, nc))
        return neighbors
    
    def get_agent_count(self) -> int:
        return self.n_agents
    
    def get_position_label(self, agent_id: int) -> str:
        row, col = self._id_to_pos(agent_id)
        return f"({row}, {col})"
    
    def _pos_to_id(self, row: int, col: int) -> int:
        return row * self.size + col
    
    def _id_to_pos(self, agent_id: int) -> tuple[int, int]:
        return agent_id // self.size, agent_id % self.size
    
    def get_belief(self, agent_id: int) -> str:
        return self.beliefs[agent_id]
    
    def set_belief(self, agent_id: int, belief: str):
        self.beliefs[agent_id] = belief


# =============================================================================
# GRID 8 NEIGHBORS (With Diagonals)
# =============================================================================

class Grid8Layout(Layout):
    """2D grid with 8 neighbors (includes diagonals)."""
    
    def __init__(self, size: int):
        self.size = size
        self.n_agents = size * size
        self.beliefs: list[str] = []
    
    def initialize(self, beliefs: list[str], n_agents: int = None) -> list[str]:
        """Initialize grid with randomly sampled beliefs."""
        self.beliefs = [random.choice(beliefs) for _ in range(self.n_agents)]
        return self.beliefs
    
    def get_neighbors(self, agent_id: int) -> list[int]:
        """Get 8-connected neighbors (including diagonals)."""
        row, col = self._id_to_pos(agent_id)
        neighbors = []
        # All 8 directions: up, down, left, right, and 4 diagonals
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip self
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    neighbors.append(self._pos_to_id(nr, nc))
        return neighbors
    
    def get_agent_count(self) -> int:
        return self.n_agents
    
    def get_position_label(self, agent_id: int) -> str:
        row, col = self._id_to_pos(agent_id)
        return f"({row}, {col})"
    
    def _pos_to_id(self, row: int, col: int) -> int:
        return row * self.size + col
    
    def _id_to_pos(self, agent_id: int) -> tuple[int, int]:
        return agent_id // self.size, agent_id % self.size
    
    def get_belief(self, agent_id: int) -> str:
        return self.beliefs[agent_id]
    
    def set_belief(self, agent_id: int, belief: str):
        self.beliefs[agent_id] = belief


# =============================================================================
# RING LAYOUT
# =============================================================================

class RingLayout(Layout):
    """Circular arrangement where each agent has exactly 2 neighbors (left and right)."""
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.beliefs: list[str] = []
    
    def initialize(self, beliefs: list[str], n_agents: int = None) -> list[str]:
        """Initialize ring with randomly sampled beliefs."""
        self.beliefs = [random.choice(beliefs) for _ in range(self.n_agents)]
        return self.beliefs
    
    def get_neighbors(self, agent_id: int) -> list[int]:
        """Get 2 neighbors (previous and next in ring)."""
        prev_id = (agent_id - 1) % self.n_agents
        next_id = (agent_id + 1) % self.n_agents
        return [prev_id, next_id]
    
    def get_agent_count(self) -> int:
        return self.n_agents
    
    def get_position_label(self, agent_id: int) -> str:
        return f"[ring pos {agent_id}]"
    
    def get_belief(self, agent_id: int) -> str:
        return self.beliefs[agent_id]
    
    def set_belief(self, agent_id: int, belief: str):
        self.beliefs[agent_id] = belief


# =============================================================================
# MESH LAYOUT (Fully Connected)
# =============================================================================

class MeshLayout(Layout):
    """Fully connected network - every agent is connected to every other agent."""
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.beliefs: list[str] = []
    
    def initialize(self, beliefs: list[str], n_agents: int = None) -> list[str]:
        """Initialize mesh with randomly sampled beliefs."""
        self.beliefs = [random.choice(beliefs) for _ in range(self.n_agents)]
        return self.beliefs
    
    def get_neighbors(self, agent_id: int) -> list[int]:
        """Get all other agents as neighbors."""
        return [i for i in range(self.n_agents) if i != agent_id]
    
    def get_agent_count(self) -> int:
        return self.n_agents
    
    def get_position_label(self, agent_id: int) -> str:
        return f"[mesh node {agent_id}]"
    
    def get_belief(self, agent_id: int) -> str:
        return self.beliefs[agent_id]
    
    def set_belief(self, agent_id: int, belief: str):
        self.beliefs[agent_id] = belief


# =============================================================================
# STAR LAYOUT
# =============================================================================

class StarLayout(Layout):
    """Star topology - one central hub connected to all other nodes.
    
    Agent 0 is the hub, all others are spokes.
    Hub is connected to all spokes.
    Spokes are only connected to the hub.
    """
    
    def __init__(self, n_agents: int):
        if n_agents < 2:
            raise ValueError("Star layout requires at least 2 agents")
        self.n_agents = n_agents
        self.hub_id = 0
        self.beliefs: list[str] = []
    
    def initialize(self, beliefs: list[str], n_agents: int = None) -> list[str]:
        """Initialize star with randomly sampled beliefs."""
        self.beliefs = [random.choice(beliefs) for _ in range(self.n_agents)]
        return self.beliefs
    
    def get_neighbors(self, agent_id: int) -> list[int]:
        """Hub connects to all; spokes only connect to hub."""
        if agent_id == self.hub_id:
            # Hub is connected to all spokes
            return [i for i in range(1, self.n_agents)]
        else:
            # Spokes only connected to hub
            return [self.hub_id]
    
    def get_agent_count(self) -> int:
        return self.n_agents
    
    def get_position_label(self, agent_id: int) -> str:
        if agent_id == self.hub_id:
            return "[hub]"
        else:
            return f"[spoke {agent_id}]"
    
    def get_belief(self, agent_id: int) -> str:
        return self.beliefs[agent_id]
    
    def set_belief(self, agent_id: int, belief: str):
        self.beliefs[agent_id] = belief


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_layout(layout_type: str, n_agents: int = 9, grid_size: int = 3) -> Layout:
    """
    Factory function to create a layout by name.
    
    Args:
        layout_type: One of 'grid4', 'grid8', 'ring', 'mesh', 'star'
        n_agents: Number of agents (used for ring, mesh, star)
        grid_size: Grid dimension (used for grid4, grid8; n_agents = grid_size^2)
    
    Returns:
        Layout instance
    """
    layout_type = layout_type.lower()
    
    if layout_type == 'grid4':
        return Grid4Layout(grid_size)
    elif layout_type == 'grid8':
        return Grid8Layout(grid_size)
    elif layout_type == 'ring':
        return RingLayout(n_agents)
    elif layout_type == 'mesh':
        return MeshLayout(n_agents)
    elif layout_type == 'star':
        return StarLayout(n_agents)
    else:
        raise ValueError(f"Unknown layout type: {layout_type}. "
                        f"Available: grid4, grid8, ring, mesh, star")


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def print_layout_info(layout: Layout):
    """Print information about a layout's connectivity."""
    print(f"\nLayout: {layout.__class__.__name__}")
    print(f"Agents: {layout.get_agent_count()}")
    print("-" * 40)
    
    for agent_id in range(layout.get_agent_count()):
        neighbors = layout.get_neighbors(agent_id)
        pos = layout.get_position_label(agent_id)
        print(f"Agent {agent_id} {pos}: neighbors = {neighbors}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test all layouts
    print("=" * 50)
    print("TESTING GRID LAYOUTS")
    print("=" * 50)
    
    test_beliefs = ["Belief A", "Belief B", "Belief C"]
    
    # Test Grid4
    layout = create_layout('grid4', grid_size=3)
    layout.initialize(test_beliefs)
    print_layout_info(layout)
    
    # Test Grid8
    layout = create_layout('grid8', grid_size=3)
    layout.initialize(test_beliefs)
    print_layout_info(layout)
    
    # Test Ring
    layout = create_layout('ring', n_agents=6)
    layout.initialize(test_beliefs)
    print_layout_info(layout)
    
    # Test Mesh
    layout = create_layout('mesh', n_agents=4)
    layout.initialize(test_beliefs)
    print_layout_info(layout)
    
    # Test Star
    layout = create_layout('star', n_agents=5)
    layout.initialize(test_beliefs)
    print_layout_info(layout)

