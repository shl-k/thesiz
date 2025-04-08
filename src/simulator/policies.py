import json
from typing import List, Dict, Optional

import networkx as nx
import numpy as np
from stable_baselines3 import PPO

# Global setting for verbose output from policies
verbose_global = False

# ---------------------------------------------------------------------------
#  Baseline Policies
# ---------------------------------------------------------------------------

class NearestDispatchPolicy:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def _travel_time(self, from_node: int, to_node: int) -> float:
        return nx.shortest_path_length(
            self.graph,
            source=int(from_node),
            target=int(to_node),
            weight="travel_time"
        )

    def select_ambulance(self, available_ambulances: List[Dict], all_ambulances: List[Dict] = None, current_time: float = None, current_call: Dict = None) -> Optional[int]:
        """Select the nearest available ambulance to the current call."""
        if not available_ambulances or not current_call:
            return None
            
        call_node = current_call["origin_node"]
        return min(
            available_ambulances,
            key=lambda amb: self._travel_time(amb["location"], call_node)
        )["id"]


class StaticRelocationPolicy:
    def __init__(self, graph: nx.Graph, base_location: int):
        self.graph = graph
        self.base_location = base_location

    def select_relocation_target(self, ambulance: Dict, **kwargs) -> int:
        """Always relocate to the base location."""
        return self.base_location
        
    def relocate_ambulances(self, available_ambulances: List[Dict], busy_ambulances: List[Dict]) -> Dict[int, int]:
        """
        Process ambulances that need relocation.
        Returns a dictionary mapping ambulance IDs to target location nodes.
        """
        relocations = {}
        for amb in available_ambulances:
            # Only relocate if not already at base
            if amb["location"] != self.base_location:
                relocations[amb["id"]] = self.base_location
                
        return relocations
