import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
import random
import heapq
import json

import networkx as nx
from typing import List, Dict, Optional

class NearestDispatchPolicy:
    """
    Policy that dispatches the nearest available ambulance to a call,
    based on the shortest travel time computed using NetworkX.
    """
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def get_travel_time(self, from_node: int, to_node: int) -> float:
        """
        Calculate travel time between two nodes using the 'travel_time' edge attribute.
        
        Args:
            from_node: Starting node ID
            to_node: Ending node ID
            
        Returns:
            Travel time in seconds
        """
        return nx.shortest_path_length(self.graph, source=from_node, target=to_node, weight='travel_time')
    
    def select_ambulance(self, call_node: int, available_ambulances: List[Dict], **kwargs) -> Optional[int]:
        """
        Select the nearest available ambulance to dispatch to a call.
        
        Args:
            call_node: Node where the call is located
            available_ambulances: List of available ambulances (each as a dict with at least keys 'id' and 'location')
            
        Returns:
            ID of the selected ambulance, or None if no ambulance is available
        """
        if not available_ambulances:
            return None

        nearest_amb = None
        min_time = float('inf')
        
        for amb in available_ambulances:
            time_to_scene = self.get_travel_time(amb['location'], call_node)
            if time_to_scene < min_time:
                min_time = time_to_scene
                nearest_amb = amb
        
        return nearest_amb['id'] if nearest_amb else None
    

class StaticRelocationPolicy:
    """
    Policy that relocates ambulances back to PFARS HQ when idle,
    using NetworkX to optionally compute travel time if desired.
    """
    def __init__(self, graph: nx.Graph, pfars_hq_node: int):
        self.graph = graph
        self.pfars_hq_node = pfars_hq_node
    
    def get_travel_time_to_hq(self, from_node: int) -> float:
        """
        Calculate travel time from a given node to PFARS HQ using 'travel_time' edge attribute.
        """
        return nx.shortest_path_length(self.graph, source=from_node, target=self.pfars_hq_node, weight='travel_time')
    
    def relocate_ambulances(self, idle_ambulances: List[Dict], busy_ambulances: List[Dict], **kwargs) -> Dict[int, int]:
        """
        Relocate idle ambulances back to PFARS HQ.
        
        Args:
            idle_ambulances: List of idle ambulances (each as a dict with keys 'id' and 'location')
            busy_ambulances: List of busy ambulances (not used here, but provided for compatibility)
            
        Returns:
            Dictionary mapping ambulance IDs to PFARS HQ location.
        """
        relocations = {}
        for amb in idle_ambulances:
            # Only relocate if not already at PFARS HQ
            if amb['location'] != self.pfars_hq_node:
                relocations[amb['id']] = self.pfars_hq_node
        return relocations