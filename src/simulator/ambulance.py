"""
Ambulance Module
This module defines the core ambulance entities used in the ambulance simulator
"""

from enum import Enum
from typing import List, Dict, Optional, Any
import numpy as np
import networkx as nx

class AmbulanceStatus(Enum):
    """
    Enum representing all possible states of an ambulance.
    
    States:
        IDLE: Ambulance is available for dispatch
        DISPATCHED: Ambulance is traveling to an emergency call
        ON_SCENE: Ambulance is at the incident scene providing service
        TRANSPORT: Ambulance is transporting patient to hospital (decision state for multiple hospitals)
        HOSPITAL: Ambulance is at hospital transferring patient
    """
    IDLE = 0 # also relocation; but when ambulance is available for dispatch and can pick up new patients
    DISPATCHED = 1
    ON_SCENE = 2
    TRANSPORT = 3
    HOSPITAL = 4

class Ambulance:
    """
    Represents an ambulance unit with its current state and properties.
    """
    def __init__(self, amb_id: int, location: int, graph: nx.Graph, distance_matrix: np.ndarray, index_to_node: Dict[int, Any] = None, node_to_index: Dict[Any, int] = None):
        """
        Initialize a new ambulance.
        
        Args:
            amb_id: Unique identifier for the ambulance
            location: Starting location OSM node ID
            graph: NetworkX graph of the road network
            distance_matrix: Matrix of distances between nodes
            index_to_node: Mapping from matrix indices to node IDs
            node_to_index: Mapping from node IDs to matrix indices
        """
        self.id = amb_id
        self.location = location  # OSM node ID
        self.location_idx = node_to_index[str(location)]  # Matrix index
        self.status = AmbulanceStatus.IDLE
        self.current_call = None
        self.path = []
        self.destination = None  # OSM node ID
        self.destination_idx = None  # Matrix index
        
        # Store graph and distance matrix
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.index_to_node = index_to_node or {i: i for i in range(len(distance_matrix))}
        self.node_to_index = node_to_index or {i: i for i in range(len(distance_matrix))}
        
        # Time tracking
        self.busy_until = 0  # Time when ambulance will become available
        self.dispatch_time = 0  # When the ambulance was dispatched
        
        # Statistics
        self.total_idle_time = 0
        self.total_travel_time = 0
        self.calls_responded = 0
        self.total_response_time = 0

    def get_path(self, start_node: int, end_node: int) -> List[int]:
        """
        Get the path between two nodes using the distance matrix.
        
        Args:
            start_node: Starting OSM node ID
            end_node: Ending OSM node ID
            
        Returns:
            List[int]: List of OSM node IDs representing the path
        """
        # Check if nodes are the same
        if start_node == end_node:
            return [start_node]
            
        # Check if nodes exist in the graph
        if start_node not in self.graph or end_node not in self.graph:
            return [start_node, end_node]  # Direct path fallback
            
        # Get path using graph node IDs
        try:
            path = nx.shortest_path(self.graph, source=start_node, target=end_node, weight='length')
        except nx.NetworkXNoPath:
            return [start_node, end_node]  # Direct path fallback
            
        # Verify path distances
        if not self.verify_path_distances(path):
            print(f"Warning: Path verification failed for path from {start_node} to {end_node}")
            
        return path

    def verify_path_distances(self, path: List[int]) -> bool:
        """
        Verify that the sum of path segment lengths matches the distance matrix entry.
        
        Args:
            path: List of OSM node IDs representing a path
            
        Returns:
            bool: True if verification passes, False otherwise
        """
        if len(path) < 2:
            return True
        
        # Get start and end nodes
        start_node = path[0]
        end_node = path[-1]
        
        # Convert OSM node IDs to matrix indices
        start_idx = self.node_to_index[str(start_node)]
        end_idx = self.node_to_index[str(end_node)]
        
        # Get distance from matrix
        matrix_distance = self.distance_matrix[start_idx][end_idx]
        
        # Calculate actual path distance by summing segments
        path_distance = 0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            current_idx = self.node_to_index[str(current_node)]
            next_idx = self.node_to_index[str(next_node)]
            segment_length = self.distance_matrix[current_idx][next_idx]
            path_distance += segment_length
        
        # Allow for small floating point differences
        return abs(matrix_distance - path_distance) < 1e-6

    def dispatch_to_call(self, call: Dict, current_time: int, avg_speed: float = 8.33):
        """
        Dispatch ambulance to a call.
        
        Args:
            call: Dictionary with call information
            current_time: Current simulation time in seconds
            avg_speed: Average speed in m/s (default 8.33 m/s = 30 km/h)
        """
        self.status = AmbulanceStatus.DISPATCHED
        self.current_call = call
        self.dispatch_time = current_time
        
        # Get both node ID and matrix index
        self.destination = int(call['origin_node'])  # OSM node ID
        self.destination_idx = int(call['origin_node_idx'])  # Matrix index
        
        # Get path to destination using OSM node IDs
        self.path = self.get_path(self.location, self.destination)
        
        # Calculate travel time to destination + 60 seconds for crew readiness
        distance = self.distance_matrix[self.location_idx][self.destination_idx]
        travel_time = int(distance / avg_speed) + 60  # Convert to seconds and add 60 seconds
        self.busy_until = current_time + travel_time
        
        self.calls_responded += 1

    def arrive_at_scene(self, current_time: int):
        """
        Arrive at scene and begin patient care.
        
        Args:
            current_time: Current simulation time in seconds
        """
        self.status = AmbulanceStatus.ON_SCENE
        self.location = self.destination  # OSM node ID
        self.location_idx = self.destination_idx  # Matrix index
        self.path = []
        
        # Sample service time from normal distribution (mean=600s=10min, std=120s=2min)
        service_time = int(np.random.normal(600, 120))
        service_time = max(120, min(900, service_time))  # Clamp between 120-900 seconds
        self.busy_until = current_time + service_time

    def begin_transport(self, destination: int, current_time: int, avg_speed: float = 8.33):
        """
        Begin transporting patient to destination.
        
        Args:
            destination: OSM node ID of the destination
            current_time: Current simulation time in seconds
            avg_speed: Average speed in m/s (default 8.33 m/s = 30 km/h)
        """
        self.status = AmbulanceStatus.TRANSPORT
        self.destination = destination  # OSM node ID
        self.destination_idx = self.node_to_index[str(destination)]  # Matrix index
        
        # Get path to destination
        self.path = self.get_path(self.location, self.destination)
        
        # Calculate transport time
        distance = self.distance_matrix[self.location_idx][self.destination_idx]
        travel_time = int(distance / avg_speed)  # Convert to seconds
        self.busy_until = current_time + travel_time

    def arrive_at_hospital(self, current_time: int):
        """
        Arrive at hospital for patient transfer.
        
        Args:
            current_time: Current simulation time in seconds
        """
        self.status = AmbulanceStatus.HOSPITAL
        self.location = self.destination  # OSM node ID
        self.location_idx = self.destination_idx  # Matrix index
        self.path = []
        
        # Sample transfer time from normal distribution (mean=1200s=20min, std=300s=5min)
        transfer_time = int(np.random.normal(1200, 300))
        transfer_time = max(600, min(2100, transfer_time))  # Clamp between 600-2100 seconds
        self.busy_until = current_time + transfer_time

    def relocate(self, new_location: int, current_time: int, avg_speed: float = 8.33):
        """
        Relocate ambulance to a new location.
        
        Args:
            new_location: OSM node ID of the new location
            current_time: Current simulation time in seconds
            avg_speed: Average speed in m/s (default 8.33 m/s = 30 km/h)
        """
        self.status = AmbulanceStatus.IDLE
        self.destination = int(new_location)  # OSM node ID
        self.destination_idx = self.node_to_index[str(new_location)]  # Matrix index
        
        # Get path to new location
        self.path = self.get_path(self.location, self.destination)
        
        # Calculate relocation time: distance/speed
        distance = self.distance_matrix[self.location_idx][self.destination_idx]
        travel_time = int(distance / avg_speed)  # Convert to seconds
        self.busy_until = 0  # No busy_until time since ambulance is IDLE and can pick up new patients