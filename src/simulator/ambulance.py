from enum import Enum
from typing import List, Dict, Optional, Any
import numpy as np
import networkx as nx
import osmnx as ox
import pickle
import os

CACHE_PATH = "data/matrices/path_cache.pkl"

with open(CACHE_PATH, "rb") as f:
    PATH_CACHE = pickle.load(f)


class AmbulanceStatus(Enum):
    """
    Enum representing all possible states of an ambulance.
    
    States:
        IDLE: Ambulance is available for dispatch
        DISPATCHED: Ambulance is traveling to an emergency call
        ON_SCENE: Ambulance is at the incident scene providing service
        TRANSPORT: Ambulance is transporting patient to hospital
        HOSPITAL: Ambulance is at hospital transferring patient
        RELOCATING: Ambulance is returning to base
    """
    IDLE = 0
    DISPATCHED = 1
    ON_SCENE = 2
    TRANSPORT = 3
    HOSPITAL = 4
    RELOCATING = 5

class Ambulance:
    """
    Represents an ambulance unit with its current state and properties.
    """
    def __init__(self, amb_id: int, location: int, graph: nx.Graph):
        """
        Initialize a new ambulance.
        
        Args:
            amb_id: Unique identifier for the ambulance
            location: Starting location OSM node ID
            graph: NetworkX graph of the road network
        """
        self.id = amb_id
        self.location = location  # OSM node ID
        self.status = AmbulanceStatus.IDLE
        self.current_call = None
        self.path = []
        self.destination = None  # OSM node ID
        
        # Store graph 
        self.graph = graph
        
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
        Get the path between two nodes.
        
        Args:
            start_node: Starting OSM node ID
            end_node: Ending OSM node ID
            
        Returns:
            List[int]: List of OSM node IDs representing the path
        """
        if start_node == end_node:
            return [start_node]
        if PATH_CACHE and start_node in PATH_CACHE and end_node in PATH_CACHE[start_node]:
            return PATH_CACHE[start_node][end_node]['path']
    
    def dispatch_to_call(self, call: Dict, current_time: int):
        """
        Dispatch ambulance to a call.
        
        Args:
            call: Dictionary with call information
            current_time: Current simulation time in seconds
        """
        self.status = AmbulanceStatus.DISPATCHED
        self.current_call = call
        self.destination = call['origin_node']
        self.dispatch_time = current_time
        
        self.path = self.get_path(self.location, self.destination)
        travel_time = PATH_CACHE[self.location][self.destination]['travel_time']
        self.busy_until = current_time + travel_time
        self.calls_responded += 1

    def arrive_at_scene(self, current_time: float):
        """
        Arrive at scene and begin patient care.
        
        Args:
            current_time: Current simulation time in seconds
        """
        self.status = AmbulanceStatus.ON_SCENE
        self.location = self.destination
        self.path = []
        service_time = int(np.random.normal(600, 120))
        service_time = max(120, min(900, service_time))
        self.busy_until = current_time + service_time

    def begin_transport(self, destination: int, current_time: float):
        """
        Begin transporting patient to destination.
        
        Args:
            destination: OSM node ID of the destination
            current_time: Current simulation time in seconds
        """
        self.status = AmbulanceStatus.TRANSPORT
        self.destination = destination
        self.path = self.get_path(self.location, self.destination)
        travel_time = PATH_CACHE[self.location][self.destination]['travel_time']
        self.busy_until = current_time + travel_time

    def arrive_at_hospital(self, current_time: int):
        """
        Arrive at hospital for patient transfer.
        
        Args:
            current_time: Current simulation time in seconds
        """
        self.status = AmbulanceStatus.HOSPITAL
        self.location = self.destination
        self.path = []
        transfer_time = int(np.random.normal(1200, 300))
        transfer_time = max(600, min(2100, transfer_time))
        self.busy_until = current_time + transfer_time

    def relocate(self, new_location: int, current_time: int):
        """
        Relocate ambulance to a new location (base). Sets the state to RELOCATING.
        
        Args:
            new_location: OSM node ID of the new location (base)
            current_time: Current simulation time in seconds
        """
        # Only start relocation if not already at the base.
        if self.location == new_location:
            return

        self.status = AmbulanceStatus.RELOCATING
        self.destination = int(new_location)
        self.path = self.get_path(self.location, self.destination)
        travel_time = PATH_CACHE[self.location][self.destination]['travel_time']
        self.busy_until = current_time + travel_time

    # Optionally, you can add a method to update location along the path,
    # but for this static case we'll simply complete relocation in one step.
