from enum import Enum
from typing import List, Dict
import numpy as np
import networkx as nx

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
    def __init__(self, amb_id: int, location: int, path_cache: Dict = None, 
                 node_to_idx: Dict = None, idx_to_node: Dict = None):
        """
        Initialize a new ambulance.
        
        Args:
            amb_id: Unique identifier for the ambulance
            location: Starting location OSM node ID
            path_cache: Dictionary of precomputed paths and travel times
            node_to_idx: Mapping from node IDs to indices
            idx_to_node: Mapping from indices to node IDs
        """
        self.id = amb_id
        self.location = location  # OSM node ID
        self.status = AmbulanceStatus.IDLE
        self.current_call = None
        self.path = []
        self.destination = None  # OSM node ID
        
        # Store path cache and mappings
        self.path_cache = path_cache
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        
        # Time tracking
        self.busy_until = 0  # Time when ambulance will become available
        self.dispatch_time = 0  # When the ambulance was dispatched
        
        # Statistics
        self.total_idle_time = 0
        self.total_travel_time = 0
        self.calls_responded = 0
        self.total_response_time = 0
        
        # Timeout handling
        self.call_id = None  # To track which call this ambulance is assigned to

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
        # Get the path from the path cache (no error handling for Thesis simplicity)
        return self.path_cache[start_node][end_node]['path']
   

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
        
        # Store call_id if available
        if 'call_id' in call:
            self.call_id = call['call_id']
        
        # Handle case where location is None
        if self.location is None:
            print("[ALERT ALERT]Ambulance location is None, setting to destination")
            # If location is None, set a default location as the base
            self.location = self.destination  # Just as a fallback
            self.path = [self.location]
            # Use a default travel time
            travel_time = 300  # Default 5 minutes
        else:
            self.path = self.get_path(self.location, self.destination)
            travel_time = self.path_cache[self.location][self.destination]['travel_time']
            
        self.busy_until = current_time + travel_time
        self.calls_responded += 1

    def abort_dispatch(self, current_time: int, base_location: int = None):
        """
        Abort a dispatch due to call timeout and start relocating to base.
        
        Args:
            current_time: Current simulation time in seconds
            base_location: Base location to relocate to (if None, will not relocate)
            
        Returns:
            bool: True if dispatch was aborted, False otherwise
        """
        # If we're already dispatched but the call timed out, abort dispatch
        if self.status == AmbulanceStatus.DISPATCHED:
            # Clear call data
            self.current_call = None
            self.destination = None
            self.path = []
            self.call_id = None
            
            # If base location is provided, start relocating
            if base_location is not None:
                self.relocate(base_location, current_time)
            else:
                # Otherwise just set to IDLE and reset busy_until to current time
                self.status = AmbulanceStatus.IDLE
                self.busy_until = current_time
                
            return True
            
        return False

    def arrive_at_scene(self, current_time: float):
        """
        Arrive at scene and begin patient care.
        
        Args:
            current_time: Current simulation time in seconds
        """
        self.status = AmbulanceStatus.ON_SCENE
        self.location = self.destination
        self.path = []
        service_time = int(np.random.normal(600, 120)) # average of 10 minutes, std of 2 minutes
        service_time = max(120, min(900, service_time)) # capped at 15 minutes
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
        
        # Handle case where location is None
        if self.location is None:
            print("[ALERT ALERT]Ambulance location is None, setting to destination")
            # If location is None, use the destination as the starting point
            # This is a fallback to prevent errors
            self.location = self.destination
            self.path = [self.location]
            # Use a default travel time
            travel_time = 300  # Default 5 minutes
        else:
            self.path = self.get_path(self.location, self.destination)
            travel_time = self.path_cache[self.location][self.destination]['travel_time']
            
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
        transfer_time = int(np.random.normal(1200, 300)) # average of 20 minutes, std of 5 minutes
        transfer_time = max(600, min(2100, transfer_time)) # capped at 35 minutes
        self.busy_until = current_time + transfer_time
        # Clear the call_id when arriving at hospital
        self.call_id = None

    def relocate(self, new_location: int, current_time: int):
        """
        Relocate ambulance to a new location (base). Sets the state to RELOCATING.
        
        Args:
            new_location: OSM node ID of the new location (base)
            current_time: Current simulation time in seconds
        """
        # Only start relocation if not already at the base.
        if self.location == new_location:
            # Silently return if already at the destination - no need to print an alert
            return

        self.status = AmbulanceStatus.RELOCATING
        self.destination = int(new_location)
        
        # Handle case where location is None
        if self.location is None:
            print("[ALERT ALERT] Ambulance location is None, setting to destination")
            # If location is None, set location to destination
            self.location = self.destination
            self.path = [self.location]
            travel_time = 0  # No travel needed if we're already there
        else:
            self.path = self.get_path(self.location, self.destination)
            travel_time = self.path_cache[self.location][self.destination]['travel_time']
            
        self.busy_until = current_time + travel_time

    def get_response_time(self, call_node: int) -> float:
        """
        Calculate the response time to a call from the current location.
        
        Args:
            call_node: OSM node ID of the call
            
        Returns:
            float: Estimated response time in seconds
        """
        # Handle case where location is None
        if self.location is None:
            print("[ALERT ALERT] Ambulance location is None, returning default response time of 10 minutes")
            return 600  # Default 10 minutes as a pessimistic estimate
            
        if self.location == call_node:
            return 0
            
        return self.path_cache[self.location][call_node]['travel_time']

    def is_available(self):
        """
        Check if ambulance is available for dispatch.
        
        Returns:
            bool: True if the ambulance is IDLE or RELOCATING
        """
        return self.status in (AmbulanceStatus.IDLE, AmbulanceStatus.RELOCATING)

        
    def reset(self, base_location):
        """Reset ambulance to initial state.
        
        Args:
            base_location: Node ID of the base location
        """
        self.position = base_location
        self.status = AmbulanceStatus.IDLE
        self.destination = None
        self.call_id = None
        self.dispatch_time = 0.0
        self.busy_until = 0.0
