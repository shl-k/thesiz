from enum import Enum
from typing import List, Dict, Optional
import numpy as np

class AmbulanceStatus(Enum):
    """ Enum representing all possible states of an ambulance. """
    IDLE = 0  # Available for dispatch / relocation
    DISPATCHED = 1  # En‑route to incident
    ON_SCENE = 2  # Treating patient on‑scene
    TRANSPORT = 3  # En‑route to hospital
    HOSPITAL = 4  # Off‑loading patient
    RELOCATING = 5  # Repositioning (return‑to‑base or policy target)

class Ambulance:
    """
    Represents an ambulance unit with its current state and properties.
    """
    def __init__(self, 
                 amb_id: int, 
                 location: int, 
                 *,
                 path_cache: Dict = None, 
                 node_to_idx: Dict = None, 
                 idx_to_node: Dict = None) -> None:
    
        self.id = amb_id
        self.location = location  # make sure give it an OSM node ID
        self.status = AmbulanceStatus.IDLE

        #Call specific
        self.current_call: Optional[Dict] = None
        self.call_id: Optional[int] = None
      

        #Path specific
        self.path_cache: Dict = path_cache
        self.node_to_idx: Dict = node_to_idx
        self.idx_to_node: Dict = idx_to_node
        self.path: List[int] = []
        self.destination: Optional[int] = None

        # Time tracking
        self.busy_until = 0  # Time when ambulance will become available
        self.dispatch_time = 0  # When the ambulance was dispatched
        
        # Statistics
        self.calls_responded = 0
        self.total_response_time = 0
        
    # ---------------------------------------------------------------------
    # State transitions (DISPATCH CYCLE)
    # ---------------------------------------------------------------------

    def dispatch_to_call(self, call: Dict, current_time: int):
        """Move from IDLE/RELOCATING → DISPATCHED."""
        self.status = AmbulanceStatus.DISPATCHED
        self.current_call = call
        self.destination = call['origin_node']
        self.call_id = call['call_id']
        self.dispatch_time = current_time
        
        self.path = self.get_path(self.location, self.destination)
        travel_time = self.path_cache[self.location][self.destination]['travel_time']
        self.busy_until = current_time + travel_time
        self.calls_responded += 1

    def abort_dispatch(self, current_time: int, base_location: int = None):
        """Abort an en‑route ambulance (call timed‑out)."""
        # If we're already dispatched but the call timed out, abort dispatch
        self.current_call = None
        self.destination = None
        self.path = []
        self.call_id = None
            
        if base_location is not None and self.location != base_location:
            self.relocate(base_location, current_time)
        else:
            self.status = AmbulanceStatus.IDLE
            self.busy_until = current_time

    def arrive_at_scene(self, current_time: float):
        """Move from DISPATCHED → ON_SCENE."""
        self.status = AmbulanceStatus.ON_SCENE
        self.location = self.destination
        self.destination = None
        self.path = []
        # average of 10 minutes, std of 2 minutes
        service_time = int(np.clip(np.random.normal(600, 120), 120, 900)) 
        self.busy_until = current_time + service_time

    def begin_transport(self, destination: int, current_time: float):
        """Move from ON_SCENE → TRANSPORT."""
        self.status = AmbulanceStatus.TRANSPORT
        self.destination = destination
        self.path = self.get_path(self.location, self.destination)
        travel_time = self.path_cache[self.location][self.destination]['travel_time']
        self.busy_until = current_time + travel_time

    def arrive_at_hospital(self, current_time: int):
        """Move from TRANSPORT → HOSPITAL."""
        self.status = AmbulanceStatus.HOSPITAL
        self.location = self.destination
        self.destination = None
        self.path = []
        # average of 20 minutes, std of 5 minutes
        transfer_time = int(np.clip(np.random.normal(1200, 300), 600, 2100)) 
        self.busy_until = current_time + transfer_time
        # Clear the call_id when arriving at hospital
        self.call_id = None
        self.current_call = None

    def relocate(self, new_location: int, current_time: int):
        """Move from HOSPITAL → RELOCATING."""

        if self.location == new_location:
            return # Well, that was fast.

        self.status = AmbulanceStatus.RELOCATING
        self.destination = int(new_location)
        self.path = self.get_path(self.location, self.destination)
        travel_time = self.path_cache[self.location][self.destination]['travel_time']
        self.busy_until = current_time + travel_time

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def get_path(self, start_node: int, end_node: int) -> List[int]:
        """ Get the path between two nodes. """
        if start_node == end_node:
            return [start_node]
        return self.path_cache[start_node][end_node]['path']   
    
    def get_response_time(self, call_node: int) -> float:
        if self.location == call_node:
            return 0  
        return self.path_cache[self.location][call_node]['travel_time']

    def is_available(self):
        return self.status in (AmbulanceStatus.IDLE, AmbulanceStatus.RELOCATING)

        
    def reset(self, base_location):
        """Hard reset between simulation days."""
        self.location = base_location
        self.status = AmbulanceStatus.IDLE
        self.destination = None
        self.call_id = None
        self.current_call = None
        self.path = []
        self.busy_until = 0.0
