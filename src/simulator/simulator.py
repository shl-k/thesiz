"""
Ambulance Simulator
A simple discrete-time simulator for ambulance operations.
"""

import os
import sys
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional
import pickle

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import ambulance related classes
from ambulance import Ambulance, AmbulanceStatus
from policies import NearestDispatchPolicy, StaticRelocationPolicy

CACHE_PATH = "data/matrices/path_cache.pkl"

with open(CACHE_PATH, "rb") as f:
    PATH_CACHE = pickle.load(f)


class AmbulanceSimulator:
    """
    Discrete-time simulator for ambulance operations.
    
    This simulator:
      1. Loads synthetic call data
      2. Initializes ambulances at base locations
      3. Steps through time in seconds
      4. Handles dispatch and relocation
      5. Tracks basic statistics
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        call_data_path: str,
        num_ambulances: int,
        base_location: int,  # PFARS HQ node ID
        hospital_node: int,
        verbose: bool = False,
        dispatch_policy: Optional[Any] = None,
        relocation_policy: Optional[Any] = None,
    ):
        """
        Initialize the simulator.
        
        Args:
            graph: NetworkX graph of the road network
            call_data_path: Path to call data CSV file
            num_ambulances: Number of ambulances to simulate
            base_location: PFARS HQ node ID
            hospital_node: Hospital node ID
            verbose: Whether to print detailed logs
        """
        # Store parameters
        self.graph = graph
        self.num_ambulances = num_ambulances
        self.base_location = base_location  # PFARS HQ
        self.hospital_node = hospital_node
        self.verbose = verbose

        # Initialize ambulances at PFARS HQ
        self.ambulances = [
            Ambulance(
                amb_id=i,
                location=base_location,  # All ambulances start at PFARS HQ
                graph=graph,
            )
            for i in range(num_ambulances)
        ]
        
        # Load call data and determine number of days
        self.call_data = self.load_call_data(call_data_path)
        self.scenario_days = self.call_data['day'].max()
        self.current_call_index = 0
        
        # Simulation state
        self.current_time = 0
        self.end_time = self.scenario_days * 24 * 3600  # Convert days to seconds
        
        # Statistics
        self.response_times = []
        self.calls_responded = 0
        self.missed_calls = 0

        # Create policy objects using the current graph.
        self.dispatch_policy = dispatch_policy or NearestDispatchPolicy(graph)
        self.relocation_policy = relocation_policy or StaticRelocationPolicy(graph, self.base_location)
    
    def load_call_data(self, call_data_path: str) -> pd.DataFrame:
        """Load and preprocess call data."""
        df = pd.read_csv(call_data_path)
        # Sort by second_of_day for accurate chronological ordering
        df = df.sort_values(['day', 'second_of_day'])
        if self.verbose:
            print(f"Loaded {len(df)} calls over {df['day'].max()} days")
        return df
    
    def step(self) -> bool:
        """
        Advance simulation by one second.
        
        Returns:
            bool: True if simulation should continue, False if it should end.
        """
        if self.current_time >= self.end_time:
            return False
            
        # Calculate current day and second
        current_day = (self.current_time // (24 * 3600)) + 1
        current_second = self.current_time % (24 * 3600)
        
        # Print progress every 50 minutes
        if self.current_time % 3000 == 0:
            print(f"\nSimulation Progress:")
            print(f"Day {current_day}, Time: {current_second//3600:02d}:{(current_second%3600)//60:02d}")
            print(f"Calls processed: {self.calls_responded + self.missed_calls}/{len(self.call_data)}")
            
        # Get calls that should happen at current time
        current_calls = self.call_data[
            (self.call_data['day'] == current_day) & 
            (self.call_data['second_of_day'] == current_second)
        ]
        
        # Process all calls for current time
        for _, call in current_calls.iterrows():
            self.process_call(call.to_dict())
        
        # Update ambulance states
        for ambulance in self.ambulances:
            if ambulance.status != AmbulanceStatus.IDLE and ambulance.busy_until <= self.current_time:
                self.handle_ambulance_state_change(ambulance)
        
        # Relocate idle ambulances (those not at base)
        self.relocate_idle_ambulances()
        
        # Advance time
        self.current_time += 1
        return True
    
    def process_call(self, call: Dict):
        """Process a new emergency call."""
        # Build a list of available ambulances as dicts for the policy.
        available_ambulances = [
            {'id': amb.id, 'location': amb.location}
            for amb in self.ambulances if amb.status == AmbulanceStatus.IDLE
        ]
        
        if not available_ambulances:
            self.missed_calls += 1
            if self.verbose:
                print(f"Time {self.current_time}: No ambulances available for call at {call['origin_node']}")
            return None, None
        
        # Use the dispatch policy to select the nearest ambulance.
        selected_amb_id = self.dispatch_policy.select_ambulance(call['origin_node'], available_ambulances)
        selected_ambulance = next((amb for amb in self.ambulances if amb.id == selected_amb_id), None)
        if selected_ambulance is None:
            return None, None
        
        # Compute path, response time, and distance using cached path info.
        cached_info = PATH_CACHE[selected_ambulance.location][call['origin_node']]
        response_time = cached_info['travel_time']
        distance = cached_info['length']
        # Calculate average speed (m/s) if response_time > 0
        avg_speed = distance / response_time if response_time > 0 else 0

        # Print call information
        print(f"\nCall Information:")
        print(f"Distance: {distance:.2f} meters")
        print(f"Response Time: {response_time:.2f} seconds")
        print(f"Average Speed: {avg_speed:.2f} m/s")
        
        # Dispatch the ambulance.
        selected_ambulance.dispatch_to_call(call, self.current_time)
        self.calls_responded += 1
        
        if self.verbose:
            print(f"Time {self.current_time}: Dispatched ambulance {selected_ambulance.id} to call at {call['origin_node']}")
        
        return response_time, distance


    def handle_ambulance_state_change(self, ambulance: Ambulance):
        """Handle state changes for an ambulance."""
        if ambulance.status == AmbulanceStatus.DISPATCHED:
            ambulance.arrive_at_scene(self.current_time)
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} arrives at scene")
        elif ambulance.status == AmbulanceStatus.ON_SCENE:
            ambulance.begin_transport(self.hospital_node, self.current_time)
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} begins transport to hospital")
        elif ambulance.status == AmbulanceStatus.TRANSPORT:
            ambulance.arrive_at_hospital(self.current_time)
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} arrives at hospital")
        elif ambulance.status == AmbulanceStatus.HOSPITAL:
            ambulance.status = AmbulanceStatus.IDLE
            ambulance.current_call = None
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} becomes idle after hospital transfer")
        elif ambulance.status == AmbulanceStatus.RELOCATING:
            # Once busy_until is reached, complete relocation: update location and mark as idle.
            ambulance.location = ambulance.destination
            ambulance.status = AmbulanceStatus.IDLE
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} completed relocation to {ambulance.location}")

    def relocate_idle_ambulances(self):
        """
        Use the relocation policy to relocate any idle ambulances that are not at base.
        """
        idle_ambulances = [
            amb for amb in self.ambulances if amb.status == AmbulanceStatus.IDLE
        ]
        busy_ambulances = [
            {'id': amb.id, 'location': amb.location}
            for amb in self.ambulances if amb.status != AmbulanceStatus.IDLE
        ]
        
        relocations = self.relocation_policy.relocate_ambulances(
            [{'id': amb.id, 'location': amb.location} for amb in idle_ambulances],
            busy_ambulances
        )
        
        for amb in idle_ambulances:
            if amb.id in relocations:
                new_loc = relocations[amb.id]
                amb.relocate(new_loc, self.current_time)
                if self.verbose:
                    print(f"Time {self.current_time}: Ambulance {amb.id} started relocating to {new_loc}")
