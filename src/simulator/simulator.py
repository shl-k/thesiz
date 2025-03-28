"""
Ambulance Simulator
A simple discrete-time simulator for ambulance operations.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Any
import time
import json

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import ambulance related classes
from ambulance import Ambulance, AmbulanceStatus

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
        distance_matrix: np.ndarray,
        num_ambulances: int,
        base_location: int,  # PFARS HQ node ID
        hospital_node: int,
        index_to_node: Dict[int, Any] = None,
        node_to_index: Dict[Any, int] = None,
        avg_speed: float = 8.33,  # Average speed in m/s (default: 30 km/h)
        verbose: bool = False,
        preserve_nodes: List[int] = None
    ):
        """
        Initialize the simulator.
        
        Args:
            graph: NetworkX graph of the road network
            call_data_path: Path to call data CSV file
            distance_matrix: Matrix of distances between nodes
            num_ambulances: Number of ambulances to simulate
            base_location: PFARS HQ node ID
            hospital_node: Hospital node ID
            index_to_node: Mapping from matrix indices to node IDs
            node_to_index: Mapping from node IDs to matrix indices
            avg_speed: Average speed in meters per second (m/s) for ambulance travel.
                     Default is 8.33 m/s (30 km/h), but can be adjusted based on:
                     - Traffic conditions
                     - Time of day
                     - Weather conditions
                     - Emergency vs non-emergency travel
            verbose: Whether to print detailed logs
            preserve_nodes: List of node IDs that should be preserved during sparsification
        """
        # Store parameters
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.num_ambulances = num_ambulances
        self.base_location = base_location  # PFARS HQ
        self.hospital_node = hospital_node
        self.avg_speed = avg_speed
        self.verbose = verbose
        self.node_to_index = node_to_index  # Store the node to index mapping
        self.preserve_nodes = preserve_nodes or []  # Store nodes to preserve
        
        # Verify critical nodes
        print("\n=== Verifying Critical Nodes ===")
        print(f"PFARS HQ node ID: {base_location}")
        print(f"Hospital node ID: {hospital_node}")
        
        # Check if nodes exist in graph
        print("\nChecking if nodes exist in graph:")
        print(f"PFARS HQ in graph: {base_location in graph}")
        print(f"Hospital in graph: {hospital_node in graph}")
        
        # Check node mappings
        print("\nChecking node mappings:")
        print(f"PFARS HQ in node_to_index: {base_location in node_to_index}")
        print(f"Hospital in node_to_index: {hospital_node in node_to_index}")
        
        if base_location in node_to_index and hospital_node in node_to_index:
            pfars_idx = node_to_index[base_location]
            hospital_idx = node_to_index[hospital_node]
            print(f"\nMatrix indices:")
            print(f"PFARS HQ matrix index: {pfars_idx}")
            print(f"Hospital matrix index: {hospital_idx}")
            
            # Check distance between nodes
            distance = distance_matrix[pfars_idx][hospital_idx]
            print(f"\nDistance between PFARS HQ and Hospital:")
            print(f"Distance: {distance:.2f} meters")
            print(f"Distance in miles: {distance/1609.34:.2f} miles")
            print(f"Expected travel time at 30 mph: {distance/(8.33*3600):.1f} minutes")
            
            # Verify this distance is reasonable (should be around 2-3 miles)
            if distance < 1000 or distance > 10000:  # Less than 1km or more than 10km
                print("\nWARNING: Distance between PFARS HQ and Hospital seems unreasonable!")
                print("This might indicate incorrect node mappings or graph sparsification issues.")
        
        # Initialize ambulances at PFARS HQ
        self.ambulances = [
            Ambulance(
                amb_id=i,
                location=base_location,  # All ambulances start at PFARS HQ
                graph=graph,
                distance_matrix=distance_matrix,
                index_to_node=index_to_node,
                node_to_index=node_to_index
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
            bool: True if simulation should continue, False if it should end
        """
        # Check if simulation should end
        if self.current_time >= self.end_time:
            return False
            
        # Calculate current day and second
        current_day = (self.current_time // (24 * 3600)) + 1
        current_second = self.current_time % (24 * 3600)
        
        # Print progress every 5 minutes
        if self.current_time % 300 == 0:  # 300 seconds = 5 minutes
            print(f"\nSimulation Progress:")
            print(f"Day {current_day}, Time: {current_second//3600:02d}:{(current_second%3600)//60:02d}")
            print(f"Calls processed: {self.calls_responded + self.missed_calls}/{len(self.call_data)}")
            print(f"Active ambulances: {sum(1 for amb in self.ambulances if amb.status != AmbulanceStatus.IDLE)}/{self.num_ambulances}")
            
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
        
        # Advance time
        self.current_time += 1
        return True
    
    def process_call(self, call: Dict):
        """Process a new emergency call."""
        # Find available ambulance
        available_ambulances = [
            amb for amb in self.ambulances 
            if amb.status == AmbulanceStatus.IDLE
        ]
        
        if not available_ambulances:
            self.missed_calls += 1
            if self.verbose:
                print(f"Time {self.current_time}: No ambulances available for call at {call['origin_node']}")
            return None, None
        
        # Get matrix index for call node
        call_node = str(call['origin_node'])  # Convert to string to match JSON format
        call_node_idx = self.node_to_index[call_node]
        
        if self.verbose:
            print(f"\nDebug - Call details:")
            print(f"call_node: {call_node}")
            print(f"call_node_idx: {call_node_idx}")
            
            # Print debug info for each ambulance
            for amb in available_ambulances:
                print(f"\nAmbulance {amb.id}:")
                print(f"amb.location: {amb.location}")
                print(f"amb.location_idx: {amb.location_idx}")
                print(f"Distance: {amb.distance_matrix[amb.location_idx][call_node_idx]}")
        
        selected_ambulance = min(
            available_ambulances,
            key=lambda amb: amb.distance_matrix[amb.location_idx][call_node_idx]
        )
        
        # Calculate distance and response time
        distance = selected_ambulance.distance_matrix[selected_ambulance.location_idx][call_node_idx]
        response_time = distance / self.avg_speed  # in seconds
        
        # Update call with correct matrix index
        call['origin_node_idx'] = call_node_idx
        
        # Dispatch ambulance
        selected_ambulance.dispatch_to_call(call, self.current_time, self.avg_speed)
        self.calls_responded += 1
        
        if self.verbose:
            print(f"Time {self.current_time}: Dispatched ambulance {selected_ambulance.id} to call at {call['origin_node']}")
        
        return response_time, distance
    
    def handle_ambulance_state_change(self, ambulance: Ambulance):
        """Handle state changes for an ambulance."""
        if ambulance.status == AmbulanceStatus.DISPATCHED:
            # Arrive at scene
            ambulance.arrive_at_scene(self.current_time)
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} arrives at scene")
                
        elif ambulance.status == AmbulanceStatus.ON_SCENE:
            # Begin transport to hospital
            ambulance.begin_transport(self.hospital_node, self.current_time, self.avg_speed)
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} begins transport to hospital")
                
        elif ambulance.status == AmbulanceStatus.TRANSPORT:
            # Arrive at hospital
            ambulance.arrive_at_hospital(self.current_time)
            if self.verbose: 
                print(f"Time {self.current_time}: Ambulance {ambulance.id} arrives at hospital")
                
        elif ambulance.status == AmbulanceStatus.HOSPITAL:
            # Ambulance becomes idle after hospital transfer
            ambulance.status = AmbulanceStatus.IDLE
            ambulance.current_call = None
            if self.verbose:
                print(f"Time {self.current_time}: Ambulance {ambulance.id} becomes idle after hospital transfer")
    
    def run(self):
        """Run the simulation and return results."""
        print("\nStarting simulation...")
        
        # Load and sort calls by timestamp
        calls_df = pd.read_csv(self.call_data_path)
        calls_df['timestamp'] = pd.to_datetime(calls_df['timestamp'])
        calls_df = calls_df.sort_values('timestamp')
        
        # Initialize simulation time
        current_time = calls_df['timestamp'].min()
        end_time = calls_df['timestamp'].max()
        
        # Initialize results tracking
        total_calls = len(calls_df)
        calls_responded = 0
        missed_calls = 0
        total_response_time = 0
        total_distance = 0
        
        # Process each call
        for _, call in calls_df.iterrows():
            # Update simulation time
            current_time = call['timestamp']
            
            # Process the call
            response_time, distance = self.process_call(call)
            
            # Update statistics
            if response_time is not None:
                calls_responded += 1
                total_response_time += response_time
                total_distance += distance
            else:
                missed_calls += 1
            
            # Print progress
            if current_time.minute % 5 == 0:
                print(f"\nSimulation Progress:")
                print(f"Day {current_time.day}, Time: {current_time.strftime('%H:%M')}")
                print(f"Calls processed: {calls_responded + missed_calls}/{total_calls}")
                print(f"Active ambulances: {sum(1 for amb in self.ambulances if amb.state != 'idle')}/{len(self.ambulances)}")
        
        # Calculate final statistics
        avg_response_time = total_response_time / calls_responded if calls_responded > 0 else 0
        avg_distance = total_distance / calls_responded if calls_responded > 0 else 0
        
        # Save detailed results
        results = {
            'total_calls': total_calls,
            'calls_responded': calls_responded,
            'missed_calls': missed_calls,
            'avg_response_time': avg_response_time,
            'avg_distance': avg_distance,
            'response_rate': calls_responded / total_calls if total_calls > 0 else 0
        }
        
        # Save results to JSON
        os.makedirs('outputs/results', exist_ok=True)
        with open('outputs/results/simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nSimulation completed!")
        print(f"Total calls: {total_calls}")
        print(f"Calls responded: {calls_responded}")
        print(f"Missed calls: {missed_calls}")
        print(f"Response rate: {results['response_rate']*100:.1f}%")
        
        return results 