"""
Simple RL environment for ambulance relocation.
This environment makes relocation decisions after an ambulance completes service.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from src.simulator.simulator import AmbulanceSimulator, EventType
from src.simulator.ambulance import Ambulance, AmbulanceStatus

class SimpleRelocationEnv(gym.Env):
    """
    A simple RL environment for ambulance relocation decisions.
    
    This environment:
    1. Presents the agent with observations when an ambulance completes service
       (hospital transfer or on-scene service when no transport needed)
    2. Accepts actions to relocate the ambulance to a specific lat/lon
    3. Provides rewards based on coverage and future response times
    4. Includes time of day in the observation space for time-based decision making
    
    Observation space:
    - ambulances: Position coordinates, status, and busy time for each ambulance
    - completed_ambulance: Position coordinates of the ambulance that just completed service
    - time_of_day: Normalized time of day (0-1 representing 0-24 hours)
    
    Action space:
    - lat/lon coordinates representing the relocation target
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        simulator: AmbulanceSimulator,
        node_to_lat_lon_file: str = 'data/matrices/node_to_lat_lon.json',
        lat_lon_to_node_file: str = 'data/matrices/lat_lon_to_node.json',
        closest_n_nodes: int = 100,  # Number of closest nodes to consider for relocation
        verbose: bool = False,
        max_steps: int = 1000000,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the environment.
        
        Args:
            simulator: The ambulance simulator
            node_to_lat_lon_file: Path to JSON file with node to lat/lon mapping
            closest_n_nodes: Number of closest nodes to consider for relocation
            verbose: Whether to print verbose output
            max_steps: Maximum number of steps per episode
            reward_scale: Scaling factor for rewards
            render_mode: How to render the environment (currently only 'human' is supported)
        """
        self.simulator = simulator
        self.verbose = verbose
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        self.closest_n_nodes = closest_n_nodes
        
        # Load node coordinates from file
        if os.path.exists(node_to_lat_lon_file):
            if verbose:
                print(f"Loading node coordinates from {node_to_lat_lon_file}")
            with open(node_to_lat_lon_file, 'r') as f:
                raw_coords = json.load(f)
                # Convert string keys to integers
                self.node_coords = {int(k): tuple(v) for k, v in raw_coords.items()}
            if verbose:
                print(f"Loaded coordinates for {len(self.node_coords)} nodes")
                
        # Load reverse mapping from lat/lon to node ID
        if os.path.exists(lat_lon_to_node_file):
            if verbose:
                print(f"Loading lat/lon mapping from {lat_lon_to_node_file}")
            with open(lat_lon_to_node_file, 'r') as f:
                self.latlon_to_node = json.load(f)
        else:
            raise FileNotFoundError(f"Lat/lon to node mapping file {lat_lon_to_node_file} not found")
                
        # Create numpy array of all node coordinates for fast nearest neighbor calculations
        self.node_ids = list(self.node_coords.keys())
        self.node_coords_array = np.array([self.node_coords[node] for node in self.node_ids])
        
        # Get path cache from simulator
        self.path_cache = simulator.path_cache
        
        # State tracking
        self.completed_ambulance = None
        self.steps = 0
        self.done = False
        
        # Number of ambulances from simulator
        self.num_ambulances = len(simulator.ambulances)
        
        # Set up observation space
        self.observation_space = spaces.Dict({
            'ambulances': spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self.num_ambulances, 4),  # lat, lon, status, busy_until
                dtype=np.float32
            ),
            'completed_ambulance': spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(3,),  # lat, lon, id
                dtype=np.float32
            ),
            'time_of_day': spaces.Box(
                low=0, 
                high=1, 
                shape=(1,),  # normalized time of day (0-1 for 0-24 hours)
                dtype=np.float32
            ),
        })
        
        # Action space: lat/lon coordinates for relocation
        self.action_space = spaces.Box(
            low=np.array([min(self.node_coords_array[:, 0]), min(self.node_coords_array[:, 1])]),
            high=np.array([max(self.node_coords_array[:, 0]), max(self.node_coords_array[:, 1])]),
            shape=(2,),
            dtype=np.float32
        )
    
    def get_node_coords(self, node_id):
        """Get the coordinates for a node."""
        return self.node_coords[node_id]
    
    def get_nearest_node(self, lat, lon):
        """
        Find the nearest node to the given lat/lon coordinates using path cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            int: Node ID of the nearest node
        """
        # Check if exact coordinates exist in the mapping
        rounded_lat, rounded_lon = round(lat, 6), round(lon, 6)
        if (rounded_lat, rounded_lon) in self.latlon_to_node:
            return self.latlon_to_node[(rounded_lat, rounded_lon)]
        
        # If exact coordinates not found, find the nearest node using coordinates
        # First get coordinates for all nodes
        node_distances = []
        for node_id, (node_lat, node_lon) in self.node_coords.items():
            # Calculate straight-line distance
            dist = ((node_lat - lat) ** 2 + (node_lon - lon) ** 2) ** 0.5
            node_distances.append((node_id, dist))
        
        # Sort by distance and take top 10 nodes
        node_distances.sort(key=lambda x: x[1])
        nearest_nodes = [node_id for node_id, _ in node_distances[:10]]
        
        # Among these nodes, find the one with minimum travel time from PFARS
        base_node = 241  # PFARS location
        min_travel_time = float('inf')
        nearest_node = None
        
        for node_id in nearest_nodes:
            if node_id in self.path_cache[base_node]:
                travel_time = self.path_cache[base_node][node_id]['travel_time']
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    nearest_node = node_id
        
        return nearest_node if nearest_node is not None else 241  # Fallback to base
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility (optional)
            options: Additional options for reset (not used)
            
        Returns:
            Initial observation and empty info dict
        """
        # Reset simulator
        self.simulator.initialize()
        
        # Reset state
        self.completed_ambulance = None
        self.steps = 0
        self.done = False
        
        # Get initial observation
        obs = self._build_observation()
        
        return obs, {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Lat/lon coordinates for relocation
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Initialize reward
        reward = 0.0
        
        # Convert lat/lon to node ID
        lat, lon = action
        target_node = self.get_nearest_node(lat, lon)
        
        # Get the ambulance that just completed service
        amb_idx = int(self.completed_ambulance["id"])
        ambulance = self.simulator.ambulances[amb_idx]
        
        # Calculate reward before relocation (for comparison)
        pre_relocation_response_time = self._calculate_average_response_time(ambulance.location)
        
        # Perform relocation
        ambulance.relocate(target_node, self.simulator.current_time)
        
        # Process simulator steps until we find an ambulance that completed service
        # or until simulation is done
        while True:
            # Check if we're done
            if self.done or self.steps >= self.max_steps:
                self.done = True
                return self._build_observation(), reward, True, False, {}
            
            # Process next event in simulator
            event_time, event_type, event_data = self.simulator.step()
            
            # Check if simulation is done
            if event_type is None:
                self.done = True
                return self._build_observation(), reward, True, False, {}
            
            # Handle service completion events (ambulance finishes at hospital or on-scene)
            if event_type == EventType.AMBULANCE_FREE:
                amb_id = event_data.get("ambulance_id")
                amb = next((a for a in self.simulator.ambulances if a.id == amb_id), None)
                
                if amb and (amb.status == AmbulanceStatus.IDLE):
                    # Ambulance has just become free
                    self.completed_ambulance = {
                        "id": amb.id,
                        "location": amb.location,
                        "coords": self.get_node_coords(amb.location)
                    }
                    
                    # Calculate reward after relocation
                    post_relocation_response_time = self._calculate_average_response_time(target_node)
                    
                    # Base reward for successful relocation
                    reward = 100
                    
                    # Response time improvement reward
                    response_time_improvement = pre_relocation_response_time - post_relocation_response_time
                    reward += response_time_improvement * 0.5  # Scale factor to balance with other rewards
                    
                    # Coverage reward based on distance to other ambulances
                    coverage_reward = self._calculate_coverage_reward(target_node)
                    reward += coverage_reward
                    
                    if self.verbose:
                        print(f"Relocation reward breakdown:")
                        print(f"  Base reward: +100")
                        print(f"  Response time improvement: +{response_time_improvement * 0.5:.1f}")
                        print(f"  Coverage reward: +{coverage_reward:.1f}")
                        print(f"  Total: {reward:.1f}")
                    
                    # Increment step counter
                    self.steps += 1
                    
                    # Return updated observation
                    return self._build_observation(), reward, False, False, {}
            
            # For other events, just continue processing
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            # Time information
            total_seconds = int(self.simulator.current_time)
            hours = total_seconds // 3600 % 24
            minutes = (total_seconds % 3600) // 60
            time_str = f"{hours:02d}:{minutes:02d}"
            
            # Ambulance information
            print(f"Time: {time_str}")
            print(f"Step: {self.steps}")
            
            if self.completed_ambulance:
                amb_id = self.completed_ambulance["id"]
                amb_loc = self.completed_ambulance["location"]
                lat, lon = self.completed_ambulance["coords"]
                print(f"Ambulance {amb_id} just completed service at node {amb_loc} ({lat:.4f}, {lon:.4f})")
                
            print("\nAmbulance status:")
            for i, amb in enumerate(self.simulator.ambulances):
                status_str = amb.status.name
                loc = amb.location
                lat, lon = self.get_node_coords(loc) if loc in self.node_coords else (0, 0)
                print(f"  Ambulance {amb.id}: {status_str} at node {loc} ({lat:.4f}, {lon:.4f})")
                
            print("\n")
    
    def _calculate_average_response_time(self, location):
        """
        Calculate average response time from a location to all active calls.
        
        Args:
            location: Node ID to calculate response time from
            
        Returns:
            float: Average response time in seconds
        """
        total_time = 0
        num_calls = len(self.simulator.active_calls)
        
        if num_calls == 0:
            return 0
            
        for call in self.simulator.active_calls.values():
            call_node = call["origin_node"]
            if call_node in self.path_cache[location]:
                travel_time = self.path_cache[location][call_node]['travel_time']
                total_time += travel_time
                
        return total_time / num_calls if num_calls > 0 else 0
        
    def _calculate_coverage_reward(self, location):
        """
        Calculate coverage reward based on positioning relative to other ambulances
        and active calls.
        
        Args:
            location: Node ID to calculate coverage for
            
        Returns:
            float: Coverage reward
        """
        reward = 0.0
        
        # Get coordinates of the location
        if location not in self.node_coords:
            return 0.0
            
        loc_coords = np.array(self.get_node_coords(location))
        
        # Calculate distances to other ambulances
        for amb in self.simulator.ambulances:
            if amb.location in self.node_coords and amb.location != location:
                amb_coords = np.array(self.get_node_coords(amb.location))
                distance = np.sqrt(np.sum((loc_coords - amb_coords)**2))
                # Reward for being not too close and not too far from other ambulances
                if 1000 < distance < 5000:  # 1-5km range
                    reward += 50
                    
        # Calculate distances to active calls
        for call in self.simulator.active_calls.values():
            call_node = call["origin_node"]
            if call_node in self.node_coords:
                call_coords = np.array(self.get_node_coords(call_node))
                distance = np.sqrt(np.sum((loc_coords - call_coords)**2))
                # Reward for being close to calls
                if distance < 3000:  # Within 3km
                    reward += 100 * (1 - distance/3000)  # Linear scaling
                    
        return reward
    
    def _build_observation(self):
        """
        Build the observation dictionary.
        
        Returns:
            dict: Observation dictionary
        """
        # Ambulance information
        ambulances_obs = []
        for amb in self.simulator.ambulances:
            if amb.location in self.node_coords:
                lat, lon = self.get_node_coords(amb.location)
            else:
                # Use default coordinates if location not in mapping
                lat, lon = 0.0, 0.0
                
            status_value = amb.status.value
            
            # Calculate remaining busy time
            if hasattr(amb, 'busy_until'):
                busy_time = max(0, amb.busy_until - self.simulator.current_time)
            else:
                busy_time = 0.0
                
            ambulances_obs.append([lat, lon, status_value, busy_time])
            
        # Completed ambulance information (if available)
        if self.completed_ambulance:
            lat, lon = self.completed_ambulance["coords"]
            amb_id = self.completed_ambulance["id"]
            completed_amb_obs = [lat, lon, amb_id]
        else:
            # Default values if no ambulance has completed service yet
            completed_amb_obs = [0.0, 0.0, 0.0]
            
        # Normalized time of day (0-1)
        time_of_day = (self.simulator.current_time % (24 * 3600)) / (24 * 3600)
        
        return {
            'ambulances': np.array(ambulances_obs, dtype=np.float32),
            'completed_ambulance': np.array(completed_amb_obs, dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32)
        } 