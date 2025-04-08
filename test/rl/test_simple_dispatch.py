"""
Test script for the simple dispatch environment.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import json
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.rl.simple_dispatch import SimpleDispatchEnv
from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import NearestDispatchPolicy

# Define a dummy dispatch policy for testing that doesn't use networkx
class DummyDispatchPolicy:
    """A simple dispatch policy that doesn't rely on the graph."""
    def __init__(self, path_cache=None):
        pass  # Don't need anything here
        
    def select_ambulance(self, call_node: int, available_ambulances: List[Dict], 
                         all_ambulances: List[Dict] = None, current_time: float = None) -> Optional[int]:
        """Simply return the first available ambulance."""
        if not available_ambulances:
            return None
        return available_ambulances[0]["id"]

# Define a dummy relocation policy that always sends ambulances to base
class DummyRelocationPolicy:
    """A simple relocation policy that always sends ambulances back to base."""
    def __init__(self, graph=None, base_location=0):
        self.base_location = base_location
        
    def relocate_ambulances(self, available_ambulances: List[Dict], busy_ambulances: List[Dict]) -> Dict[int, int]:
        """Always relocate to the base location."""
        relocations = {}
        for amb in available_ambulances:
            # Only relocate if not already at base
            if amb["location"] != self.base_location:
                relocations[amb["id"]] = self.base_location
                
        return relocations

# Dummy function for loading test data and network
def load_test_data(num_calls=10):
    """Load a small test dataset."""
    # Create a small road network
    G = nx.grid_2d_graph(10, 10)  # 10x10 grid graph
    
    # Convert to integer node IDs
    G = nx.convert_node_labels_to_integers(G)           
    
    # Add edge weights (travel times)
    for u, v in G.edges():
        G[u][v]['travel_time'] = 60  # 1 minute travel time between adjacent nodes
    
    # Create node coordinates dictionary
    node_coords = {n: [n % 10, n // 10] for n in G.nodes()}  # Use lists instead of tuples
    
    # Create a simple call dataset
    calls = pd.DataFrame({
        'second_of_day': [i * 600 for i in range(num_calls)],  # One call every 10 minutes
        'day': [1] * num_calls,
        'origin_node': np.random.choice(list(G.nodes()), num_calls),
        'destination_node': np.random.choice(list(G.nodes()), num_calls),
        'intensity': np.random.uniform(0.5, 1.5, num_calls),  # Random priority
    })
    
    # Create a simple path cache for travel times
    path_cache = {}
    for u in G.nodes():
        path_cache[u] = {}
        for v in G.nodes():
            if u == v:
                path_cache[u][v] = {'travel_time': 0, 'path': [u]}
            else:
                try:
                    path = nx.shortest_path(G, u, v, weight='travel_time')
                    travel_time = sum(G[path[i]][path[i+1]]['travel_time'] for i in range(len(path)-1))
                    path_cache[u][v] = {'travel_time': travel_time, 'path': path}
                except:
                    path_cache[u][v] = {'travel_time': 9999, 'path': []}
    
    # Create fake node mappings
    node_to_idx = {n: n for n in G.nodes()}
    idx_to_node = {n: n for n in G.nodes()}
    
    return G, calls, path_cache, node_to_idx, idx_to_node, node_coords

def main():
    """Run a simple test of the dispatch environment."""
    # Load test data
    G, calls, path_cache, node_to_idx, idx_to_node, node_coords = load_test_data(num_calls=10)
    
    # Create a dispatch policy that doesn't use the graph (to avoid errors)
    # This policy is only used by the simulator when not controlled by the RL agent
    dispatch_policy = DummyDispatchPolicy()
    
    # Create a relocation policy that always returns ambulances to base
    relocation_policy = DummyRelocationPolicy(base_location=0)
    
    # Create the simulator
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=calls,
        num_ambulances=3,
        base_location=0,  # Base is at node 0 
        hospital_node=50,  # Hospital is at node 50 (may fall outside grid, but this is just for testing)
        call_timeout_mean=600,  # 10 minute timeout
        call_timeout_std=60,   # 1 minute std
        dispatch_policy=dispatch_policy,
        relocation_policy=relocation_policy,  # Add the relocation policy
        verbose=True,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node
    )
    
    # Create test lat/lon mapping file for testing
    test_lat_lon_file = "test_lat_lon_mapping.json"
    with open(test_lat_lon_file, 'w') as f:
        # Convert node_coords (which has int keys) to use string keys for JSON
        json_node_coords = {str(k): v for k, v in node_coords.items()}
        json.dump(json_node_coords, f)
    
    print(f"Created test lat/lon mapping file with {len(node_coords)} nodes")
    
    # Create the environment
    env = SimpleDispatchEnv(
        simulator=simulator,
        lat_lon_file=test_lat_lon_file,
        verbose=True,
        max_steps=50,  # Just enough steps to handle our 10 test calls
        render_mode="human"
    )
    
    # Run the test episode
    run_test_episode(env, simulator, "Simple Dispatch Test")
    
    # Clean up test file
    os.remove(test_lat_lon_file)
    print(f"Removed test lat/lon mapping file")

def run_test_episode(env, simulator, test_name):
    """Run a test episode with the given environment."""
    obs, info = env.reset()
    print(f"\n--- {test_name} ---")
    print(f"Initial observation: {obs}")
    
    done = False
    total_reward = 0
    
    while not done:
        # Choose a random action (but with higher chance of valid ambulance dispatch)
        # This gives a better test of the environment
        no_dispatch_prob = 0.3
        if np.random.random() < no_dispatch_prob:
            # Choose "no dispatch" action
            action = env.num_ambulances
            action_type = "No dispatch"
        else:
            # Choose a specific ambulance (0 to num_ambulances-1)
            action = np.random.randint(0, env.num_ambulances)
            action_type = f"Dispatch ambulance {action}"
        
        print(f"\nTaking action: {action} ({action_type})")
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the environment
        env.render()
        
        print(f"Reward: {reward}")
        print(f"Total reward: {total_reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        
        done = terminated or truncated
        if done:
            break
    
    # Print final statistics
    print(f"\nFinal statistics ({test_name}):")
    print(f"Total reward: {total_reward}")
    print(f"Steps: {env.steps}")
    print(f"Calls responded: {simulator.calls_responded}")
    print(f"Missed calls: {simulator.missed_calls}")
    print(f"Timed out calls: {simulator.timed_out_calls}")

if __name__ == "__main__":
    main() 