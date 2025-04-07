"""
Run Event-Driven Ambulance Simulator with Call Timeouts

This script demonstrates the baseline nearest-ambulance dispatch policy
with call timeouts of approximately 491 seconds (73 + 418).
"""

import os
import sys
import pickle
import json
import pandas as pd

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Project root
sys.path.append(parent_dir)

# Import simulator and policies
from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import NearestDispatchPolicy, StaticRelocationPolicy

def main():
    # Load all required data
    print("Loading Princeton road graph...")
    with open('data/processed/princeton_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    print("Loading path cache...")
    with open('data/matrices/path_cache.pkl', 'rb') as f:
        path_cache = pickle.load(f)
    
    print("Loading node mappings...")
    with open('data/matrices/node_id_to_idx.json', 'r') as f:
        node_to_idx = {int(k): v for k, v in json.load(f).items()}
    
    with open('data/matrices/idx_to_node_id.json', 'r') as f:
        idx_to_node = {int(k): int(v) for k, v in json.load(f).items()}

    # Set up simulation parameters
    call_data_path = 'data/processed/synthetic_calls.csv'
    call_data = pd.read_csv(call_data_path)
    
    num_ambulances = 2
    pfars_node = 241  # PFARS HQ node
    hospital_node = 1293  # Hospital node
    
    # Set up timeout parameters
    call_timeout_mean = 491.0  # seconds (73 sec base + 418 sec avg travel time)
    call_timeout_std = 10.0   # seconds
    
    # Create policies
    dispatch_policy = NearestDispatchPolicy(G)
    relocation_policy = StaticRelocationPolicy(G, base_location=pfars_node)
    
    # Create and run simulator with standard timeouts
    print("\n===== Running Event-Driven Simulator with Call Timeouts =====")
    print(f"Mean timeout: {call_timeout_mean} seconds ({call_timeout_mean/60:.1f} minutes)")
    print(f"Timeout std dev: {call_timeout_std} seconds")
    print(f"Number of ambulances: {num_ambulances}")
    
    # Create simulator with all required data
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=call_data,
        num_ambulances=num_ambulances,
        base_location=pfars_node,
        hospital_node=hospital_node,
        verbose=True,  # Set to True for detailed logs
        call_timeout_mean=call_timeout_mean,
        call_timeout_std=call_timeout_std,
        dispatch_policy=dispatch_policy,
        relocation_policy=relocation_policy,
        path_cache=path_cache,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        manual_mode=False  # Let the policy handle dispatches
    )
    
    # Run simulation
    simulator.run()
    
    # Print final statistics
    print("\n===== Final Statistics =====")
    print(f"Total calls: {simulator.calls_responded + simulator.missed_calls}")
    print(f"Calls responded: {simulator.calls_responded}")
    print(f"Missed calls: {simulator.missed_calls}")
    
    if simulator.response_times:
        avg_rt = sum(simulator.response_times) / len(simulator.response_times)
        print(f"Average response time: {avg_rt/60:.1f} minutes")
        print(f"95th percentile response time: {sorted(simulator.response_times)[int(len(simulator.response_times)*0.95)]/60:.1f} minutes")

if __name__ == "__main__":
    main() 