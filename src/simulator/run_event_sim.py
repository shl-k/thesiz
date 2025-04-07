"""
Run Event-Driven Ambulance Simulator with Call Timeouts

This script demonstrates the new fully event-driven simulator
with call timeouts of approximately 491 seconds (73 + 418).
"""

import os
import sys
import pickle
import pandas as pd


# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Project root
sys.path.append(parent_dir)

# Import simulator
from src.simulator.simulator import AmbulanceSimulator

def main():
    # Load the Princeton road graph
    print("Loading Princeton road graph...")
    with open('data/processed/princeton_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    

    # Set up simulation parameters
    call_data_path = 'data/processed/synthetic_calls.csv'
    # Load call data from CSV
    call_data = pd.read_csv(call_data_path)
    
    num_ambulances = 2
    pfars_node = 241  # PFARS HQ node
    hospital_node = 1293  # Hospital node
    
    # Set up timeout parameters
    call_timeout_mean = 491.0  # seconds (73 sec base + 418 sec avg travel time)
    call_timeout_std = 10.0   # seconds
    
    # Create and run simulator with standard timeouts
    print("\n===== Running Event-Driven Simulator with Call Timeouts =====")
    print(f"Mean timeout: {call_timeout_mean} seconds ({call_timeout_mean/60:.1f} minutes)")
    print(f"Timeout std dev: {call_timeout_std} seconds")
    print(f"Number of ambulances: {num_ambulances}")
    
    # Create simulator
    simulator = AmbulanceSimulator(
        graph=G,
        call_data=call_data,
        num_ambulances=num_ambulances,
        base_location=pfars_node,
        hospital_node=hospital_node,
        verbose=True,  # Set to True for detailed logs
        call_timeout_mean=call_timeout_mean,
        call_timeout_std=call_timeout_std
    )
    
    # Run simulation
    simulator.run()
    '''
    # Run again with shorter timeout parameters for comparison
    shorter_timeout = 300.0  # 5 minutes
    print("\n\n===== Running with Shorter Timeouts =====")
    print(f"Mean timeout: {shorter_timeout} seconds ({shorter_timeout/60:.1f} minutes)")
    
    simulator_short = AmbulanceSimulator(
        graph=G,
        call_data=call_data,
        num_ambulances=num_ambulances,
        base_location=pfars_node,
        hospital_node=hospital_node,
        verbose=False,
        call_timeout_mean=shorter_timeout,
        call_timeout_std=5.0
    )
    
    # Use run() to print statistics automatically
    simulator_short.run()
    
    # Run with more ambulances
    more_ambulances = 5
    print("\n\n===== Running with More Ambulances =====")
    print(f"Mean timeout: {call_timeout_mean} seconds ({call_timeout_mean/60:.1f} minutes)")
    print(f"Number of ambulances: {more_ambulances}")
    
    simulator_more_amb = AmbulanceSimulator(
        graph=G,
        call_data=call_data,
        num_ambulances=more_ambulances,
        base_location=pfars_node,
        hospital_node=hospital_node,
        verbose=False,
        call_timeout_mean=call_timeout_mean,
        call_timeout_std=call_timeout_std
    )
    
    # Use run() to print statistics automatically
    simulator_more_amb.run()
    
    print("\nSimulation comparison complete!")
'''
if __name__ == "__main__":
    main() 