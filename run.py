"""
Run the ambulance simulation using pre-generated graph and data.
"""

import json
import networkx as nx
import numpy as np
from src.simulator.simulator import AmbulanceSimulator
import pickle

def main():
    #load gaph
    with open('data/processed/princeton_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    # Critical nodes (these should be consistent from princeton_data_prep.py)
    pfars_node = 241  # PFARS HQ
    hospital_node = 1293  # Hospital
    
    # Create simulator
    print("\nCreating simulator...")
    simulator = AmbulanceSimulator(
        graph=G,
        call_data_path='data/processed/synthetic_calls.csv',
        num_ambulances=2,  # Start with 2 ambulances
        base_location=pfars_node,
        hospital_node=hospital_node,
        verbose=True
    )
    
    # Run simulation
    print("\nStarting simulation...")
    while simulator.step():
        pass
    
    # Print final statistics
    print("\nSimulation completed!")
    print(f"Total calls: {simulator.calls_responded + simulator.missed_calls}")
    print(f"Calls responded: {simulator.calls_responded}")
    print(f"Missed calls: {simulator.missed_calls}")
    print(f"Response rate: {simulator.calls_responded/(simulator.calls_responded + simulator.missed_calls)*100:.1f}%")

if __name__ == "__main__":
    main() 