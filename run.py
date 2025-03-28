"""
Run the ambulance simulation using pre-generated graph and data.
"""
import os
import json
import networkx as nx
import numpy as np
from src.simulator.simulator import AmbulanceSimulator
import pickle

def main():
    # Load the pre-generated graph and data
    print("\nLoading pre-generated data...")
    
    # Load node mappings
    with open('data/matrices/node_to_index.json', 'r') as f:
        node_to_index = json.load(f)
    with open('data/matrices/index_to_node.json', 'r') as f:
        index_to_node = json.load(f)
    
    # Load distance matrix
    distance_matrix = np.load('data/matrices/distance_matrix.npy')
    
    # Load graph
    with open('data/processed/princeton_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    # Critical nodes (these should be consistent from princeton_data_prep.py)
    pfars_node = "104040325"  # PFARS HQ
    hospital_node = "1770476466"  # Hospital
    
    # Create simulator
    print("\nCreating simulator...")
    simulator = AmbulanceSimulator(
        graph=G,
        call_data_path='data/processed/synthetic_calls.csv',
        distance_matrix=distance_matrix,
        num_ambulances=2,  # Start with 2 ambulances
        base_location=pfars_node,
        hospital_node=hospital_node,
        index_to_node=index_to_node,
        node_to_index=node_to_index,
        avg_speed=8.33,  # 30 km/h
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