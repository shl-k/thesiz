import os
import sys
import numpy as np
import networkx as nx
from train_rl import train_rl
from src.utils.geo_utils import osm_graph, create_distance_matrix
from src.data.princeton_data_prep import sparsify_graph, load_princeton_statistics

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main():
    # Get Princeton graph and data
    print('Getting Princeton graph...')
    G_pton = osm_graph(location='Princeton, NJ', network_type='drive')
    
    # Sparsify the graph
    print('Sparsifying graph...')
    G_pton = sparsify_graph(G_pton, min_edge_length=30, simplify=True)
    
    # Get travel time matrix
    print('Calculating travel time matrix...')
    G_pton, travel_time_matrix, node_to_index, index_to_node = create_distance_matrix(graph=G_pton, save_file=True)
    
    # Load Princeton statistics for base locations etc.
    stats = load_princeton_statistics()
    
    # Set up data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Run training with small number of episodes
    print("Starting small training test...")
    agent, env, stats = train_rl(
        graph=G_pton,
        travel_time_matrix=travel_time_matrix,
        base_locations=stats['base_locations'],
        hospital_node=stats['hospital_node'],
        num_episodes=10,
        algorithm='q-learning',
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        coverage_model='dsm',  # Use DSM coverage model
        verbose=True
    )
    
    print("\nTraining completed successfully!")
    print("\nFinal Statistics:")
    print(f"Mean response time: {stats['mean_response_times'][-1]:.2f} minutes")
    print(f"Calls served: {stats['calls_served'][-1]}")
    print(f"Calls dropped: {stats['calls_dropped'][-1]}")

if __name__ == "__main__":
    main() 