"""
Script to visualize synthetic emergency calls on the Princeton road network.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import osmnx as ox

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.geo_utils import (
    get_osm_graph,
    sparsify_graph,
    visualize_synthetic_calls,
    lat_lon_to_node,
    create_distance_matrix
)
from src.data.princeton_data_prep import generate_demand_with_temporal_pattern

# Define Princeton bounding box
# Format: (west, south, east, north)
PRINCETON_BBOX = (-74.696560, 40.297067, -74.596615, 40.389835)

def visualize_historical_trips(G, medical_trips_file='data/raw/medical_trips.csv'):
    """
    Visualize historical medical trips data on the graph.
    """
    # Load historical data
    medical_trips = pd.read_csv(medical_trips_file)
    
    # Count trips at each location
    location_counts = medical_trips.groupby(['origin_lat', 'origin_lon']).size().reset_index(name='count')
    
    # Create demand vector for nodes
    demand = np.zeros(len(G))
    
    # Map lat/lon to node IDs and count trips
    for _, row in location_counts.iterrows():
        lat, lon = row['origin_lat'], row['origin_lon']
        count = row['count']
        
        # Find nearest node
        min_dist = float('inf')
        nearest_node = None
        
        for node, data in G.nodes(data=True):
            dist = ox.distance.great_circle(lat1=lat, lon1=lon, lat2=data['y'], lon2=data['x'])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        if nearest_node is not None:
            # Convert node ID to index
            node_idx = list(G.nodes()).index(nearest_node)
            demand[node_idx] = count
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the graph
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    
    # Plot only nodes with non-zero demand
    nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_size=20,
                                 node_color=demand,
                                 cmap='plasma',
                                 alpha=0.7,
                                 nodelist=[node for i, node in enumerate(G.nodes()) if demand[i] > 0])
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Number of Historical Trips')
    
    # Add title
    plt.title('Historical Medical Trip Origins in Princeton', fontsize=16)
    
    # Save the plot
    plt.savefig('princeton_historical_trips.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Historical trips visualization saved to princeton_historical_trips.png")
    print(f"Total unique locations: {len(location_counts)}")
    print(f"Total trips: {len(medical_trips)}")
    print("\nTop 10 most common locations:")
    print(location_counts.nlargest(10, 'count'))

def main():
    # Get Princeton graph
    print("Getting Princeton graph...")
    G_pton = get_osm_graph(bbox=PRINCETON_BBOX, network_type='drive')
    
    # Sparsify graph with very conservative parameters
    print("Sparsifying graph...")
    G_pton_sparse = sparsify_graph(G_pton, min_edge_length=10, simplify=False)  # Reduced min_edge_length and disabled simplification
    
    # Generate new synthetic calls using sparsified graph
    print("Generating new synthetic calls...")
    generate_demand_with_temporal_pattern(G_pton_sparse, num_days=7)
    
    # Visualize historical trips
    print("\nVisualizing historical trips...")
    visualize_historical_trips(G_pton_sparse)
    
    # Visualize synthetic calls
    print("\nVisualizing synthetic calls...")
    fig, ax = visualize_synthetic_calls(G_pton_sparse)
    
    # Save the plot
    plt.savefig('princeton_synthetic_calls.png')
    plt.close()
    
    print('\nVisualization complete! Check princeton_synthetic_calls.png')

if __name__ == "__main__":
    main() 