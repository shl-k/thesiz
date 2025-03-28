"""
Test script for OSM functions
"""

import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.geo_utils import osm_distance
from src.utils.geo_utils import osm_graph

def test_osm_functions():
    """Test the OSM functions with a small location"""
    
    print("=" * 50)
    print("TESTING OSM FUNCTIONS")
    print("=" * 50)
    
    # Test location
    location = "Princeton University, NJ"
    print(f"\nTesting with location: {location}")
    
    # 1. Test osm_graph
    print("\n" + "=" * 50)
    print("1. Testing osm_graph function...")
    print("=" * 50)
    
    try:
        G = osm_graph(location=location, network_type="drive")
        
        print(f"Graph successfully created!")
        print(f"Number of nodes: {len(G.nodes)}")
        print(f"Number of edges: {len(G.edges)}")
        
        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 8))
        ox.plot_graph(G, ax=ax, node_size=30, edge_linewidth=0.5)
        plt.title(f'Street Network for {location}')
        plt.savefig('osm_graph_test.png')
        print(f"Graph visualization saved to 'osm_graph_test.png'")
        
    except Exception as e:
        print(f"Error testing osm_graph: {str(e)}")
    
    # 2. Test osm_distance
    print("\n" + "=" * 50)
    print("2. Testing osm_distance function...")
    print("=" * 50)
    
    try:
        D = osm_distance(location=location, network_type="drive")
        
        print(f"Distance matrix successfully created!")
        print(f"Matrix shape: {D.shape}")
        
        # Basic checks on the distance matrix
        print(f"Min distance: {D.min()}")
        print(f"Max distance: {D.max()}")
        print(f"Mean distance: {D.mean()}")
        
        # Check that diagonal is zero
        diag_zero = np.allclose(np.diag(D), 0)
        print(f"Diagonal is zero: {diag_zero}")
        
    except Exception as e:
        print(f"Error testing osm_distance: {str(e)}")
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    test_osm_functions() 