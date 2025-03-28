"""
Test script for geo_utils.py functions
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.geo_utils import (
    get_osm_graph,
    create_distance_matrix,
    sparsify_graph,
    lat_lon_to_node,
    node_to_lat_lon,
    plot_demand_heatmap
)

def test_geo_utils():
    """Test the main functions in geo_utils.py"""
    
    print("=" * 50)
    print("TESTING GEO_UTILS FUNCTIONS")
    print("=" * 50)
    
    # Test bounding box coordinates
    bbox = (-74.696560, 40.297067, -74.596615, 40.389835)
    
    # 1. Test get_osm_graph with bounding box
    print("\n1. Testing get_osm_graph with bounding box...")
    G = get_osm_graph(bbox=bbox)
    print(f"Successfully created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # 2. Test create_distance_matrix
    print("\n2. Testing create_distance_matrix...")
    try:
        G, D = create_distance_matrix(bbox=bbox, network_type='drive')
        print(f"Successfully created distance matrix with shape {D.shape}")
        print(f"Min distance: {D.min():.2f}m")
        print(f"Max distance: {D.max():.2f}m")
        print(f"Mean distance: {D.mean():.2f}m")
        
    except Exception as e:
        print(f"Error testing create_distance_matrix: {str(e)}")
    
    # 3. Test sparsify_graph
    print("\n3. Testing graph simplification and sparsification...")
    try:
        # Get a fresh copy of the original graph
        G_original = get_osm_graph(bbox=bbox)
        print(f"Original graph: {len(G_original.nodes)} nodes, {len(G_original.edges)} edges")
        
        # Convert to directed graph before simplification
        G_directed = G_original.to_directed()
        G_simple = ox.simplify_graph(G_directed)
        print(f"Simplified graph: {len(G_simple.nodes)} nodes, {len(G_simple.edges)} edges")
        print(f"Reduction in nodes: {(1 - len(G_simple.nodes) / len(G_original.nodes)) * 100:.2f}%")
        print(f"Reduction in edges: {(1 - len(G_simple.edges) / len(G_original.edges)) * 100:.2f}%")
        
        # Get sparsified graph
        G_sparse = sparsify_graph(G_original, min_edge_length=15, simplify=False)
        print(f"Sparsified graph: {len(G_sparse.nodes)} nodes, {len(G_sparse.edges)} edges")
        print(f"Reduction in nodes: {(1 - len(G_sparse.nodes) / len(G_original.nodes)) * 100:.2f}%")
        print(f"Reduction in edges: {(1 - len(G_sparse.edges) / len(G_original.edges)) * 100:.2f}%")
        
        # Create a figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        
        # Get the bounds from the original graph
        bounds = ox.utils_geo.bbox_from_point((bbox[1], bbox[0]), dist=1000)
        
        # Plot original graph
        ox.plot_graph(G_original, ax=ax1, node_size=0, edge_linewidth=0.5, 
                     show=False, close=False, bbox=bounds)
        ax1.set_title(f'Original Graph\n{len(G_original.nodes)} nodes, {len(G_original.edges)} edges')
        
        # Plot simplified graph
        ox.plot_graph(G_simple, ax=ax2, node_size=0, edge_linewidth=0.5,
                     show=False, close=False, bbox=bounds)
        ax2.set_title(f'Simplified Graph\n{len(G_simple.nodes)} nodes, {len(G_simple.edges)} edges')
        
        # Plot sparsified graph
        ox.plot_graph(G_sparse, ax=ax3, node_size=0, edge_linewidth=0.5,
                     show=False, close=False, bbox=bounds)
        ax3.set_title(f'Sparsified Graph\n{len(G_sparse.nodes)} nodes, {len(G_sparse.edges)} edges')
        
        # Save the side-by-side comparison
        plt.tight_layout()
        plt.savefig('test_princeton_graph_comparison.png')
        plt.close()
        print("Graph comparison visualization saved to 'test_princeton_graph_comparison.png'")
        
    except Exception as e:
        print(f"Error testing graph simplification: {str(e)}")
    
    # 4. Test lat_lon_to_node and node_to_lat_lon
    print("\n4. Testing coordinate conversion functions...")
    try:
        # Test with PFARS HQ coordinates
        pfars_lat, pfars_lon = 40.361395, -74.664879
        node_id = lat_lon_to_node(G, pfars_lat, pfars_lon)
        print(f"PFARS HQ coordinates ({pfars_lat}, {pfars_lon}) mapped to node {node_id}")
        
        # Convert back to coordinates
        lat, lon = node_to_lat_lon(G, node_id)
        print(f"Node {node_id} mapped back to coordinates ({lat}, {lon})")
        print(f"Coordinate conversion error: {abs(lat - pfars_lat):.6f}, {abs(lon - pfars_lon):.6f}")
        
    except Exception as e:
        print(f"Error testing coordinate conversion: {str(e)}")
    
    # 5. Test plot_demand_heatmap
    print("\n5. Testing plot_demand_heatmap...")
    try:
        # Create random demand vector
        demand_vec = np.random.rand(len(G.nodes))
        fig, ax = plot_demand_heatmap(
            G,
            demand_vec,
            title="Test Demand Heatmap",
            filename="test_demand_heatmap.png",
            colorbar_label="Random Demand"
        )
        plt.close()
        print("Successfully created and saved demand heatmap to 'test_demand_heatmap.png'")
        
    except Exception as e:
        print(f"Error testing plot_demand_heatmap: {str(e)}")

if __name__ == "__main__":
    test_geo_utils() 