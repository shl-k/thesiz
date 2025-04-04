"""
Geographical utilities for generating and working with road networks, distance matrices, and travel time matrices.
"""
import os
import sys
import time
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import Tuple, Dict, List, Optional, Any

def get_osm_graph(location: Optional[str] = None, bbox: Optional[Tuple] = None, network_type: str = 'drive'):
    """
    Get a street network graph from OSM using either a location name or bounding box coordinates.

    Parameters:
        location: Name of the place (e.g., "Princeton, NJ")
        bbox: Bounding box as (west, south, east, north) e.g., (-74.696560, 40.297067, -74.596615, 40.389835)
        network_type: Type of street network to retrieve. Default is 'drive'
    
    Returns:
        NetworkX graph with OSM attributes
    """
    if location:
        # Get graph directly from location name, without simplification
        G = ox.graph_from_place(location, network_type=network_type, simplify=False)
    elif bbox:
        # Fetch graph using the bounding box, without simplification
        # bbox should be (west, south, east, north)
        G = ox.graph_from_bbox(bbox, network_type=network_type, simplify=False)
    else:
        raise ValueError("Please provide either a location name or bounding box coordinates.")
    
    # Ensure the graph is undirected for distance matrix calculations
    G_undirected = G.to_undirected()
    
    return G_undirected

def create_distance_matrix(location: Optional[str] = None, bbox: Optional[Tuple] = None, 
                         network_type: str = 'drive', save_file: bool = False,
                         output_dir: str = "data/matrices",
                         min_edge_length: int = 15, simplify: bool = False,
                         graph: Optional[nx.Graph] = None):
    """
    Create a distance matrix from a location, bounding box, or pre-existing graph.
    
    Parameters:
        location: Name of the place (e.g., "Princeton, NJ")
        bbox: Bounding box as (minx, miny, maxx, maxy)
        network_type: Type of street network to retrieve. Default is 'drive'
        save_file: Whether to save the matrix to file
        output_dir: Directory to save matrix (if save_file is True)
        sparsify: Whether to sparsify the graph (default: True)
        min_edge_length: Minimum edge length to keep when sparsifying (meters)
        simplify: Whether to simplify the graph when sparsifying
        graph: Pre-existing NetworkX graph to use instead of fetching from OSM
        
    Returns:
        Tuple of (graph, distance_matrix)
    """
    
    # Get graph
    if graph is not None:
        print("Using provided graph...")
        G_original = graph
    else:
        print("Getting graph from OSM...")
        G_original = get_osm_graph(location=location, bbox=bbox, network_type=network_type)
    
    print(f"Original graph: {len(G_original.nodes)} nodes, {len(G_original.edges)} edges")
    
    # Ensure graph is undirected
    if not isinstance(G, nx.Graph):
        G = G.to_undirected()

    # Compute the distance matrix using the 'length' attribute for edge weights
    print("Computing distance matrix...")
    distance_matrix = nx.floyd_warshall_numpy(G, weight='length')
    
    # Save matrix if requested
    if save_file:
        os.makedirs(output_dir, exist_ok=True)
        distance_matrix_path = os.path.join(output_dir, "distance_matrix.npy")
        np.save(distance_matrix_path, distance_matrix)
    
    print(f"Final matrix shape: {distance_matrix.shape}")
    
    return G, distance_matrix

def calculate_travel_time_matrix(distance_matrix: np.ndarray, avg_speed: float = 8.33) -> np.ndarray:
    """
    Calculate travel time matrix from distance matrix.
    
    Args:
        distance_matrix: Matrix of distances between nodes in meters
        avg_speed: Average speed in m/s (default 8.33 m/s = 30 km/h)
        
    Returns:
        Travel time matrix in seconds
    """
    return distance_matrix / avg_speed  # Convert to seconds

def calculate_travel_time(distance: float, speed: float = 8.33) -> float:
    """
    Calculate travel time for a given distance.
    
    Args:
        distance: Distance in meters
        speed: Speed in m/s (default 8.33 m/s = 30 km/h)
        
    Returns:
        Travel time in seconds
    """
    return distance / speed  # Convert to seconds

#defaulted to using osmnx's inbuilt sparsification function
'''def sparsify_graph(G, min_edge_length=15, simplify=False):
    """
    Sparsify a graph by removing edges that are too short and ensuring connectivity.
    
    Parameters:
        G: NetworkX graph
        min_edge_length: Minimum edge length to keep (meters), default 15m
        simplify: Whether to simplify the graph (remove nodes with degree=2), default False
        
    Returns:
        Sparsified NetworkX graph
    """
    print(f"Original graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Create a copy of the graph to work with
    G_sparse = G.copy()
    
    # Step 1: Remove edges shorter than min_edge_length, but only if removing them
    # doesn't disconnect nodes that would become isolated
    edges_to_remove = []
    for u, v, data in G_sparse.edges(data=True):
        if 'length' in data and data['length'] < min_edge_length:
            # Only remove edge if both nodes will still have other connections
            if G_sparse.degree(u) > 1 and G_sparse.degree(v) > 1:
                edges_to_remove.append((u, v))
    
    G_sparse.remove_edges_from(edges_to_remove)
    print(f"After removing short edges: {len(G_sparse.nodes)} nodes, {len(G_sparse.edges)} edges")
    
    # Step 2: Ensure graph is connected by keeping only the largest component
    if not nx.is_connected(G_sparse):
        print("Graph is not connected, keeping only the largest component...")
        largest_cc = max(nx.connected_components(G_sparse), key=len)
        G_sparse = G_sparse.subgraph(largest_cc).copy()
        print(f"After keeping largest component: {len(G_sparse.nodes)} nodes, {len(G_sparse.edges)} edges")
    
    # Step 3: Simplify the graph if requested (but be more conservative)
    if simplify:
        # Only remove degree-2 nodes that form a straight line
        # (i.e., nodes that don't represent turns or intersections)
        degree_2_nodes = [node for node, degree in dict(G_sparse.degree()).items() if degree == 2]
        nodes_to_remove = []
        
        for node in degree_2_nodes:
            neighbors = list(G_sparse.neighbors(node))
            if len(neighbors) == 2:
                # Get coordinates of the node and its neighbors
                node_coords = (G_sparse.nodes[node]['y'], G_sparse.nodes[node]['x'])
                n1_coords = (G_sparse.nodes[neighbors[0]]['y'], G_sparse.nodes[neighbors[0]]['x'])
                n2_coords = (G_sparse.nodes[neighbors[1]]['y'], G_sparse.nodes[neighbors[1]]['x'])
                
                # Calculate angles to determine if the node is part of a straight line
                # If it is, we can safely remove it
                angle = abs(np.arctan2(n2_coords[0] - node_coords[0], n2_coords[1] - node_coords[1]) -
                          np.arctan2(node_coords[0] - n1_coords[0], node_coords[1] - n1_coords[1]))
                
                # If angle is close to 180 degrees (straight line)
                if abs(angle - np.pi) < 0.1:  # Allow small deviation from straight line
                    edge1_data = G_sparse.get_edge_data(node, neighbors[0])
                    edge2_data = G_sparse.get_edge_data(node, neighbors[1])
                    new_length = edge1_data.get('length', 0) + edge2_data.get('length', 0)
                    G_sparse.add_edge(neighbors[0], neighbors[1], length=new_length)
                    nodes_to_remove.append(node)
        
        G_sparse.remove_nodes_from(nodes_to_remove)
        print(f"After simplification: {len(G_sparse.nodes)} nodes, {len(G_sparse.edges)} edges")
    
    return G_sparse
'''

def lat_lon_to_node(G, lat, lon):
    """
    Convert a lat/long coordinate to the nearest node in the OSMnx graph.
    
    Parameters:
    G: NetworkX graph from OSMnx
    lat: Latitude
    lon: Longitude
    
    Returns:
    node_id: The OSM node ID of the nearest node
    """
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)


def node_to_lat_lon(G, node_id):
    """
    Convert a node ID to lat/long coordinates from the OSMnx graph.
    
    Parameters:
    G: NetworkX graph from OSMnx
    node_id: The ID of the node in the graph
    
    Returns:
    tuple: (lat, lon) coordinates
    """
    # Get the node data
    node_data = G.nodes[node_id]
    
    # OSMnx stores coordinates as (x, y) where x is longitude and y is latitude
    return node_data['y'], node_data['x'] #(lat, lon)

#visualization functions
def plot_demand_heatmap(G, demand_vec, title="Demand Heatmap", filename=None, colorbar_label="Demand"):
    """
    Plot a heatmap of demand on the graph
    
    Parameters:
        G: NetworkX graph
        demand_vec: Vector of demand values for each node
        title: Title for the plot
        filename: If provided, save the plot to this file
        colorbar_label: Label for the colorbar
        
    Returns:
        fig, ax: The figure and axis objects
    """
    # Create a dictionary of node values for coloring
    node_values = {node: demand_vec[i] for i, node in enumerate(G.nodes)}
    
    # Use OSMnx's plot_graph function
    fig, ax = ox.plot_graph(
        G, 
        node_color=[node_values[node] for node in G.nodes],
        node_size=20,
        edge_linewidth=0.5,
        figsize=(12, 10),
        show=False,
        close=False
    )
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(colorbar_label)
    
    plt.title(title)
    
    # Return the figure and axis for display in the notebook
    return fig, ax

def plot_ambulance_locations(G, ambulance_nodes, title="Ambulance Locations", node_mapping=None):
    """
    Plot ambulance locations on the graph with the same styling as the heatmap plots
    
    Parameters:
        G: NetworkX graph
        ambulance_nodes: List of node IDs or indices where ambulances are located
        title: Title for the plot
        node_mapping: Optional mapping from indices to node IDs (if ambulance_nodes contains indices)
        
    Returns:
        fig, ax: The figure and axis objects
    """
    # Ensure the graph has a CRS attribute
    if 'crs' not in G.graph:
        G.graph['crs'] = 'EPSG:4326'  # Standard WGS84 coordinate system
    
    # Convert indices to node IDs if node_mapping is provided
    if node_mapping is not None:
        ambulance_node_ids = [node_mapping[idx] for idx in ambulance_nodes]
    else:
        ambulance_node_ids = ambulance_nodes
    
    # Print ambulance locations for reference
    print(f"\nAmbulance Locations ({len(ambulance_node_ids)} total):")
    for i, node in enumerate(ambulance_node_ids):
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        print(f"Ambulance {i+1}: Node {node} at coordinates ({x:.6f}, {y:.6f})")
    
    # Create node colors and sizes - using the same color scheme as heatmap
    node_colors = []
    node_sizes = []
    
    for node in G.nodes:
        if node in ambulance_node_ids:
            # Use a bright red color that stands out against the plasma colormap
            node_colors.append('#FF4136')  # Red for ambulance locations
            node_sizes.append(100)         # Larger size for ambulance locations
        else:
            # Use a light gray that matches the heatmap style
            node_colors.append('#CCCCCC')  # Light gray for regular nodes
            node_sizes.append(20)          # Same size as in heatmap plots
    
    # Use OSMnx's plot_graph function with the same styling as heatmap plots
    fig, ax = ox.plot_graph(
        G, 
        node_color=node_colors,
        node_size=node_sizes,
        node_alpha=0.9,
        edge_linewidth=0.5,
        figsize=(12, 10),
        show=False,
        close=False
    )
    
    # Add a title with the same styling as heatmap plots
    plt.title(title)
    
    # Add a legend to identify ambulance locations
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4136', 
               markersize=10, label='Ambulance Location'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC', 
               markersize=6, label='Network Node')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add small labels next to ambulances to distinguish them
    for i, node in enumerate(ambulance_node_ids):
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        ax.annotate(f"{i+1}", 
                   xy=(x, y), 
                   xytext=(5, 5),
                   textcoords="offset points",
                   fontsize=9,
                   fontweight='bold',
                   color='black',
                   bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", alpha=0.8))
    
    return fig, ax

def visualize_synthetic_calls(G, synthetic_calls_file='data/processed/synthetic_calls.csv'):
    """
    Visualize synthetic calls on the graph using a heatmap.
    
    Parameters:
        G: NetworkX graph
        synthetic_calls_file: Path to the synthetic calls CSV file
        
    Returns:
        fig, ax: The figure and axis objects
    """
    # Read synthetic calls
    calls_df = pd.read_csv(synthetic_calls_file)
    
    # Create demand vector (count of calls per node)
    # Initialize with zeros for each node in the graph
    demand_vec = {node: 0 for node in G.nodes()}
    
    # Count calls at each node using OSM node IDs
    for node_id in calls_df['origin_node']:
        if node_id in demand_vec:
            demand_vec[node_id] += 1
    
    # Convert to list maintaining order of G.nodes()
    demand_list = [demand_vec[node] for node in G.nodes()]
    
    # Plot the heatmap
    fig, ax = plot_demand_heatmap(
        G, 
        np.array(demand_list),
        title="Synthetic Emergency Calls Heatmap",
        filename="princeton_synthetic_calls.png",
        colorbar_label="Number of Calls"
    )
    
    print(f"Total calls: {len(calls_df)}")
    print(f"Unique nodes with calls: {len([n for n, d in demand_vec.items() if d > 0])}")
    
    return fig, ax

# Aliases for backward compatibility
osm_graph = get_osm_graph
osm_distance = create_distance_matrix
osmnx_to_graph = get_osm_graph
osmnx_to_distance_matrix = create_distance_matrix 