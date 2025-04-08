"""
Utility functions for clustering nodes based on coordinates and demand data.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import pandas as pd
import json
import os

def load_node_mappings(
    node_to_lat_lon_file: str = 'data/matrices/node_to_lat_lon.json',
    lat_lon_to_node_file: str = 'data/matrices/lat_lon_to_node.json'
) -> Tuple[Dict[int, Tuple[float, float]], Dict[Tuple[float, float], int]]:
    """
    Load node coordinate mappings from files.
    
    Args:
        node_to_lat_lon_file: Path to JSON file mapping node IDs to lat/lon coordinates
        lat_lon_to_node_file: Path to JSON file mapping lat/lon coordinates to node IDs
        
    Returns:
        Tuple containing:
        - Dictionary mapping node IDs to (lat, lon) coordinates
        - Dictionary mapping (lat, lon) coordinates to node IDs
    """
    # Load node to lat/lon mapping
    with open(node_to_lat_lon_file, 'r') as f:
        raw_node_to_lat_lon = json.load(f)
        # Convert string keys to integers and lists to tuples
        node_to_lat_lon = {int(k): tuple(v) for k, v in raw_node_to_lat_lon.items()}
    
    # Load lat/lon to node mapping
    with open(lat_lon_to_node_file, 'r') as f:
        raw_lat_lon_to_node = json.load(f)
        # Convert string keys to tuples
        lat_lon_to_node = {tuple(map(float, k.strip('()').split(','))): int(v) 
                          for k, v in raw_lat_lon_to_node.items()}
    
    return node_to_lat_lon, lat_lon_to_node

def cluster_nodes(
    node_to_lat_lon: Dict[int, Tuple[float, float]],
    call_data: pd.DataFrame,
    n_clusters: int = 10,
    random_state: int = 42
) -> Tuple[Dict[int, int], List[int]]:
    """
    Cluster nodes based on their coordinates and demand data from medical trips.
    
    Args:
        node_to_lat_lon: Dictionary mapping node IDs to (lat, lon) coordinates
        call_data: DataFrame with call data (not used anymore, kept for compatibility)
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Dictionary mapping node IDs to cluster IDs
        - List of cluster centers (node IDs)
    """
    # Extract coordinates and create feature matrix
    coords = np.array([node_to_lat_lon[node_id] for node_id in node_to_lat_lon.keys()])
    node_ids = list(node_to_lat_lon.keys())
    
    # Load medical trips data
    medical_trips = pd.read_csv('data/raw/medical_trips.csv')
    
    # Calculate demand weights for each node based on medical trips
    demand_weights = np.zeros(len(node_ids))
    for i, node_id in enumerate(node_ids):
        node_lat, node_lon = node_to_lat_lon[node_id]
        # Count trips where origin is closest to this node
        distances = np.sqrt(
            (medical_trips['origin_lat'] - node_lat)**2 + 
            (medical_trips['origin_lon'] - node_lon)**2
        )
        # Count trips where this node is the closest
        demand_weights[i] = np.sum(distances == distances.min())
    
    # Normalize demand weights
    demand_weights = demand_weights / demand_weights.sum()
    
    # Perform weighted K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(coords, sample_weight=demand_weights)
    
    # Create mapping from node IDs to cluster IDs
    node_to_cluster = {node_id: label for node_id, label in zip(node_ids, cluster_labels)}
    
    # Find cluster centers (nodes closest to cluster centroids)
    cluster_centers = []
    for i in range(n_clusters):
        # Get nodes in this cluster
        cluster_nodes = [node_id for node_id, label in node_to_cluster.items() if label == i]
        if not cluster_nodes:
            continue
            
        # Get coordinates of cluster nodes
        cluster_coords = np.array([node_to_lat_lon[node_id] for node_id in cluster_nodes])
        
        # Get cluster centroid
        centroid = kmeans.cluster_centers_[i]
        
        # Find node closest to centroid
        distances = np.linalg.norm(cluster_coords - centroid, axis=1)
        closest_node = cluster_nodes[np.argmin(distances)]
        cluster_centers.append(closest_node)
    
    return node_to_cluster, cluster_centers

def get_cluster_nodes(
    node_to_cluster: Dict[int, int],
    cluster_id: int
) -> List[int]:
    """
    Get all nodes in a specific cluster.
    
    Args:
        node_to_cluster: Dictionary mapping node IDs to cluster IDs
        cluster_id: ID of the cluster to get nodes for
        
    Returns:
        List of node IDs in the specified cluster
    """
    return [node_id for node_id, c_id in node_to_cluster.items() if c_id == cluster_id]

def get_best_node_in_cluster(
    cluster_id: int,
    node_to_cluster: Dict[int, int],
    node_to_lat_lon: Dict[int, Tuple[float, float]],
    active_calls: Dict,
    path_cache: Dict,
    current_time: float
) -> int:
    """
    Find the best node within a cluster based on average response time to active calls.
    
    Args:
        cluster_id: ID of the cluster to find best node in
        node_to_cluster: Dictionary mapping node IDs to cluster IDs
        node_to_lat_lon: Dictionary mapping node IDs to (lat, lon) coordinates
        active_calls: Dictionary of active emergency calls
        path_cache: Path cache for travel times
        current_time: Current simulation time
        
    Returns:
        Node ID of the best node in the cluster
    """
    # Get all nodes in the cluster
    cluster_nodes = get_cluster_nodes(node_to_cluster, cluster_id)
    if not cluster_nodes:
        return None
        
    # If no active calls, return the node closest to cluster center
    if not active_calls:
        cluster_coords = np.array([node_to_lat_lon[node_id] for node_id in cluster_nodes])
        centroid = np.mean(cluster_coords, axis=0)
        distances = np.linalg.norm(cluster_coords - centroid, axis=1)  # Compute distance along each row
        return cluster_nodes[np.argmin(distances)]
    
    # Find node with lowest average response time to active calls
    best_avg_time = float('inf')
    best_node = None
    
    for node_id in cluster_nodes:
        total_time = 0
        for call in active_calls.values():
            call_node = call["origin_node"]
            if call_node in path_cache[node_id]:
                total_time += path_cache[node_id][call_node]['travel_time']
        avg_time = total_time / len(active_calls) if active_calls else 0
        
        if avg_time < best_avg_time:
            best_avg_time = avg_time
            best_node = node_id
    
    return best_node 