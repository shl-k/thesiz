"""
Princeton Data Preparation
- Uses pre-filtered medical trips data
- Generates realistic demand patterns based on Princeton's geography and temporal patterns
"""
import numpy as np
import osmnx as ox
import networkx as nx
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys
from pathlib import Path
import pickle
import json

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.utils.geo_utils import lat_lon_to_node
import pandas as pd
from scipy.stats import poisson
from sklearn.neighbors import KernelDensity

# Constants file path
PRINCETON_TRIPS_FILE = 'data/raw/Princeton_trip_data.csv'
MEDICAL_TRIPS_FILE = 'data/raw/medical_trips.csv'
SYNTHETIC_CALLS_FILE = 'data/processed/synthetic_calls.csv'
GRAPH_FILE = 'data/processed/princeton_graph.gpickle'
NUMDAYS = 2

# Define Princeton bounding box
# Format: (west, south, east, north)
PRINCETON_BBOX = (-74.696560, 40.297067, -74.596615, 40.389835)

def generate_graph(bbox=PRINCETON_BBOX, network_type='drive'):
    """
    Generate a graph of Princeton using OSMnx.
    
    Parameters:
        bbox: Bounding box of Princeton
    """
    G = ox.graph_from_bbox(bbox, network_type=network_type, simplify=True)
    print(f"After graph_from_bbox: type(G) = {type(G)}")
    
    G = ox.project_graph(G) 
    print(f"After project_graph: type(G) = {type(G)}")
    
    G  = ox.simplification.consolidate_intersections(
        G ,
        tolerance=15,
        rebuild_graph=True,
        reconnect_edges=True
    )
    print(f"After consolidate_intersections: type(G) = {type(G)}")
    
    # Find strongly connected components and keep only the largest one
    strongly_connected_components = list(nx.strongly_connected_components(G))
    print(f"Type of strongly_connected_components: {type(strongly_connected_components)}")
    print(f"Type of first component: {type(strongly_connected_components[0])}")
    
    largest_component = max(strongly_connected_components, key=len)
    print(f"Type of largest_component: {type(largest_component)}")
    
    # Create a subgraph with only the largest component
    G = G.subgraph(largest_component).copy()
    print(f"After subgraph: type(G) = {type(G)}")
    
    G = ox.routing.add_edge_speeds(G)
    print(f"After add_edge_speeds: type(G) = {type(G)}")
    
    G = ox.routing.add_edge_travel_times(G)
    print(f"After add_edge_travel_times: type(G) = {type(G)}")
    
    G = ox.project_graph(G, to_crs='epsg:4326')  # convert back to lat/lon
    print(f"After project_graph to lat/lon: type(G) = {type(G)}")
    
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/princeton_graph.gpickle', 'wb') as f:
        pickle.dump(G, f)
    return G
    


def generate_demand_with_temporal_pattern(G, medical_trips_file=MEDICAL_TRIPS_FILE, num_days=7):
    """
    Generate synthetic demand based on historical spatial and temporal distributions
    from Princeton's medical trip data.
    
    Parameters:
    G: NetworkX graph of Princeton (should be sparsified)
    medical_trips_file: CSV file containing historical medical trips
    num_days: Number of days to generate data for
    
    Returns:
    synthetic_calls: DataFrame with day, timestamp, origin node, destination (hospital) node, and intensity
    """
    
    medical_trips = pd.read_csv(MEDICAL_TRIPS_FILE)
    
    # Get PFARS HQ node and hospital node
    pfars_node = lat_lon_to_node(G, 40.361395, -74.664879)
    hospital_node = lat_lon_to_node(G, 40.340339, -74.623913)
    print(f"Using PFARS HQ node: {pfars_node}")
    print(f"Using hospital node: {hospital_node}")
    
    # Verify critical nodes are in the graph
    if pfars_node not in G.nodes:
        raise ValueError(f"PFARS HQ node {pfars_node} not found in sparsified graph")
    if hospital_node not in G.nodes:
        raise ValueError(f"Hospital node {hospital_node} not found in sparsified graph")
    
    # Get all valid nodes in the sparsified graph
    valid_nodes = list(G.nodes(data=True))

    print(f"Generating calls using {len(valid_nodes)} valid nodes in sparsified graph")
    
    # Create spatial distribution from historical data
    # Extract origin coordinates from medical trips
    historical_lats = medical_trips['origin_lat'].values
    historical_lons = medical_trips['origin_lon'].values
    
    # Fit a Kernel Density Estimator to historical locations
    # Using a much larger bandwidth to spread out the demand more
    kde = KernelDensity(bandwidth=0.005, kernel='gaussian')  # 5x larger bandwidth
    historical_coords = np.column_stack([historical_lats, historical_lons])
    kde.fit(historical_coords)
    
    # Calculate probability of each valid node based on KDE
    valid_coords = np.array([[node_data['y'], node_data['x']] for _, node_data in valid_nodes])
    # Get log probabilities and convert to actual probabilities
    log_probs = kde.score_samples(valid_coords)
    node_probs = np.exp(log_probs)
    # Add a larger constant to ensure more uniform base probability
    node_probs = node_probs + 0.1  # Much larger base probability
    node_probs = node_probs / node_probs.sum()  # Normalize
    
    # Group by seconds and calculate frequency with smoothing
    seconds_in_day = 24 * 60 * 60  # 86400 seconds
    second_counts = medical_trips.groupby('ODepartureTime').size()
    
    # Define multiple peak times (in seconds)
    peak_times = [
        25200,  # 7 AM
        43200,  # 12 PM
        64800,  # 6 PM
        79200   # 10 PM
    ]
    peak_weights = [0.1, 0.1, 0.4, 0.4]  # Weights for each peak
    std_dev = 2 * 60 * 60  # 2 hours in seconds
    
    # Create smoothing weights for each second using multiple peaks
    smoothing_weights = np.zeros(seconds_in_day)
    for peak_time, weight in zip(peak_times, peak_weights):
        smoothing_weights += weight * np.exp(-(np.arange(seconds_in_day) - peak_time)**2 / (2 * std_dev**2))
    
    # Normalize weights
    smoothing_weights = smoothing_weights / smoothing_weights.sum()
    
    # Apply smoothing to all seconds, ensuring each gets a baseline probability
    second_probs = np.array([second_counts.get(s, 0) + smoothing_weights[s] * 100 for s in range(seconds_in_day)])
    
    # Normalize final probabilities
    second_probs = second_probs / second_probs.sum()
    
    # Generate synthetic calls
    synthetic_calls = []
    
    # Generate calls for each day using truncated normal distribution
    for day in range(1, num_days + 1):
        # Number of calls for this day follows truncated normal(8, 1.5)
        while True:
            num_calls_today = int(np.random.normal(8, 1.5))
            if 5 <= num_calls_today <= 11:  # Truncate to reasonable range
                break
        
        for _ in range(num_calls_today):
            # Sample second of day from temporal distribution
            second_of_day = np.random.choice(seconds_in_day, p=second_probs)
            
            # Sample a node based on spatial distribution from historical data
            node_idx = np.random.choice(len(valid_nodes), p=node_probs)
            node_id, node_data = valid_nodes[node_idx]
            
            # Generate intensity (1-100)
            # Using a right-skewed distribution to favor lower intensities
            intensity = int(np.random.beta(2, 5) * 100) + 1
            
            # Calculate time components
            hour = second_of_day // 3600
            minute = (second_of_day % 3600) // 60
            second = second_of_day % 60
            
            # Create timestamp string (Day: HH:MM:SS format)
            timestamp = f"Day {day}: {hour:02d}:{minute:02d}:{second:02d}"
            
            # Add to synthetic calls
            synthetic_calls.append({
                'day': day,
                'second_of_day': second_of_day,
                'timestamp': timestamp,
                'origin_lat': node_data['y'],
                'origin_lon': node_data['x'],
                'origin_node': node_id,
                'destination_node': hospital_node,
                'intensity': intensity
            })

    # Convert to DataFrame
    calls_df = pd.DataFrame(synthetic_calls)
    
    # Save the synthetic calls to a CSV file
    output_file = 'data/processed/synthetic_calls.csv'
    calls_df.to_csv(output_file, index=False)
    print(f"Synthetic calls saved to: {output_file}")
    print(f"Total calls generated: {len(calls_df)}")
    print("\nCalls per day:")
    print(calls_df.groupby('day').size())
    
    # Print unique locations stats
    print("\nUnique locations generated:")
    print(f"Unique lat/lon pairs: {len(calls_df.groupby(['origin_lat', 'origin_lon']))}")
    print(f"Unique nodes: {len(calls_df['origin_node'].unique())}")
    print(f"PFARS HQ node: {pfars_node}")
    print(f"Destination node (hospital): {hospital_node}")
    
    return calls_df

# old demand generation function, prior to using Princeton's Korny data
''' def generate_realistic_demand(G, num_clusters=3, cluster_std=0.01, base_demand=1):
    """
    Generate realistic demand patterns based on Princeton's actual EMS data.
    
    Parameters:
        G: NetworkX graph
        num_clusters: Number of demand clusters
        cluster_std: Standard deviation of clusters
        base_demand: Base demand level for all nodes
        
    Returns:
        demand_vec: Demand vector for each node
        intensity_vec: Intensity vector for each node (for objective weighting)
        gamma_vec: Priority vector for each node (for unmet demand penalties)
    """
    # Extract node coordinates
    node_coords = np.array([[G.nodes[node]['y'], G.nodes[node]['x']] for node in G.nodes])
    
    # Define key locations based on actual Princeton data
    princeton_university = [40.3431, -74.6551]  # Princeton University (high call volume)
    downtown_princeton = [40.3500, -74.6590]    # Nassau Street area
    princeton_shopping = [40.3583, -74.6429]    # Princeton Shopping Center
    construction_areas = [40.3470, -74.6610]    # Growing construction projects
    
    # Define hospital locations (for potential future use)
    princeton_medical_center = [40.3586, -74.6810]  # Princeton Medical Center (88.1% of transports)
    
    # Define cluster centers with weights based on actual call distribution
    cluster_centers = np.array([
        princeton_university,   # Higher weight for university area
        downtown_princeton,
        princeton_shopping,
        construction_areas      # Growing call volume in construction areas
    ])
    
    # Weights for each cluster based on reported call distribution
    # These weights determine how much each area contributes to the overall demand
    cluster_weights = np.array([0.4, 0.25, 0.2, 0.15])  # University gets highest weight (40%)
    
    # Calculate distance from each node to each cluster center
    distances = distance.cdist(node_coords, cluster_centers)
    
    # Convert distances to weights using a Gaussian function
    # Nodes closer to cluster centers get higher weights
    # The cluster_std parameter controls how quickly weights decrease with distance
    weights = np.exp(-distances**2 / (2 * cluster_std**2))
    
    # Apply cluster weights to reflect the different importance of each area
    # University (40%), Downtown (25%), Shopping (20%), Construction (15%)
    for i in range(len(cluster_weights)):
        weights[:, i] *= cluster_weights[i]
    
    # Combine weights from all clusters to get a single weight per node
    combined_weights = weights.sum(axis=1)
    
    # Normalize weights to create intensity vector (for objective function)
    # This makes the values sum to the number of nodes and average to 1.0
    intensity_vec = combined_weights / combined_weights.sum()
    intensity_vec = intensity_vec * len(intensity_vec)
    
    # Create demand vector (actual number of calls)
    # Using Princeton's actual annual call volume of approximately 1700 calls
    annual_calls = 1700
    
    # IMPROVED APPROACH: Only generate calls for "active" nodes
    # Define a threshold below which nodes are considered unlikely to generate calls
    # This creates a more realistic pattern where remote areas have zero demand
    active_threshold = 0.5  # Nodes with intensity < 0.5 are unlikely to generate calls
    active_nodes = intensity_vec >= active_threshold
    
    # Calculate the sum of intensities for active nodes only
    active_intensity = intensity_vec[active_nodes]
    
    # Scale based only on active nodes to distribute the annual calls more realistically
    # This ensures calls are concentrated in areas where they're likely to occur
    call_scaling = annual_calls / active_intensity.sum()
    
    # Initialize demand vector with zeros (no calls by default)
    demand_vec = np.zeros_like(intensity_vec)
    
    # Generate calls only for active nodes using Poisson distribution
    # Poisson is ideal for modeling count data like emergency calls
    # It adds realistic randomness while maintaining the expected total
    np.random.seed(42)  # For reproducibility
    demand_vec[active_nodes] = np.random.poisson(active_intensity * call_scaling)
    
    # Create gamma vector (priority/penalty for unmet demand)
    # Higher values for university area based on actual data
    gamma_vec = 100 * intensity_vec  # Base penalty proportional to intensity
    
    # Add extra priority to university area (critical to serve)
    # This reflects the importance of responding to university emergencies
    university_distances = distances[:, 0]  # University is first cluster
    university_priority = np.exp(-university_distances**2 / (2 * (cluster_std/2)**2))
    gamma_vec += 50 * university_priority
    
    # Print some statistics about the generated demand
    active_node_count = np.sum(active_nodes)
    total_demand = np.sum(demand_vec)
    print(f"Generated demand on {active_node_count} active nodes out of {len(G.nodes)} total nodes")
    print(f"Total annual calls: {total_demand} (target: {annual_calls})")
    print(f"Average calls per active node: {total_demand/active_node_count:.2f}")
    print(f"Maximum calls at a single node: {np.max(demand_vec)}")
    
    return demand_vec, intensity_vec, gamma_vec
'''

# used once to extract medical trips from the original data
def extract_medical_trips(input_file=PRINCETON_TRIPS_FILE, output_file='data/raw/medical_trips.csv'):
    # Read the original data
    print("Reading data file...")
    df = pd.read_csv(input_file, header=0, names=[
        'Person ID', 'OType', 'OName', 'OLon', 'OLat', 'OXCoord', 'OYCoord', 
        'ODepartureTime', 'DType', 'DName', 'DLon', 'DLat', 'DXCoord', 'DYCoord', 
        'GCDistance', 'Prob. of Driving Themselves', 'Prob. of Getting a Ride', 
        'Prob. of Using Uber/MT', 'Prob. of Taking AV System', 'Trip_Type'
    ])
    
    # Define medical keywords more precisely
    medical_keywords = [
        'MEDICAL', 'HOSPITAL', 'CLINIC', 'HEALTHCARE', 'HEALTH',
        'THERAPY', 'DOCTOR', 'PHYSICIAN', 'DENTAL', 'REHAB',
        'EMERGENCY', 'URGENT CARE', 'PHARMACY', 'LABORATORY',
        'DIAGNOSTIC', 'TREATMENT', 'CARE CENTER'
    ]
    
    # Filter for medical trips
    medical_trips = df[
        # Include if destination contains medical keywords
        (df['DName'].str.contains('|'.join(medical_keywords), case=False, na=False))
    ]
    
    # Convert timestamp and create formatted version
    medical_trips['datetime'] = pd.to_datetime(medical_trips['ODepartureTime'], unit='s')
    medical_trips['formatted_time'] = medical_trips['datetime'].dt.strftime('%H:%M:%S')
    
    # Keep only essential columns in specified order
    columns_to_keep = [
        'ODepartureTime', 'formatted_time', 'OName', 'OLat', 'OLon', 
        'DName', 'DLat', 'DLon'
    ]
    
    medical_trips = medical_trips[columns_to_keep]
    
    # Rename columns for clarity
    medical_trips = medical_trips.rename(columns={
        'OLat': 'origin_lat',
        'OLon': 'origin_lon',
        'OName': 'origin_name',
        'DLat': 'destination_lat',
        'DLon': 'destination_lon',
        'DName': 'destination_name'
    })
    
    # Sort by departure time
    medical_trips = medical_trips.sort_values('ODepartureTime')
    
    # Save to new CSV file
    medical_trips.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total trips in original data: {len(df)}")
    print(f"Total medical trips found: {len(medical_trips)}")
    print(f"\nMedical trips saved to: {output_file}")
    
    return medical_trips

def visualize_graph(G, pfars_node, hospital_node, title="Princeton Road Network"):
    """
    Visualize the graph with PFARS HQ and hospital locations highlighted.
    
    Args:
        G: NetworkX graph
        pfars_node: PFARS HQ node ID
        hospital_node: Hospital node ID
        title: Plot title
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot the base graph
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='gray', edge_linewidth=0.5, show=False)
    
    # Get coordinates for all nodes
    node_coords = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    # Debug print coordinates
    print("\nDebug - Visualization coordinates:")
    print(f"PFARS node {pfars_node}: {node_coords[pfars_node]}")
    print(f"Hospital node {hospital_node}: {node_coords[hospital_node]}")
    
    # Plot PFARS HQ
    pfars_coords = node_coords[pfars_node]
    ax.scatter(pfars_coords[0], pfars_coords[1], c='red', s=100, label='PFARS HQ')
    ax.annotate('PFARS HQ', (pfars_coords[0], pfars_coords[1]), 
                xytext=(10, 10), textcoords='offset points')
    
    # Plot Hospital
    hospital_coords = node_coords[hospital_node]
    ax.scatter(hospital_coords[0], hospital_coords[1], c='blue', s=100, label='Hospital')
    ax.annotate('Hospital', (hospital_coords[0], hospital_coords[1]), 
                xytext=(10, 10), textcoords='offset points')
    
    # Add legend and title
    ax.legend()
    ax.set_title(title)
    
    # Save the plot
    plt.savefig('data/processed/princeton_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph visualization saved to data/processed/princeton_graph.png")
    print(f"Total nodes: {len(G.nodes)}")
    print(f"Total edges: {len(G.edges)}")

def visualize_medical_trips(G, medical_trips_file=MEDICAL_TRIPS_FILE, title="Princeton Medical Trip Origins"):
    """
    Visualize all medical trip origin points on the graph.
    
    Args:
        G: NetworkX graph
        medical_trips_file: Path to the medical trips CSV file
        title: Plot title
    """
    # Read medical trips data
    trips_df = pd.read_csv(medical_trips_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot the base graph
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='gray', edge_linewidth=0.5, show=False)
    
    # Get coordinates for all nodes
    node_coords = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    # Plot each origin point
    for _, trip in trips_df.iterrows():
        # Find nearest node for origin
        
        # Plot origin point using lat/lon coordinates
        ax.scatter(trip['origin_lon'], trip['origin_lat'], c='red', s=50, alpha=0.5)
        
        # Print origin node info
        #print(f"Origin: {trip['origin_name']}, Node: {origin_node}, Coords: ({trip['origin_lat']}, {trip['origin_lon']})")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Call Origin')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    ax.set_title(title)
    
    # Save the plot
    plt.savefig('data/processed/princeton_medical_origins.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nMedical trip origins visualization saved to data/processed/princeton_medical_origins.png")
    print(f"Total origins plotted: {len(trips_df)}")

def generate_node_list(G):
    """
    Generate node ID to index mappings and save them to JSON files.
    These mappings are used by the simulator to convert between node IDs and indices.
    
    Args:
        G: NetworkX graph
    """
    node_list = list(G.nodes)
    node_id_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node_id = {idx: node for idx, node in enumerate(node_list)}
    
    # Create the matrices directory if it doesn't exist
    os.makedirs('data/matrices', exist_ok=True)
    
    # Convert dictionary keys to strings for JSON serialization
    node_id_to_idx_str = {str(k): v for k, v in node_id_to_idx.items()}
    idx_to_node_id_str = {str(k): int(v) for k, v in idx_to_node_id.items()}

    # Save to JSON files
    with open(os.path.join('data/matrices', 'node_id_to_idx.json'), 'w') as f:
        json.dump(node_id_to_idx_str, f, indent=2)
    
    with open(os.path.join('data/matrices', 'idx_to_node_id.json'), 'w') as f:
        json.dump(idx_to_node_id_str, f, indent=2)
    
    print(f"Generated node mappings for {len(node_list)} nodes")
    print(f"Saved node mappings to data/matrices/node_id_to_idx.json and data/matrices/idx_to_node_id.json")

def precompute_shortest_paths(G, weight='travel_time'):
    """
    Precompute the shortest paths for every source node in the graph.
    Returns a dictionary where for each source node, there is a dictionary mapping 
    each target node to a dict with 'travel_time' and 'path'.
    """
    shortest_paths = {}
    for source in G.nodes:
        lengths, paths = nx.single_source_dijkstra(G, source, weight=weight)
        shortest_paths[source] = {
            target: {
                'travel_time': lengths[target],
                'path': paths[target],
                'length': nx.path_weight(G, paths[target], weight='length')
            }
            for target in lengths
        }

    
    # Ensure the output directory exists
    output_dir = "data/matrices"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "path_cache.pkl")

    # Save the shortest paths cache to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(shortest_paths, f)

    # Print summary information
    num_nodes = len(shortest_paths)
    total_targets = sum(len(v) for v in shortest_paths.values())
    avg_targets = total_targets / num_nodes if num_nodes > 0 else 0
    print(f"Precomputed shortest paths for {num_nodes} nodes.")
    print(f"Total target entries: {total_targets} (average {avg_targets:.2f} per source).")
    print(f"Path cache saved to: {output_file}")

    return shortest_paths

def main():
    # Check if graph file exists, if not generate it
    if not os.path.exists(GRAPH_FILE):
        print("Generating new graph...")
        G = generate_graph(PRINCETON_BBOX, network_type='drive')
    else:
        print("Loading existing graph...")
        G = pickle.load(open(GRAPH_FILE, 'rb'))

    # Generate node list for mapping node IDs to indices
    #generate_node_list(G)

    #generate synthetic calls
    generate_demand_with_temporal_pattern(G, num_days=NUMDAYS)
    
    print("G is of type:", type(G))
    print(f"Original graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Get PFARS HQ node and hospital node
    print("\nDebug - PFARS HQ coordinates:")
    print(f"Lat: 40.361395, Lon: -74.664879")
    pfars_node = lat_lon_to_node(G, 40.361395, -74.664879)
    print(f"Found PFARS node: {pfars_node}")
    print(f"PFARS node coordinates: {G.nodes[pfars_node]['y']}, {G.nodes[pfars_node]['x']}")
    
    print("\nDebug - Hospital coordinates:")
    print(f"Lat: 40.340339, Lon: -74.623913")
    hospital_node = lat_lon_to_node(G, 40.340339, -74.623913)
    print(f"Found hospital node: {hospital_node}")
    print(f"Hospital node coordinates: {G.nodes[hospital_node]['y']}, {G.nodes[hospital_node]['x']}")
    
    # Precompute shortest paths
    precompute_shortest_paths(G)

    # Generate demand patterns using actual medical trips data
    #print('\nGenerating demand patterns...')
    #generate_demand_with_temporal_pattern(G, num_days=7)
    
    # Visualize the graph with critical nodes
    #visualize_graph(G, pfars_node, hospital_node, "Princeton Road Network")
    
    # Visualize medical trips
    #visualize_medical_trips(G, MEDICAL_TRIPS_FILE, "Princeton Medical Trip Origins")
    
    # Visualize synthetic calls
    #visualize_medical_trips(G, SYNTHETIC_CALLS_FILE, "Princeton Synthetic Call Origins")


if __name__ == "__main__":
    #print("this is working")
    main()