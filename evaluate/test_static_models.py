from src.static_models.ertm_model import ertm_model
from src.static_models.dsm_model import dsm_model
from src.static_models.artm_model import artm_model
import pickle
import osmnx as ox, networkx as nx, pandas as pd
import numpy as np
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import time
import sys
# Add parent directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

SYNTHETIC_CALLS_FILE = 'data/processed/synthetic_calls.csv'


def run_model(model_type, distance_matrix, demand_vec, p, **kwargs):
    num_nodes = len(demand_vec)
    
    # Start timing
    start_time = time.time()
    
    if model_type == 'ertm':
        q = kwargs.get('q', 0.3)
        model = ertm_model(distance_matrix, demand_vec, p, q)
    elif model_type == 'dsm':
        r1 = kwargs.get('r1', 300) 
        r2 = kwargs.get('r2', 600)
        model = dsm_model(distance_matrix, demand_vec, p, r1, r2)
    elif model_type == 'artm':
        model = artm_model(distance_matrix, demand_vec, p)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.optimize()
    
    # End timing
    end_time = time.time()
    solve_time = end_time - start_time
    
    if model.status == GRB.OPTIMAL:
        x = model.getVars()
        ambulance_placement = {j: int(x[j].x) for j in range(num_nodes)}
        obj_value = model.objVal
        return {
            'objective': obj_value,
            'placement': ambulance_placement,
            'solve_time': solve_time
        }
    return None

def visualize_trips(G, trips_data, result=None, model_name=None, p=None):
    """
    Visualize all medical trip origin points on the graph with optimal ambulance locations.
    
    Args:
        G: NetworkX graph
        trips_data: Either a DataFrame containing trip data or a path to a CSV file
        result: Dictionary containing optimal ambulance placement from optimization
        model_name: Name of the model used (for title)
        p: Number of ambulances
    """
    # Read medical trips data if a file path is provided
    if isinstance(trips_data, str):
        trips_df = pd.read_csv(trips_data)
    else:
        trips_df = trips_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot the base graph
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='gray', edge_linewidth=0.5, show=False)
    
    # Get coordinates for all nodes
    node_coords = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    # Plot each origin point
    for _, trip in trips_df.iterrows():
        # Plot origin point using lat/lon coordinates
        ax.scatter(trip['origin_lon'], trip['origin_lat'], c='red', s=trip['intensity']*2, alpha=0.5)
    
    # Plot PFARS HQ
    pfars_coords = node_coords[241]
    ax.scatter(pfars_coords[0], pfars_coords[1], c='green', s=75, alpha=0.95)
    
    # Plot Hospital
    hospital_coords = node_coords[1293]
    ax.scatter(hospital_coords[0], hospital_coords[1], c='blue', s=75, alpha=0.95)
    
    # Plot optimal ambulance locations if provided
    if result and 'placement' in result:
        for idx, count in result['placement'].items():
            if count > 0:
                # Get the actual node ID from the graph
                node = list(G.nodes())[idx]
                if node in node_coords:
                    x, y = node_coords[node]
                    ax.scatter(x, y, c='purple', s=75, alpha=0.95)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Call Origin'), 
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Hospital'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='PFARS HQ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
               markersize=10, label='Ambulance')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title if model info provided
    if model_name and p is not None:
        ax.set_title(f'{model_name} - Optimal Ambulance Locations (p={p})', fontsize=14)
    
    # Save the plot
    if model_name and p is not None:
        save_path = f'results/{model_name.lower()}_princeton_p{p}.png'
    else:
        save_path = 'results/uhoh.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {save_path}")
    print(f"Total origins plotted: {len(trips_df)}")

def main():
    print("Loading graph...")
    with open('data/processed/princeton_graph.gpickle', 'rb') as f:
        G = pickle.load(f)


    '''
    print("Loading synthetic calls...")
    synthetic_calls = pd.read_csv(SYNTHETIC_CALLS_FILE)

    # Calculate distance matrix for the full graph
    distance_matrix = nx.floyd_warshall_numpy(G, weight='length')
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # Create demand vector for the full graph
    demand_vec = np.zeros(len(G.nodes))
    demand_data = synthetic_calls.groupby('origin_node').size().reset_index(name='demand')

    # Create mapping from node IDs to indices in the full graph
    node_to_idx = {node: idx for idx, node in enumerate(sorted(G.nodes))}

    for _, row in demand_data.iterrows():
        node = row['origin_node']
        if node in node_to_idx:  # Make sure node exists in graph
            demand_vec[node_to_idx[node]] = row['demand']

    p = 3
    models = ['ertm']  # or ['dsm', 'ertm', 'artm'] to run all

    for model_type in models:
        print(f"\nRunning {model_type.upper()} model with p={p}")
        result = run_model(model_type, distance_matrix, demand_vec, p)

        if result:
            print(f"Objective value: {result['objective']:.2f}")
            print(f"Solve time: {result['solve_time']:.2f} seconds")
            print("Ambulance placement:")
            for node, count in result['placement'].items():
                if count > 0:
                    print(f"  Node {node}: {count} ambulances")

            # Visualize with optimal ambulance locations
            visualize_trips(G, synthetic_calls, result, model_type.upper(), p)
        else:
            print("No optimal solution found.")
    '''

if __name__ == "__main__":
    main()
