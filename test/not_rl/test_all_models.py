"""
Run and compare all coverage models on a test instance.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# Import models
from src.models.artm_model import artm_model
from src.models.ertm_model import ertm_model
from src.models.ertm_demand_model import ertm_demand_model
from src.models.dsm_model import dsm_model

# Import utilities
from src.utils.geo_utils import get_osm_graph, create_distance_matrix

def test_all_models():
    """Test and compare all ambulance location models with a simple example."""
    
    print("Testing all ambulance location models...")
    location = "Princeton, NJ, USA"
    
    # Get the graph and distance matrix
    print(f"Getting graph and distance matrix for {location}...")
    G = get_osm_graph(location=location)
    D = create_distance_matrix(graph=G)
    
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
    print(f"Distance matrix shape: {D.shape}")
    
    # Generate demand points (random subset of nodes for this test)
    np.random.seed(42)  # For reproducibility
    num_demand_points = 50
    if num_demand_points > len(G.nodes):
        num_demand_points = len(G.nodes)
    
    demand_indices = np.random.choice(D.shape[0], num_demand_points, replace=False)
    demand = np.zeros(D.shape[0])
    demand[demand_indices] = np.random.uniform(0.1, 1.0, size=num_demand_points)
    demand = demand / demand.sum()  # Normalize
    
    # 1. Create a small test case
    print("\n1. Creating test data...")
    
    # Small distance matrix (3 demand points, 2 potential bases)
    distance_matrix = np.array([
        [500, 2000],   # Demand point 0: Base 0 is closer
        [3000, 1000],  # Demand point 1: Base 1 is closer
        [1500, 1500]   # Demand point 2: Equidistant
    ])
    
    # Demand vector
    demand_vec = np.array([2, 1, 3])  # Higher demand at points 0 and 2
    
    # For ERTM with demand, we need additional vectors
    intensity_vec = demand_vec.copy()  # Same as demand for simplicity
    gamma_vec = np.array([100, 100, 100])  # Equal penalty for unmet demand
    
    # Parameters
    p = 2  # Number of ambulances
    q = 0.3  # Probability parameter for ERTM
    r1 = 1000  # Primary threshold for DSM (meters)
    r2 = 2000  # Secondary threshold for DSM (meters)
    
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Demand vector: {demand_vec}")
    print(f"Parameters: p={p}, q={q}, r1={r1}, r2={r2}")
    
    # 2. Test ARTM model
    print("\n" + "=" * 50)
    print("2. Testing ARTM model...")
    print("=" * 50)
    
    artm = artm_model(distance_matrix, demand_vec, p)
    artm.optimize()
    
    if artm.status == gp.GRB.OPTIMAL:
        print(f"ARTM objective value: {artm.objVal}")
        
        # Get solution
        x_vals = {j: artm.getVarByName(f"x[{j}]").X for j in range(distance_matrix.shape[1])}
        
        # Print selected base locations
        selected_bases = [j for j, val in x_vals.items() if val > 0.5]
        print(f"Selected bases: {selected_bases}")
    else:
        print(f"ARTM model status: {artm.status}")
    
    # 3. Test ERTM model
    print("\n" + "=" * 50)
    print("3. Testing ERTM model...")
    print("=" * 50)
    
    ertm = ertm_model(distance_matrix, demand_vec, p, q)
    ertm.optimize()
    
    if ertm.status == gp.GRB.OPTIMAL:
        print(f"ERTM objective value: {ertm.objVal}")
        
        # Get solution
        x_vals = {j: ertm.getVarByName(f"x[{j}]").X for j in range(distance_matrix.shape[1])}
        
        # Print selected base locations and number of ambulances
        selected_bases = {j: int(val) for j, val in x_vals.items() if val > 0.5}
        print(f"Selected bases and ambulance counts: {selected_bases}")
    else:
        print(f"ERTM model status: {ertm.status}")
    
    # 4. Test ERTM with Demand model
    print("\n" + "=" * 50)
    print("4. Testing ERTM with Demand model...")
    print("=" * 50)
    
    ertm_d = ertm_demand_model(distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q)
    ertm_d.optimize()
    
    if ertm_d.status == gp.GRB.OPTIMAL:
        print(f"ERTM with Demand objective value: {ertm_d.objVal}")
        
        # Get solution
        x_vals = {j: ertm_d.getVarByName(f"x[{j}]").X for j in range(distance_matrix.shape[1])}
        unmet_vals = {i: ertm_d.getVarByName(f"unmet[{i}]").X for i in range(distance_matrix.shape[0])}
        
        # Print selected base locations and number of ambulances
        selected_bases = {j: int(val) for j, val in x_vals.items() if val > 0.5}
        print(f"Selected bases and ambulance counts: {selected_bases}")
        
        # Print unmet demand
        total_unmet = sum(unmet_vals.values())
        print(f"Total unmet demand: {total_unmet}")
    else:
        print(f"ERTM with Demand model status: {ertm_d.status}")
    
    # 5. Test DSM model
    print("\n" + "=" * 50)
    print("5. Testing DSM model...")
    print("=" * 50)
    
    dsm = dsm_model(distance_matrix, demand_vec, p, r1, r2)
    dsm.optimize()
    
    if dsm.status == gp.GRB.OPTIMAL:
        print(f"DSM objective value: {dsm.objVal}")
        
        # Get solution
        x_vals = {j: dsm.getVarByName(f"x[{j}]").X for j in range(distance_matrix.shape[1])}
        y_vals = {i: dsm.getVarByName(f"y[{i}]").X for i in range(distance_matrix.shape[0])}
        z_vals = {i: dsm.getVarByName(f"z[{i}]").X for i in range(distance_matrix.shape[0])}
        
        # Print selected base locations
        selected_bases = [j for j, val in x_vals.items() if val > 0.5]
        print(f"Selected bases: {selected_bases}")
        
        # Print coverage statistics
        covered = sum(1 for val in y_vals.values() if val > 0.5)
        double_covered = sum(1 for val in z_vals.values() if val > 0.5)
        print(f"Covered demand points: {covered} out of {distance_matrix.shape[0]}")
        print(f"Double-covered demand points: {double_covered} out of {distance_matrix.shape[0]}")
    else:
        print(f"DSM model status: {dsm.status}")
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    test_all_models() 