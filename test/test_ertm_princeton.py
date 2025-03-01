import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from src.distance_matrix_to_ertm_model import distance_matrix_to_ertm_model
from src.distance_matrix_to_artm_model import distance_matrix_to_artm_model
from src.osmnx_to_distance_matrix import osmnx_to_distance_matrix
from src.osmnx_to_graph import osmnx_to_graph

# Get Princeton graph and distance matrix
print('Getting Princeton graph and distance matrix...')
G_pton = osmnx_to_graph(location='Princeton, NJ', network_type='drive')
D_pton = osmnx_to_distance_matrix(location='Princeton, NJ', network_type='drive')
print(f'Matrix shape: {D_pton.shape}')

# Generate random demand vector (same for both models)
np.random.seed(42)  # For reproducibility
demand_vec = np.random.poisson(lam=2, size=D_pton.shape[0])

# Parameters
p = 2  # Number of ambulances
q = 0.3  # Probability parameter for backup coverage (ERTM only)

# Run ARTM model
print('\nSolving ARTM model...')
artm_model = distance_matrix_to_artm_model(D_pton, demand_vec, p)
artm_model.optimize()

print('\nARTM Results:')
print(f'Objective value (average response time): {artm_model.objVal}')

artm_locations = []
for v in artm_model.getVars():
    if v.varName.startswith('x[') and v.x > 0.9:
        idx = int(v.varName.split('[')[1].split(']')[0])
        artm_locations.append(idx)
        node = list(G_pton.nodes())[idx]
        lat = G_pton.nodes[node]['y']
        lon = G_pton.nodes[node]['x']
        print(f'ARTM Ambulance at node {idx}: Lat={lat}, Lon={lon}')

# Run ERTM model
print('\nSolving ERTM model...')
ertm_model = distance_matrix_to_ertm_model(D_pton, demand_vec, p, q)
ertm_model.optimize()

print('\nERTM Results:')
print(f'Objective value (expected response time): {ertm_model.objVal}')

ertm_locations = []
for v in ertm_model.getVars():
    if v.varName.startswith('x[') and v.x > 0.5:
        idx = int(v.varName.split('[')[1].split(']')[0])
        num_ambulances = int(v.x)
        ertm_locations.append(idx)
        node = list(G_pton.nodes())[idx]
        lat = G_pton.nodes[node]['y']
        lon = G_pton.nodes[node]['x']
        print(f'ERTM Ambulance at node {idx}: Lat={lat}, Lon={lon}')

# Plot both results on the same map
fig, ax = ox.plot_graph(G_pton, show=False, close=False)

# Plot ARTM locations in red
for idx in artm_locations:
    node = list(G_pton.nodes())[idx]
    lat = G_pton.nodes[node]['y']
    lon = G_pton.nodes[node]['x']
    ax.scatter(lon, lat, c='red', s=100, zorder=5, label='ARTM' if idx == artm_locations[0] else "")

# Plot ERTM locations in blue
for idx in ertm_locations:
    node = list(G_pton.nodes())[idx]
    lat = G_pton.nodes[node]['y']
    lon = G_pton.nodes[node]['x']
    ax.scatter(lon, lat, c='blue', s=100, zorder=5, label='ERTM' if idx == ertm_locations[0] else "")

plt.legend()
plt.title('Princeton - ARTM vs ERTM Ambulance Locations')
plt.show()

# Print location overlap analysis
common_locations = set(artm_locations) & set(ertm_locations)
print(f'\nLocation Analysis:')
print(f'Number of common locations: {len(common_locations)}')
if common_locations:
    print(f'Common location nodes: {common_locations}') 