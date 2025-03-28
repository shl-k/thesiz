import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.geo_utils import osm_graph, create_distance_matrix
from src.data.princeton_data_prep import sparsify_graph, generate_realistic_demand

# Get Princeton graph
print('Getting Princeton graph...')
G_pton_original = osm_graph(location='Princeton, NJ', network_type='drive')

# Basic graph statistics for original graph
print('\nOriginal Graph Statistics:')
print(f'Number of nodes: {len(G_pton_original.nodes())}')
print(f'Number of edges: {len(G_pton_original.edges())}')

# Sparsify the graph
print('\nSparsifying graph...')
G_pton = sparsify_graph(G_pton_original, min_edge_length=30, simplify=True)

# Basic graph statistics for sparsified graph
print('\nSparsified Graph Statistics:')
print(f'Number of nodes: {len(G_pton.nodes())}')
print(f'Number of edges: {len(G_pton.edges())}')
print(f'Reduction in nodes: {(1 - len(G_pton.nodes()) / len(G_pton_original.nodes())) * 100:.2f}%')
print(f'Reduction in edges: {(1 - len(G_pton.edges()) / len(G_pton_original.edges())) * 100:.2f}%')

# Check edge attributes to confirm units
print('\nEdge Length Statistics (Sparsified Graph):')
edge_lengths = []
for u, v, data in G_pton.edges(data=True):
    if 'length' in data:
        edge_lengths.append(data['length'])

if edge_lengths:
    print(f'Min edge length: {min(edge_lengths):.2f}')
    print(f'Max edge length: {max(edge_lengths):.2f}')
    print(f'Mean edge length: {np.mean(edge_lengths):.2f}')
    print(f'Median edge length: {np.median(edge_lengths):.2f}')
    
    # Plot histogram of edge lengths
    plt.figure(figsize=(10, 6))
    plt.hist(edge_lengths, bins=50)
    plt.title('Distribution of Edge Lengths in Sparsified Princeton Network')
    plt.xlabel('Edge Length (meters)')
    plt.ylabel('Frequency')
    plt.savefig('princeton_edge_lengths_sparsified.png')
    plt.close()
else:
    print('No length attribute found in edges')

# Get distance matrix for sparsified graph
print('\nCalculating distance matrix for sparsified graph...')
G_pton, D_pton, node_to_index, index_to_node = create_distance_matrix(graph=G_pton, save_file=True)

# Analyze distance matrix
print('\nDistance Matrix Statistics:')
print(f'Shape: {D_pton.shape}')
print(f'Min distance: {D_pton.min():.2f} meters')
print(f'Max distance: {D_pton.max():.2f} meters')
print(f'Mean distance: {D_pton.mean():.2f} meters')
print(f'Median distance: {np.median(D_pton):.2f} meters')

# Check how many nodes are within different distance thresholds
for dist in [100, 500, 1000, 2000]:
    coverage = np.sum(D_pton <= dist) / (D_pton.shape[0] * D_pton.shape[1]) * 100
    print(f'Percentage of node pairs within {dist}m: {coverage:.2f}%')

# Generate realistic demand patterns
print('\nGenerating realistic demand patterns...')
demand_vec, intensity_vec, gamma_vec = generate_realistic_demand(G_pton)

print('\nDemand Statistics:')
print(f'Total demand: {demand_vec.sum()} calls')
print(f'Min demand: {demand_vec.min()}')
print(f'Max demand: {demand_vec.max()}')
print(f'Mean demand: {demand_vec.mean():.2f}')

print('\nIntensity Statistics:')
print(f'Min intensity: {intensity_vec.min():.2f}')
print(f'Max intensity: {intensity_vec.max():.2f}')
print(f'Mean intensity: {intensity_vec.mean():.2f}')

print('\nPriority (Gamma) Statistics:')
print(f'Min priority: {gamma_vec.min():.2f}')
print(f'Max priority: {gamma_vec.max():.2f}')
print(f'Mean priority: {gamma_vec.mean():.2f}')

# Plot the graphs to visualize
fig, ax = ox.plot_graph(G_pton_original, node_size=5, figsize=(12, 10), show=False, close=False)
plt.title('Original Princeton Road Network')
plt.savefig('princeton_network_original.png')
plt.close()

fig, ax = ox.plot_graph(G_pton, node_size=5, figsize=(12, 10), show=False, close=False)
plt.title('Sparsified Princeton Road Network')
plt.savefig('princeton_network_sparsified.png')
plt.close()

# Create a heatmap of demand
node_values = {node: demand_vec[i] for i, node in enumerate(G_pton.nodes)}
fig, ax = ox.plot_graph(
    G_pton, 
    node_color=[node_values[node] for node in G_pton.nodes],
    node_size=20,
    edge_linewidth=0.5,
    node_cmap=plt.cm.plasma,
    figsize=(12, 10),
    show=False,
    close=False
)
plt.title('Princeton Demand Heatmap')
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Demand')
plt.savefig('princeton_demand_heatmap.png')
plt.close()

print('\nAnalysis complete! Check the output images for visualizations.') 