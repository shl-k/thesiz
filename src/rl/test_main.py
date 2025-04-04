# test_main.py
from AmbulanceEnv import AmbulanceEnv
import networkx as nx
import numpy as np
import json



# 1) Load or create a small Nx graph
# E.g. trivial graph with 5 nodes
G = nx.DiGraph()
G.add_nodes_from([101, 102, 103, 104, 105])
# Add some edges with 'travel_time'
G.add_edge(101, 102, travel_time=10)
G.add_edge(102, 103, travel_time=15)
G.add_edge(103, 104, travel_time=5)
G.add_edge(104, 105, travel_time=20)
G.add_edge(105, 101, travel_time=25)
G.add_node(999)  
G.add_edge(105, 999, travel_time=30)
G.add_edge(999, 101, travel_time=30)

# Create a simple node mapping that's consistent with our test graph
node_list = list(G.nodes)
node_id_to_idx = {node: idx for idx, node in enumerate(node_list)}
idx_to_node_id = {idx: node for idx, node in enumerate(node_list)}

# Convert to string keys for JSON compatibility
node_id_to_idx_str = {str(k): v for k, v in node_id_to_idx.items()}
idx_to_node_id_str = {str(k): int(v) for k, v in idx_to_node_id.items()}

# Save to temporary JSON files for the environment to use
import tempfile
import os

temp_dir = tempfile.mkdtemp()
idx_to_node_path = os.path.join(temp_dir, "idx_to_node_id.json")
node_to_idx_path = os.path.join(temp_dir, "node_id_to_idx.json")

with open(idx_to_node_path, 'w') as f:
    json.dump(idx_to_node_id_str, f)

with open(node_to_idx_path, 'w') as f:
    json.dump(node_id_to_idx_str, f)

print(f"Created temporary node mappings:")
print(f"Node ID to Index: {node_id_to_idx}")
print(f"Index to Node ID: {idx_to_node_id}")

# Create CSV file with test calls
import pandas as pd
test_calls = pd.DataFrame([
    {"day": 1, "second_of_day": 10, "origin_node": 101, "destination_node": 999, "intensity": 5, "origin_lat": 0, "origin_lon": 0},
    {"day": 1, "second_of_day": 40, "origin_node": 103, "destination_node": 999, "intensity": 2, "origin_lat": 0, "origin_lon": 0}
])
test_calls_path = os.path.join(temp_dir, "test_calls.csv")
test_calls.to_csv(test_calls_path, index=False)

env = AmbulanceEnv(
    graph=G,
    call_data_path=test_calls_path,
    num_ambulances=4,
    base_location=101,
    hospital_node=999,
    idx_to_node_path=idx_to_node_path,
    node_to_idx_path=node_to_idx_path,
    verbose=True
)

# 3) Import the test function
from test_env import test_ambulance_env
test_ambulance_env(env, max_steps=50)

# Add simple verification of node mappings at the end
print("\n=== Verifying Node Mappings ===")
print(f"Test graph nodes: {list(G.nodes())}")
print(f"Node ID to Index mapping: {node_id_to_idx}")
print(f"Index to Node ID mapping: {idx_to_node_id}")

# Test bidirectional mapping for each node
print("\nTesting bidirectional mapping:")
for node_id in G.nodes():
    idx = node_id_to_idx[node_id]
    mapped_back = idx_to_node_id[idx]
    print(f"Node ID: {node_id} → Index: {idx} → Mapped back: {mapped_back} | Match: {node_id == mapped_back}")

# Clean up temporary files
import shutil
shutil.rmtree(temp_dir)
