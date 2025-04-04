import os
import json
import pickle
import numpy as np
import networkx as nx

# Constants
GRAPH_FILE = 'data/processed/princeton_graph.gpickle'
OUTPUT_DIR = 'data/matrices'

def generate_node_mappings(G):
    """Generate node ID to index mappings and vice versa."""
    node_list = list(G.nodes)
    node_id_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node_id = {idx: node for idx, node in enumerate(node_list)}
    
    return node_id_to_idx, idx_to_node_id

def save_mappings(node_id_to_idx, idx_to_node_id):
    """Save mappings to JSON files."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert dictionary keys to strings for JSON serialization
    node_id_to_idx_str = {str(k): v for k, v in node_id_to_idx.items()}
    idx_to_node_id_str = {str(k): int(v) for k, v in idx_to_node_id.items()}
    
    # Save to JSON files
    with open(os.path.join(OUTPUT_DIR, 'node_id_to_idx.json'), 'w') as f:
        json.dump(node_id_to_idx_str, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'idx_to_node_id.json'), 'w') as f:
        json.dump(idx_to_node_id_str, f, indent=2)
    
    print(f"Mappings saved to {OUTPUT_DIR}")
    print(f"Number of nodes mapped: {len(node_id_to_idx)}")

def main():
    # Load graph
    if not os.path.exists(GRAPH_FILE):
        print(f"Error: Graph file {GRAPH_FILE} not found. Please run princeton_data_prep.py first.")
        return
    
    print(f"Loading graph from {GRAPH_FILE}...")
    G = pickle.load(open(GRAPH_FILE, 'rb'))
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Generate and save mappings
    print("Generating node mappings...")
    node_id_to_idx, idx_to_node_id = generate_node_mappings(G)
    save_mappings(node_id_to_idx, idx_to_node_id)

if __name__ == "__main__":
    main() 