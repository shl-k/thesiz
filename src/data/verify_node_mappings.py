import os
import json
import pickle
import networkx as nx

# Constants
GRAPH_FILE = 'data/processed/princeton_graph.gpickle'
MAPPING_DIR = 'data/matrices'
NODE_ID_TO_IDX_FILE = os.path.join(MAPPING_DIR, 'node_id_to_idx.json')
IDX_TO_NODE_ID_FILE = os.path.join(MAPPING_DIR, 'idx_to_node_id.json')

def load_mappings():
    """Load node mappings from JSON files."""
    # Check if files exist
    if not os.path.exists(NODE_ID_TO_IDX_FILE):
        print(f"Error: Mapping file {NODE_ID_TO_IDX_FILE} not found.")
        return None, None
    
    if not os.path.exists(IDX_TO_NODE_ID_FILE):
        print(f"Error: Mapping file {IDX_TO_NODE_ID_FILE} not found.")
        return None, None
    
    # Load mappings
    with open(NODE_ID_TO_IDX_FILE, 'r') as f:
        node_id_to_idx_str = json.load(f)
    
    with open(IDX_TO_NODE_ID_FILE, 'r') as f:
        idx_to_node_id_str = json.load(f)
    
    # Convert string keys back to integers
    node_id_to_idx = {int(k): v for k, v in node_id_to_idx_str.items()}
    idx_to_node_id = {int(k): int(v) for k, v in idx_to_node_id_str.items()}
    
    return node_id_to_idx, idx_to_node_id

def verify_mappings(G, node_id_to_idx, idx_to_node_id):
    """Verify that the mappings are correct and complete."""
    print("\n=== Verifying Node Mappings ===")
    
    # Check 1: All graph nodes are in the mapping
    all_nodes_in_mapping = all(node in node_id_to_idx for node in G.nodes)
    print(f"1. All graph nodes in mapping: {all_nodes_in_mapping}")
    
    if not all_nodes_in_mapping:
        missing_nodes = [node for node in G.nodes if node not in node_id_to_idx]
        print(f"   Missing nodes: {missing_nodes[:5]}... ({len(missing_nodes)} total)")
    
    # Check 2: Mapping length matches number of nodes
    correct_length = len(node_id_to_idx) == len(G.nodes) and len(idx_to_node_id) == len(G.nodes)
    print(f"2. Mapping length matches graph nodes: {correct_length}")
    print(f"   Graph nodes: {len(G.nodes)}, node_id_to_idx: {len(node_id_to_idx)}, idx_to_node_id: {len(idx_to_node_id)}")
    
    # Check 3: Bidirectional mapping works
    if len(node_id_to_idx) > 0:
        # Sample 5 random nodes to test
        import random
        test_nodes = random.sample(list(G.nodes), min(5, len(G.nodes)))
        
        print("\n3. Testing bidirectional mapping with sample nodes:")
        for node_id in test_nodes:
            idx = node_id_to_idx[node_id]
            mapped_back = idx_to_node_id[idx]
            print(f"   Node ID: {node_id} → Index: {idx} → Mapped back: {mapped_back} | Match: {node_id == mapped_back}")
    
    # Check 4: Indices are continuous from 0
    indices = sorted(idx_to_node_id.keys())
    continuous = indices == list(range(len(indices)))
    print(f"\n4. Indices are continuous from 0 to {len(indices)-1}: {continuous}")
    if not continuous:
        gaps = [i for i in range(len(indices)) if i not in indices]
        print(f"   Gaps in indices: {gaps[:5]}... ({len(gaps)} total)")
    
    # Overall verification
    all_checks_passed = all_nodes_in_mapping and correct_length and continuous
    print(f"\n=== All checks passed: {all_checks_passed} ===")
    
    return all_checks_passed

def main():
    # Load node mappings
    print("Loading node mappings...")
    node_id_to_idx, idx_to_node_id = load_mappings()
    
    if node_id_to_idx is None or idx_to_node_id is None:
        print("Failed to load mappings. Please run save_node_mappings.py first.")
        return
    
    print(f"Loaded mappings with {len(node_id_to_idx)} nodes.")
    
    # Load graph
    if not os.path.exists(GRAPH_FILE):
        print(f"Error: Graph file {GRAPH_FILE} not found.")
        return
    
    print(f"Loading graph from {GRAPH_FILE}...")
    G = pickle.load(open(GRAPH_FILE, 'rb'))
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Verify mappings
    verify_mappings(G, node_id_to_idx, idx_to_node_id)

if __name__ == "__main__":
    main() 