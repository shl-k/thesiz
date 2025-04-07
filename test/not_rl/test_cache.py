import os
import pickle
import random
import networkx as nx

import os
import pickle
import random
import networkx as nx

def test_path_cache_apples(tol=1e-6, sample_size=10):
    graph_path = 'data/processed/princeton_graph.gpickle'
    cache_path = 'data/matrices/path_cache.pkl'
    
    if not os.path.exists(graph_path):
        print("Graph file not found:", graph_path)
        return
    if not os.path.exists(cache_path):
        print("Path cache file not found:", cache_path)
        return

    # Load graph and cache
    G = pickle.load(open(graph_path, 'rb'))
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    
    # Get list of nodes that exist in the cache (i.e. keys in cache)
    available_nodes = list(cache.keys())
    total_nodes = len(available_nodes)
    print(f"Testing path cache on {sample_size} random source nodes out of {total_nodes}")

    error_count = 0

    for _ in range(sample_size):
        source = random.choice(available_nodes)
        # Choose a random target from keys in the cache[source]
        target = random.choice(list(cache[source].keys()))
        if source == target:
            continue

        # Get cached values
        cached_entry = cache[source][target]
        cached_path = cached_entry.get('path', None)
        cached_travel_time = cached_entry.get('travel_time', None)
        cached_length = cached_entry.get('length', None)

        # Compute on the fly using the same criteria (using weight="travel_time" for path/travel_time)
        try:
            computed_path = nx.shortest_path(G, source=source, target=target, weight='travel_time')
            computed_travel_time = nx.shortest_path_length(G, source=source, target=target, weight='travel_time')
            computed_length = nx.path_weight(G, computed_path, weight='length')
        except Exception as e:
            print(f"Error computing values for {source} -> {target}: {e}")
            error_count += 1
            continue

        # Compare travel time
        if abs(cached_travel_time - computed_travel_time) > tol:
            print(f"Travel time mismatch for {source} -> {target}:")
            print(f"  Cached:   {cached_travel_time}")
            print(f"  Computed: {computed_travel_time}")
            error_count += 1

        # Compare length
        if cached_length is not None:
            if abs(cached_length - computed_length) > tol:
                print(f"Length mismatch for {source} -> {target}:")
                print(f"  Cached:   {cached_length}")
                print(f"  Computed: {computed_length}")
                error_count += 1
        else:
            print(f"Warning: 'length' key missing for {source} -> {target}")
            error_count += 1

        # Compare path lists.
        # Note: There may be multiple optimal paths.
        if cached_path != computed_path:
            # As a fallback, compare the computed travel times of each path.
            alt_travel_time = nx.path_weight(G, cached_path, weight='travel_time')
            if abs(alt_travel_time - computed_travel_time) > tol:
                print(f"Path mismatch for {source} -> {target}:")
                print(f"  Cached path:   {cached_path}")
                print(f"  Computed path: {computed_path}")
                error_count += 1

    if error_count == 0:
        print("All tests passed: The path cache matches computed values (apples to apples)!")
    else:
        print(f"Total errors found: {error_count}")

if __name__ == "__main__":
    test_path_cache_apples()