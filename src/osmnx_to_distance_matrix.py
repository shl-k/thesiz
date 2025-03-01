import osmnx as ox
import networkx as nx

def osmnx_to_distance_matrix(location=None, bbox=None, network_type='drive'):
    """
    Get distance matrix of graph using either a location name or bounding box coordinates.

    Parameters:
        location (str, optional): Name of the place (e.g., "Princeton, NJ").
        bbox (tuple, optional): Bounding box as (minx, miny, maxx, maxy).
        network_type (str, optional): Type of street network to retrieve. Default is 'drive'.
    
    Returns:
        nxm matrix of distances between demand point i in n and base j in m
    """

    if location:
        # Get bounding box from location name
        gdf = ox.geocode_to_gdf(location)
        bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
    
    if bbox:
        # Fetch graph using the bounding box (pass bbox as a tuple, not unpacked)
        G = ox.graph_from_bbox(bbox, network_type='drive')
    else:
        raise ValueError("Please provide either a location name or bounding box coordinates.")

    # Handle one-ways
    ccs = nx.strongly_connected_components(G)
    largest_cc = max(ccs, key=len)
    G = G.subgraph(largest_cc).copy()
    G_undirected = nx.Graph(G)

    # Compute the distance matrix (currently using floyd warshall)
    distance_matrix = nx.floyd_warshall_numpy(G_undirected)

    return distance_matrix


