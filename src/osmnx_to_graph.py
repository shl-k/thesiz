import osmnx as ox
import networkx as nx

def osmnx_to_graph(location=None, bbox=None, network_type='drive'):
    """
    Plot a street network graph using either a location name or bounding box coordinates.

    Parameters:
        location (str, optional): Name of the place (e.g., "Princeton, NJ").
        bbox (tuple, optional): Bounding box as (minx, miny, maxx, maxy).
        network_type (str, optional): Type of street network to retrieve. Default is 'drive'.
    
    Returns:
        NetworkX graph with OSM attributes
    """
    
    if location:
        # Get bounding box from location name
        gdf = ox.geocode_to_gdf(location)
        bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
    
    if bbox:
        # Fetch graph using the bounding box (pass bbox as a tuple, not unpacked)
        G = ox.graph_from_bbox(bbox, network_type=network_type)
    else:
        raise ValueError("Please provide either a location name or bounding box coordinates.")
    
    # Convert to undirected using NetworkX's method
    G_undirected = G.to_undirected()
    
    return G_undirected



