import osmnx as ox
import networkx as nx
from src.osm_graph import osm_graph

def test_osm_graph():
    """Test the OSM graph function with a simple location"""
    
    try:
        # Test with a location name
        G = osm_graph(location="Princeton University, NJ", network_type="drive")
        
        # Check that we got a valid graph
        assert isinstance(G, nx.Graph), "Result should be a NetworkX graph"
        assert len(G.nodes) > 0, "Graph should have nodes"
        assert len(G.edges) > 0, "Graph should have edges"
        
        # Check that the graph is undirected
        assert not G.is_directed(), "Graph should be undirected"
        
        print("OSM graph test passed successfully!")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_osm_graph() 