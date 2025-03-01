import pytest
import networkx as nx
from osmnx_to_graph import osmnx_to_graph

def test_osmnx_to_graph():
    # Test with location
    G1 = osmnx_to_graph(location="Princeton, NJ")
    assert isinstance(G1, nx.Graph)
    assert len(G1.nodes) > 0
    assert len(G1.edges) > 0

    # Test with bbox (coordinates around Princeton, NJ)
    bbox = (-74.6772, 40.3373, -74.6551, 40.3520)  # Princeton area
    G2 = osmnx_to_graph(bbox=bbox)
    assert isinstance(G2, nx.Graph)
    assert len(G2.nodes) > 0
    assert len(G2.edges) > 0

    # Test error case - no inputs
    with pytest.raises(ValueError):
        osmnx_to_graph()

    # Test error case - invalid location
    with pytest.raises(Exception):
        osmnx_to_graph(location="ThisPlaceDoesNotExist12345") 