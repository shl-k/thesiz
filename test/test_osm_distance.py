import numpy as np
from src.osm_distance import osm_distance

def test_osm_distance():
    """Test the OSM distance function with a simple location"""
    
    try:
        # Test with a location name
        distance_matrix = osm_distance(location="Princeton University, NJ", network_type="drive")
        
        # Check that we got a valid distance matrix
        assert isinstance(distance_matrix, np.ndarray), "Result should be a NumPy array"
        assert distance_matrix.ndim == 2, "Result should be a 2D matrix"
        assert distance_matrix.shape[0] > 0, "Matrix should have rows"
        assert distance_matrix.shape[1] > 0, "Matrix should have columns"
        assert distance_matrix.shape[0] == distance_matrix.shape[1], "Matrix should be square"
        
        # Check that the diagonal is zero (distance from a node to itself)
        assert np.allclose(np.diag(distance_matrix), 0), "Diagonal elements should be zero"
        
        print("OSM distance test passed successfully!")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_osm_distance() 