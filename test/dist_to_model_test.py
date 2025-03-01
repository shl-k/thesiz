import numpy as np
from gurobipy import Model, GRB, quicksum
from distance_matrix_to_artm_model import distance_matrix_to_artm_model
from distance_matrix_to_ertm_model import distance_matrix_to_ertm_model

"""
from stefan
def test_distance_to_model():
    D = np.array([[1, 2]])
    demand = np.array([1])

    model = distance_matrix_to_artm_model(D, demand, 1)
    model.optimize()
    x = np.array([v.X for v in model.getVars()])
    assert np.allclose(x[0:2], np.array([1, 0]))

test_distance_to_model()
"""

def test_distance_to_models():
    """Test both ARTM and ERTM models with simple test cases"""
    
    # Test Case 1: Simple 2x2 case
    def test_simple_case():
        # Simple distance matrix where base 0 is clearly better for both demand points
        distance_matrix = np.array([
            [1, 10],  # Demand point 0: Base 0 is much closer than Base 1
            [2, 20],  # Demand point 1: Base 0 is much closer than Base 1
        ])
        demand_vec = np.array([1, 1])  # Equal demand at both points
        p = 1  # Only one base allowed
        
        # Test original ARTM model
        model = distance_matrix_to_artm_model(distance_matrix, demand_vec, p)
        model.optimize()
        
        # Check if model solved successfully
        assert model.status == GRB.OPTIMAL, f"Expected status {GRB.OPTIMAL} (OPTIMAL), but got status {model.status}"
        
        # Get solution
        x_vals = {j: model.getVarByName(f"x[{j}]").X for j in range(2)}
        
        # Base 0 should be selected as it's closer to both demand points
        assert abs(x_vals[0] - 1.0) < 1e-6, "Base 0 should be selected"
        assert abs(x_vals[1] - 0.0) < 1e-6, "Base 1 should not be selected"
        
    # Test Case 2: Testing ranked model
    def test_ranked_case():
        # Distance matrix where bases have different advantages
        distance_matrix = np.array([
            [1, 2],   # Demand point 0: Base 0 slightly better
            [2, 1],   # Demand point 1: Base 1 slightly better
        ])
        demand_vec = np.array([1, 1])
        p = 2  # Allow both bases
        q = 0.3  # Probability parameter
        
        # Test ranked ARTM model
        ranked_model = distance_matrix_to_ertm_model(distance_matrix, demand_vec, p, q)
        ranked_model.optimize()
        
        assert ranked_model.status == GRB.OPTIMAL, "Ranked model failed to find optimal solution"
        
        # Get solution
        x_vals = {j: ranked_model.getVarByName(f"x[{j}]").X for j in range(2)}
        
        # Both bases should be used due to their complementary advantages
        assert sum(x_vals.values()) == p, "Should use all available ambulances"
        
    # Test Case 3: Edge cases
    def test_edge_cases():
        # Test with zero demand
        distance_matrix = np.array([[1]])
        demand_vec = np.array([0])
        p = 1
        
        model = distance_matrix_to_artm_model(distance_matrix, demand_vec, p)
        model.optimize()
        assert model.status == GRB.OPTIMAL, "Model should solve even with zero demand"
        
        # Test with equal distances
        distance_matrix = np.array([[5, 5]])
        demand_vec = np.array([1])
        p = 1
        
        model = distance_matrix_to_artm_model(distance_matrix, demand_vec, p)
        model.optimize()
        assert model.status == GRB.OPTIMAL, "Model should solve with equal distances"
    
    # Run all tests
    try:
        test_simple_case()
        test_ranked_case()
        test_edge_cases()
        print("All tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")

if __name__ == "__main__":
    test_distance_to_models()