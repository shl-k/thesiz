import numpy as np
from gurobipy import GRB
from distance_matrix_with_demand_to_ertm_model import distance_matrix_to_ranked_demand_model

def test_ranked_demand_model():
    """Test the ranked demand model with various scenarios"""
    
    def test_simple_demand_case():
        """Test with simple 2x2 case where demand can be fully met"""
        # Setup simple test case
        distance_matrix = np.array([
            [1, 10],  # Demand point 0: Base 0 is much closer
            [2, 1],   # Demand point 1: Base 1 is much closer
        ])
        intensity_vec = np.array([1, 1])      # Equal intensity weights
        demand_vec = np.array([2, 2])         # 2 calls at each point
        gamma_vec = np.array([100, 100])      # High penalty for unmet demand
        p = 4                                 # Enough ambulances to serve all demand
        q = 0.3
        
        model = distance_matrix_to_ranked_demand_model(
            distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q
        )
        model.optimize()
        
        assert model.status == GRB.OPTIMAL, f"Expected optimal solution, got status {model.status}"
        
        # Check if all demand is met (unmet should be 0)
        unmet_vals = {i: model.getVarByName(f"unmet[{i}]").X for i in range(2)}
        assert all(val < 1e-6 for val in unmet_vals.values()), "All demand should be met"
        
    def test_insufficient_capacity():
        """Test case where there aren't enough ambulances to meet all demand"""
        distance_matrix = np.array([
            [1, 2],
            [2, 1],
        ])
        intensity_vec = np.array([1, 1])
        demand_vec = np.array([10, 10])       # High demand
        gamma_vec = np.array([1, 1])          # Equal penalties
        p = 2                                 # Only 2 ambulances available
        q = 0.3
        
        model = distance_matrix_to_ranked_demand_model(
            distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q
        )
        model.optimize()
        
        assert model.status == GRB.OPTIMAL, f"Expected optimal solution, got status {model.status}"
        
        # Should have some unmet demand
        unmet_vals = {i: model.getVarByName(f"unmet[{i}]").X for i in range(2)}
        assert any(val > 0 for val in unmet_vals.values()), "Should have some unmet demand"
        
    def test_priority_based_allocation():
        """Test if model prioritizes high-gamma demand points"""
        distance_matrix = np.array([
            [1, 1],  # Equal distances
            [1, 1],  # Equal distances
        ])
        intensity_vec = np.array([1, 1])
        demand_vec = np.array([5, 5])         # Equal demand
        gamma_vec = np.array([1, 10])         # Point 1 has much higher priority
        p = 3                                 # Not enough for all demand
        q = 0.3
        
        model = distance_matrix_to_ranked_demand_model(
            distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q
        )
        model.optimize()
        
        # Check if high-priority point has less unmet demand
        unmet_vals = {i: model.getVarByName(f"unmet[{i}]").X for i in range(2)}
        assert unmet_vals[1] < unmet_vals[0], "Higher priority point should have less unmet demand"
        
    def test_edge_cases():
        """Test edge cases"""
        # Zero demand case
        distance_matrix = np.array([[1]])
        intensity_vec = np.array([1])
        demand_vec = np.array([0])
        gamma_vec = np.array([1])
        p = 1
        q = 0.3
        
        model = distance_matrix_to_ranked_demand_model(
            distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q
        )
        model.optimize()
        
        assert model.status == GRB.OPTIMAL, "Model should solve with zero demand"
        
        # Get unmet demand
        unmet = model.getVarByName("unmet[0]").X
        assert unmet < 1e-6, "Should have no unmet demand when demand is zero"
    
    # Run all tests
    try:
        test_simple_demand_case()
        test_insufficient_capacity()
        test_priority_based_allocation()
        test_edge_cases()
        print("All ranked demand model tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")

if __name__ == "__main__":
    test_ranked_demand_model()