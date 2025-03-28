import numpy as np
from gurobipy import GRB
from src.models.ertm_demand_model import ertm_demand_model

def test_ertm_demand_model():
    """Test the ERTM with demand model with various scenarios"""
    
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
        
        # Create and solve model
        model = ertm_demand_model(distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q)
        model.optimize()
        
        # Check if model solved successfully
        assert model.status == GRB.OPTIMAL, f"Expected status {GRB.OPTIMAL} (OPTIMAL), but got status {model.status}"
        
        # Get solution
        x_vals = {j: model.getVarByName(f"x[{j}]").X for j in range(2)}
        unmet_vals = {i: model.getVarByName(f"unmet[{i}]").X for i in range(2)}
        
        # Both bases should have ambulances
        assert x_vals[0] >= 2, "Base 0 should have at least 2 ambulances"
        assert x_vals[1] >= 2, "Base 1 should have at least 2 ambulances"
        
        # No unmet demand
        assert abs(unmet_vals[0]) < 1e-6, "Demand point 0 should have no unmet demand"
        assert abs(unmet_vals[1]) < 1e-6, "Demand point 1 should have no unmet demand"
    
    def test_insufficient_capacity():
        """Test with insufficient ambulance capacity"""
        # Setup test case with limited ambulances
        distance_matrix = np.array([
            [1, 10],  # Demand point 0: Base 0 is much closer
            [2, 1],   # Demand point 1: Base 1 is much closer
        ])
        intensity_vec = np.array([1, 1])      # Equal intensity weights
        demand_vec = np.array([2, 2])         # 2 calls at each point
        gamma_vec = np.array([100, 100])      # High penalty for unmet demand
        p = 3                                 # Not enough ambulances for all demand
        q = 0.3
        
        # Create and solve model
        model = ertm_demand_model(distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q)
        model.optimize()
        
        # Check if model solved successfully
        assert model.status == GRB.OPTIMAL, f"Expected status {GRB.OPTIMAL} (OPTIMAL), but got status {model.status}"
        
        # Get solution
        x_vals = {j: model.getVarByName(f"x[{j}]").X for j in range(2)}
        unmet_vals = {i: model.getVarByName(f"unmet[{i}]").X for i in range(2)}
        
        # Total ambulances should equal p
        assert abs(sum(x_vals.values()) - p) < 1e-6, f"Total ambulances should be {p}"
        
        # Total unmet demand should be 1
        assert abs(sum(unmet_vals.values()) - 1) < 1e-6, "Total unmet demand should be 1"
    
    def test_priority_based_allocation():
        """Test with different priorities for demand points"""
        # Setup test case with different priorities
        distance_matrix = np.array([
            [1, 10],  # Demand point 0: Base 0 is much closer
            [2, 1],   # Demand point 1: Base 1 is much closer
        ])
        intensity_vec = np.array([1, 1])      # Equal intensity weights
        demand_vec = np.array([2, 2])         # 2 calls at each point
        gamma_vec = np.array([200, 100])      # Higher penalty for point 0
        p = 3                                 # Not enough ambulances for all demand
        q = 0.3
        
        # Create and solve model
        model = ertm_demand_model(distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q)
        model.optimize()
        
        # Check if model solved successfully
        assert model.status == GRB.OPTIMAL, f"Expected status {GRB.OPTIMAL} (OPTIMAL), but got status {model.status}"
        
        # Get solution
        unmet_vals = {i: model.getVarByName(f"unmet[{i}]").X for i in range(2)}
        
        # Point 1 should have more unmet demand due to lower priority
        assert unmet_vals[0] < unmet_vals[1], "Point 1 should have more unmet demand due to lower priority"
    
    # Run all tests
    try:
        test_simple_demand_case()
        test_insufficient_capacity()
        test_priority_based_allocation()
        print("All ERTM with demand tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")

if __name__ == "__main__":
    test_ertm_demand_model() 