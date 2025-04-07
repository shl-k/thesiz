import numpy as np
from gurobipy import Model, GRB, quicksum
from src.models.dsm_model import dsm_model

def test_dsm_model():
    """Test the Double Standard Model with simple test cases"""
    
    # Test Case 1: Simple coverage case
    def test_simple_coverage():
        # Distance matrix where some points are within coverage thresholds
        distance_matrix = np.array([
            [500, 2000],   # Demand point 0: Base 0 within r1, Base 1 within r2
            [3000, 1000],  # Demand point 1: Base 0 within r2, Base 1 within r1
            [5000, 5000]   # Demand point 2: Neither base within thresholds
        ])
        demand_vec = np.array([1, 1, 1])  # Equal demand at all points
        p = 2  # Allow both bases
        r1 = 1500  # Primary threshold (strict)
        r2 = 4000  # Secondary threshold (relaxed)
        
        # Test DSM model
        model = dsm_model(distance_matrix, demand_vec, p, r1, r2)
        model.optimize()
        
        # Check if model solved successfully
        assert model.status == GRB.OPTIMAL, f"Expected status {GRB.OPTIMAL} (OPTIMAL), but got status {model.status}"
        
        # Get solution
        x_vals = {j: model.getVarByName(f"x[{j}]").X for j in range(2)}
        y_vals = {i: model.getVarByName(f"y[{i}]").X for i in range(3)}
        z_vals = {i: model.getVarByName(f"z[{i}]").X for i in range(3)}
        
        # Both bases should be selected
        assert abs(x_vals[0] - 1.0) < 1e-6, "Base 0 should be selected"
        assert abs(x_vals[1] - 1.0) < 1e-6, "Base 1 should be selected"
        
        # Demand points 0 and 1 should be covered (within r2)
        assert abs(y_vals[0] - 1.0) < 1e-6, "Demand point 0 should be covered"
        assert abs(y_vals[1] - 1.0) < 1e-6, "Demand point 1 should be covered"
        assert abs(y_vals[2] - 0.0) < 1e-6, "Demand point 2 should not be covered"
        
        # No demand point should be double-covered (would need two bases within r1)
        assert abs(z_vals[0] - 0.0) < 1e-6, "Demand point 0 should not be double-covered"
        assert abs(z_vals[1] - 0.0) < 1e-6, "Demand point 1 should not be double-covered"
        assert abs(z_vals[2] - 0.0) < 1e-6, "Demand point 2 should not be double-covered"
    
    # Test Case 2: Double coverage case
    def test_double_coverage():
        # Distance matrix where one point can be double-covered
        distance_matrix = np.array([
            [1000, 1000],  # Demand point 0: Both bases within r1
            [3000, 1000],  # Demand point 1: Only Base 1 within r1
        ])
        demand_vec = np.array([2, 1])  # Higher demand at point 0
        p = 2  # Allow both bases
        r1 = 1500  # Primary threshold
        r2 = 4000  # Secondary threshold
        
        # Test DSM model
        model = dsm_model(distance_matrix, demand_vec, p, r1, r2)
        model.optimize()
        
        assert model.status == GRB.OPTIMAL, "Model failed to find optimal solution"
        
        # Get solution
        x_vals = {j: model.getVarByName(f"x[{j}]").X for j in range(2)}
        y_vals = {i: model.getVarByName(f"y[{i}]").X for i in range(2)}
        z_vals = {i: model.getVarByName(f"z[{i}]").X for i in range(2)}
        
        # Both bases should be selected
        assert abs(x_vals[0] - 1.0) < 1e-6, "Base 0 should be selected"
        assert abs(x_vals[1] - 1.0) < 1e-6, "Base 1 should be selected"
        
        # Both demand points should be covered
        assert abs(y_vals[0] - 1.0) < 1e-6, "Demand point 0 should be covered"
        assert abs(y_vals[1] - 1.0) < 1e-6, "Demand point 1 should be covered"
        
        # Demand point 0 should be double-covered
        assert abs(z_vals[0] - 1.0) < 1e-6, "Demand point 0 should be double-covered"
        assert abs(z_vals[1] - 0.0) < 1e-6, "Demand point 1 should not be double-covered"
    
    # Run all tests
    try:
        test_simple_coverage()
        test_double_coverage()
        print("All DSM tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")

if __name__ == "__main__":
    test_dsm_model() 