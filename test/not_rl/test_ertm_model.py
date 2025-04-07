import numpy as np
from gurobipy import Model, GRB, quicksum
from src.ertm_model import ertm_model

def test_ertm_model():
    """Test the Expected Response Time Model with a simple test case"""
    
    # Simple test case
    distance_matrix = np.array([
        [1, 3],  # Demand point 0: Base 0 is closer
        [2, 1],  # Demand point 1: Base 1 is closer
    ])
    demand_vec = np.array([1, 1])  # Equal demand
    p = 2  # Allow two ambulances
    q = 0.3  # Probability parameter
    
    # Create and solve the model
    model = ertm_model(distance_matrix, demand_vec, p, q)
    model.optimize()
    
    # Check if model solved successfully
    assert model.status == GRB.OPTIMAL, f"Expected status {GRB.OPTIMAL} (OPTIMAL), but got status {model.status}"
    
    # Get solution
    x_vals = {j: model.getVarByName(f"x[{j}]").X for j in range(2)}
    
    # Both bases should have one ambulance
    assert abs(x_vals[0] - 1.0) < 1e-6, "Base 0 should have 1 ambulance"
    assert abs(x_vals[1] - 1.0) < 1e-6, "Base 1 should have 1 ambulance"
    
    print("ERTM model test passed successfully!")

if __name__ == "__main__":
    test_ertm_model() 