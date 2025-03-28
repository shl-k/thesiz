import numpy as np
from gurobipy import Model, GRB, quicksum
from src.artm_model import artm_model


def test_artm_model():
    """Test the Average Response Time Model with a simple test case"""
    
    # Simple test case
    distance_matrix = np.array([[1, 2]])  # One demand point, two potential bases
    demand_vec = np.array([1])  # Unit demand
    p = 1  # Allow only one base
    
    # Create and solve the model
    model = artm_model(distance_matrix, demand_vec, p)
    model.optimize()
    
    # Check if model solved successfully
    assert model.status == GRB.OPTIMAL, f"Expected status {GRB.OPTIMAL} (OPTIMAL), but got status {model.status}"
    
    # Get solution
    x_vals = {j: model.getVarByName(f"x[{j}]").X for j in range(2)}
    
    # Base 0 should be selected (has shorter distance)
    assert abs(x_vals[0] - 1.0) < 1e-6, "Base 0 should be selected"
    assert abs(x_vals[1] - 0.0) < 1e-6, "Base 1 should not be selected"
    
    print("ARTM model test passed successfully!")


if __name__ == "__main__":
    test_artm_model() 