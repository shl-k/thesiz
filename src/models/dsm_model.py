from gurobipy import Model, GRB, quicksum
import numpy as np

def dsm_model(distance_matrix, demand_vec, p, r1, r2, w1=1, w2=2):
    """
    Double Standard Model (DSM) for ambulance location.
    
    Inputs: 
        distance_matrix (np array, shape: [demand_points, num_bases]) (units: meters)
        demand_vec (np vec: shape: [demand_points]) - demand at each point
        p (int): maximum number of ambulances/bases
        r1 (float): primary (strict) response time threshold for double coverage
        r2 (float): secondary (less strict) response time threshold, with r2 > r1
        w1 (float): weight for single coverage (default: 1)
        w2 (float): weight for double coverage (default: 2)
    
    Outputs:
        Gurobi model for Double Standard Model (DSM)
    """
    # Parameters
    num_demand_points = distance_matrix.shape[0]  # Number of demand points
    num_bases = distance_matrix.shape[1]          # Number of potential base locations
    
    # Sets
    I = list(range(num_demand_points))  # Demand points
    J = list(range(num_bases))          # Potential base locations
    
    # Parameters required for optimization
    t = {(i, j): distance_matrix[i, j] for j in J for i in I}  # Travel time between base j and demand point i
    d = {i: demand_vec[i] for i in I}                          # Demand at each location
    
    # Initialize model
    model = Model("Double Standard Model")
    
    # Decision Variables
    x = model.addVars(J, vtype=GRB.BINARY, name="x")  # 1 if a base is opened at location j
    y = model.addVars(I, vtype=GRB.BINARY, name="y")  # 1 if demand point i is covered (within r2)
    z = model.addVars(I, vtype=GRB.BINARY, name="z")  # 1 if demand point i is double covered (within r1)
    
    # Objective: Maximize weighted coverage
    model.setObjective(
        quicksum(d[i] * (w1 * y[i] + w2 * z[i]) for i in I),
        GRB.MAXIMIZE
    )
    
    # Constraints
    # 1. Ensure basic coverage (y_i = 1 only if at least one base within r2)
    for i in I:
        model.addConstr(
            quicksum(x[j] for j in J if t[i, j] <= r2) >= y[i],
            name=f"BasicCoverage_{i}"
        )
    
    # 2. Ensure double coverage (z_i = 1 only if at least two bases within r1)
    for i in I:
        model.addConstr(
            quicksum(x[j] for j in J if t[i, j] <= r1) >= 2 * z[i],
            name=f"DoubleCoverage_{i}"
        )
    
    # 3. Ensure the total number of bases
    model.addConstr(
        quicksum(x[j] for j in J) == p,
        name="BaseLimit"
    )
    
    return model

# For backward compatibility
distance_matrix_to_dsm_model = dsm_model 