from gurobipy import Model, GRB, quicksum
import numpy as np

def distance_matrix_to_ertm_model(distance_matrix, demand_vec, p, q):
    """
    Inputs: 
        distance_matrix (np array, shape: [demand_points, num_bases]) (units: meters)
        demand_vec (np vec: shape: [demand_points])
        p (int): maximum number of ambulances
        q (float): probability parameter for backup coverage (0 < q < 1)

    Outputs:
        Gurobi model for Ranked ERTM
    """
    # Parameters
    num_demand_points = distance_matrix.shape[0]
    num_bases = distance_matrix.shape[1]
    
    # Sets
    I = list(range(num_demand_points))  # Demand points
    J = list(range(num_bases))          # Potential base locations
    K = list(range(1, p+1))            # Ranks (1 to p)

    # Parameters required for optimization
    t = {(i, j): distance_matrix[i, j] for j in J for i in I}
    d = {i: demand_vec[i] for i in I}
    
    # Initialize model
    model = Model("Ranked Expected Response Time Model")

    # Decision Variables
    x = model.addVars(J, vtype=GRB.INTEGER, name="x")  # Number of ambulances at base j
    z = model.addVars(I, J, K, vtype=GRB.BINARY, name="z")  # 1 if base j serves demand i as kth nearest

    # Objective: Minimize weighted response time with backup coverage
    # First term: response times for ranks 1 to p-1
    obj1 = quicksum(
        d[i] * t[i, j] * (1-q) * (q**(k-1)) * z[i, j, k]
        for i in I for j in J for k in range(1, p)
    )
    # Second term: response time for rank p
    obj2 = quicksum(
        d[i] * t[i, j] * (q**(p-1)) * z[i, j, p]
        for i in I for j in J
    )
    model.setObjective(obj1 + obj2, GRB.MINIMIZE)

    # Constraints
    # 1. Each demand point must have exactly one kth nearest ambulance for each rank
    for i in I:
        for k in K:
            model.addConstr(
                quicksum(z[i, j, k] for j in J) == 1,
                name=f"RankAssignment_{i}_{k}"
            )

    # 2. Can't assign more ranks than ambulances at a base
    for i in I:
        for j in J:
            model.addConstr(
                quicksum(z[i, j, k] for k in K) <= x[j],
                name=f"BaseCapacity_{i}_{j}"
            )

    # 3. Total number of ambulances constraint
    model.addConstr(
        quicksum(x[j] for j in J) <= p,
        name="TotalAmbulances"
    )

    return model