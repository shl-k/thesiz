from gurobipy import Model, GRB, quicksum
import numpy as np

def distance_matrix_with_demand_to_ertm_model(distance_matrix, intensity_vec, demand_vec, gamma_vec, p, q):
    """
    Inputs: 
        distance_matrix: [demand_points, num_bases] (meters)
        intensity_vec: [demand_points] (weights for objective)
        demand_vec: [demand_points] (actual number of calls to serve)
        gamma_vec: [demand_points] (priority/penalty weights for unmet demand)
          (called demand_vec in artm and ertm models)
        p: max number of ambulances
        q: probability parameter for backup coverage
    """
    # Parameters
    num_demand_points = distance_matrix.shape[0]
    num_bases = distance_matrix.shape[1]

    # Sets
    I = list(range(num_demand_points))
    J = list(range(num_bases))
    K = list(range(1, p+1))

    # Parameters for optimization
    t = {(i, j): distance_matrix[i, j] for j in J for i in I}

    # Initialize model
    model = Model("Expected Response Time Model with Demand")

    # Decision Variables
    x = model.addVars(J, vtype=GRB.INTEGER, name="x")  # Number of ambulances at base j
    z = model.addVars(I, J, K, vtype=GRB.BINARY, name="z")  # 1 if base j serves demand i as kth nearest
    unmet = model.addVars(I, name="unmet")  # Amount of unmet demand at point i

    # Objective: Response time + penalty for unmet demand
    obj1 = quicksum(
        intensity_vec[i] * t[i, j] * (1-q) * (q**(k-1)) * z[i, j, k]
        for i in I for j in J for k in range(1, p)
    )
    obj2 = quicksum(
        intensity_vec[i] * t[i, j] * (q**(p-1)) * z[i, j, p]
        for i in I for j in J
    )
    obj3 = quicksum(gamma_vec[i] * unmet[i] for i in I)
    
    model.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)

    # Constraints
    # 1. Each demand point must have exactly one kth nearest ambulance for each rank
    for i in I:
        for k in K:
            model.addConstr(
                quicksum(z[i, j, k] for j in J) == 1,
                name=f"RankAssignment_{i}_{k}"
            )

    # 2. Demand satisfaction with possible unmet demand
    for i in I:
        model.addConstr(
            quicksum(z[i, j, k] for j in J for k in K) + unmet[i] >= demand_vec[i],
            name=f"DemandSatisfaction_{i}"
        )

    # 3. Base capacity constraints (one person per ambulance)
    for j in J:
        model.addConstr(
            quicksum(z[i, j, k] for i in I for k in K) <= x[j],
            name=f"BaseCapacity_{j}"
        )

    # 4. Total number of ambulances constraint
    model.addConstr(
        quicksum(x[j] for j in J) <= p,
        name="TotalAmbulances"
    )

    return model