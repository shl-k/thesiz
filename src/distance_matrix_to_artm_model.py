from gurobipy import Model, GRB, quicksum
import numpy as np


def distance_matrix_to_artm_model(distance_matrix, demand_vec, p):
    """
    Inputs: 
        distance matrix (np array, shape: [demand_points, num_bases]) (units: meters)
        demand vector (np vec: shape: [demand_points])

    Outputs:
        Gurobi model for ARTM
        """
    # Parameters
    num_demand_points = distance_matrix.shape[0]  # Number of locations where calls are generated
    num_bases = distance_matrix.shape[1]           # Number of potential ambulance base locations

    # Sets
    I = list(range(num_demand_points))  # Demand points
    J = list(range(num_bases))          # Potential base locations

    # Parameters required for optimization
    t = {(i, j): distance_matrix[i, j] for j in J for i in I}  # Travel time between base j and demand point i
    d = {i: demand_vec[i] for i in I}                          # Demand at each location
                                  
    # Initialize model
    model = Model("Average Response Time Model")

    # Decision Variables
    x = model.addVars(J, vtype=GRB.BINARY, name="x")  # 1 if a base is opened at location j
    z = model.addVars(I, J, vtype=GRB.BINARY, name="z")  # 1 if demand point i is served by base j

    # Objective: Minimize average response time weighted by demand
    model.setObjective(
        quicksum(d[i] * t[i, j] * z[i, j] for j in J for i in I),
        GRB.MINIMIZE
    )

    # Constraints
    # 1. Ensure each demand point is served by exactly one base
    for i in I:
        model.addConstr(quicksum(z[i, j] for j in J) == 1, name=f"DemandCovered_{i}")

    # 2. Ensure a base must be opened to serve a demand point
    for j in J:
        for i in I:
            model.addConstr(z[i, j] <= x[j], name=f"BaseOpen_{j}_{i}")

    # 3. Limit the total number of ambulances deployed
    model.addConstr(quicksum(x[j] for j in J) <= p, name="AmbulanceLimit")

    return model
    