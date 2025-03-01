import numpy as np
from gurobipy import Model, GRB, quicksum
from src.distance_matrix_to_artm_model import distance_matrix_to_model


def test_distance_matrix_to_model():
    D = np.array([[1, 2]])
    demand = np.array([1])

    model = distance_matrix_to_model(D, demand, 1)
    model.optimize()
    x = np.array([v.X for v in model.getVars()])
    assert np.allclose(x[0:2], np.array([1, 0]))


test_distance_matrix_to_model()