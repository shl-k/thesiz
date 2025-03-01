# Ambulance Location Optimization Models

This repository contains the implementation and analysis of two ambulance location optimization models:
1. Average Response Time Model (ARTM)
2. Enhanced Response Time Model (ERTM)

## Project Structure

```
.
├── src/                    # Source code
│   ├── distance_matrix_to_artm_model.py    # ARTM implementation
│   ├── distance_matrix_to_ertm_model.py    # ERTM implementation
│   ├── osmnx_to_distance_matrix.py         # Distance matrix generation
│   └── osmnx_to_graph.py                   # Graph utilities
├── test/                   # Test files
│   ├── dist_to_model_test.py
│   └── osmnx_to_graph_test.py
└── workspace.ipynb         # Jupyter notebook with examples
```

## Features

- Implementation of ARTM and ERTM optimization models
- Integration with OpenStreetMap data through OSMnx
- Comprehensive testing suite
- Real-world case study with Princeton, NJ data

## Requirements

- Python 3.12+
- Gurobi Optimizer
- OSMnx
- NetworkX
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have a valid Gurobi license

## Usage

1. Basic usage with the ARTM model:
```python
from src.distance_matrix_to_artm_model import distance_matrix_to_artm_model
from src.osmnx_to_distance_matrix import osmnx_to_distance_matrix

# Get distance matrix for a location
D = osmnx_to_distance_matrix(location="Princeton, NJ")

# Create and solve model
model = distance_matrix_to_artm_model(D, demand_vec, p=2)
model.optimize()
```

2. Using the ERTM model with backup coverage:
```python
from src.distance_matrix_to_ertm_model import distance_matrix_to_ertm_model

# Create and solve model with backup coverage
model = distance_matrix_to_ertm_model(D, demand_vec, p=2, q=0.3)
model.optimize()
```

## Testing

Run the test suite:
```bash
python -m pytest test/
```

## Results

The models have been tested on Princeton, NJ's road network with the following results:
- Network size: 1,583 nodes
- Mean distance: 27.52 meters
- Optimal ambulance locations identified for both ARTM and ERTM
- Successful incorporation of backup coverage in ERTM

## License

[Your chosen license]

## Contact

[Your contact information] 