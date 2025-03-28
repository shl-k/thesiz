# Ambulance Simulator

This directory contains the core simulation engine for ambulance dispatch and relocation.

## Key Components

- **simulator.py**: The main simulator engine, containing the `AmbulanceSimulator` class that manages the simulation environment, ambulances, calls, and events.
- **ambulance.py**: Defines the `Ambulance` class and ambulance states for tracking ambulance movement and status.
- **policies.py**: Contains various dispatch and relocation policies that determine ambulance actions.
- **run_simulation.py**: Command-line interface for running simulations with different parameters.

## Understanding the Simulator

### Event-Driven Simulation

The simulator is event-driven, where events are processed in chronological order. Key events include:
- New emergency calls
- Ambulance dispatch
- Ambulance arrival at scene
- Patient transport to hospital
- Ambulance return to base

Each event is handled by updating the simulation state and potentially scheduling follow-up events.

### Ambulance States

Ambulances can be in several states:
- IDLE: Available at base
- DISPATCHED: En route to an emergency
- ON_SCENE: At the emergency scene
- TRANSPORT: Transporting patient to hospital
- HOSPITAL: At hospital for patient transfer
- RETURNING: Returning to base

### Policies

Two key policy types control ambulance behavior:
1. **Dispatch Policies**: Determine which ambulance to send to each emergency
2. **Relocation Policies**: Decide where idle ambulances should be positioned

## Running Simulations

Use `run_simulation.py` to execute simulations. Example:

```bash
python run_simulation.py --distance-matrix ../princeton_data/distance_matrix.npy \
                        --travel-time-matrix ../princeton_data/travel_time_matrix.npy \
                        --call-data ../princeton_data/synthetic_calls.csv \
                        --num-ambulances 5 \
                        --policy nearest-static
```

For policy comparisons, add the `--compare-policies` flag to test different policy combinations.

## Overview

This simulator models ambulance operations in a geographic area (like Princeton, NJ), allowing for the evaluation of different dispatch and relocation policies. It simulates emergency calls, ambulance movements, patient service, hospital transports, and strategic relocations - all on a realistic road network.

## Components

- **AmbulanceSimulator**: The core simulation engine
- **Ambulance**: Represents individual ambulance units with their states and operations
- **Policies**: Decision-making logic for dispatch and relocation
- **Coverage Models**: Mathematical optimization models for ambulance placement

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: numpy, pandas, networkx, matplotlib, osmnx, gurobipy (for optimization models)

```bash
pip install numpy pandas networkx matplotlib osmnx gurobipy
```

### Basic Usage

1. **Generate distance and travel time matrices**:

```python
from src.utils.geo_utils import create_distance_matrix, create_travel_time_matrix

# Generate matrices for Princeton, NJ
G, distance_matrix = create_distance_matrix(
    location="Princeton, NJ", 
    network_type="drive",
    save_file=True
)
travel_time_matrix = create_travel_time_matrix(distance_matrix)
```

2. **Run a simulation**:

```python
from src.simulator import AmbulanceSimulator

# Create and run simulator
simulator = AmbulanceSimulator(
    graph=graph,
    call_data_path="path/to/calls.csv",
    distance_matrix=distance_matrix,
    travel_time_matrix=travel_time_matrix,
    num_ambulances=5,
    base_locations=[100, 200, 300, 400, 500],  # Node indices
    hospital_node=792,  # Hospital node index
    dispatch_policy="nearest",
    relocation_policy="static",
    verbose=True
)

# Run simulation
results = simulator.run_simulation()

# Print results
print(f"Total calls: {results['total_calls']}")
print(f"Average response time: {results['average_response_time']:.2f} minutes")
```

3. **Compare different policies**:

```python
from src.simulator.run_simulation import run_and_compare_policies, plot_response_time_comparison

# Define policies to compare
policies = [
    {
        'name': 'Nearest-Static',
        'dispatch': 'nearest',
        'relocation': 'static',
        'coverage_model': None
    },
    {
        'name': 'Nearest-ARTM',
        'dispatch': 'nearest',
        'relocation': 'coverage',
        'coverage_model': 'artm'
    }
]

# Run comparison
results = run_and_compare_policies(
    graph=graph,
    call_data_path="path/to/calls.csv",
    distance_matrix=distance_matrix,
    travel_time_matrix=travel_time_matrix,
    num_ambulances=5,
    base_locations=[100, 200, 300, 400, 500],
    hospital_node=792,
    policies=policies
)

# Plot comparison
plot_response_time_comparison(results)
```

### Command Line Interface

The simulator can also be run from the command line:

```bash
python run_simulation.py --distance-matrix path/to/distance_matrix.npy \
                        --travel-time-matrix path/to/travel_time_matrix.npy \
                        --call-data path/to/calls.csv \
                        --num-ambulances 5 \
                        --policy nearest-static
```

## Policies

The simulator supports several dispatch and relocation policies:

### Dispatch Policies

- **Nearest**: Dispatches the closest available ambulance to a call
- **ADP**: Uses approximate dynamic programming for dispatch decisions

### Relocation Policies

- **Static**: Returns ambulances to their assigned base when idle
- **Coverage**: Relocates idle ambulances to maximize coverage
- **ADP**: Uses approximate dynamic programming for relocation decisions

## Coverage Models

For the coverage-based relocation policy, several models are available:

- **ARTM**: Average Response Time Model - minimizes average response time
- **DSM**: Double Standard Model - maximizes demand covered within two time thresholds
- **ERTM**: Expected Response Time Model - accounts for ambulance busy probabilities

## Debugging

If you encounter issues, please refer to the `DEBUGGING_GUIDE.md` file for common problems and solutions.

## License

This project is licensed under the MIT License. 
