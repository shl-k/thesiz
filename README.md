# Ambulance Dispatch Optimization

This project implements an ambulance dispatch and relocation system using reinforcement learning. The system optimizes ambulance positioning and dispatch decisions to minimize response times and maximize coverage.

## Project Structure

```
.
├── src/
│   ├── data/           # Data preparation and processing
│   ├── models/         # Coverage models (DSM, ARTM, ERTM)
│   ├── rl/            # Reinforcement learning components
│   ├── simulator/      # Ambulance simulation environment
│   └── utils/         # Utility functions
├── test/              # Test files
├── run.py            # Main simulation runner
└── train_rl.py       # RL training script
```

## Features

- Ambulance dispatch simulation
- Multiple coverage models (DSM, ARTM, ERTM)
- Reinforcement learning for dispatch optimization
- Real-world road network integration
- Performance metrics and visualization

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the simulation:
```bash
python run.py
```

To train the RL agent:
```bash
python train_rl.py --mode train --algorithm q-learning --episodes 100
``` 