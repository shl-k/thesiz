# Utilities

This directory contains utility functions and tools used throughout the ambulance simulator.

## Files

- **osm_graph.py**: Functions for working with OpenStreetMap graphs, including downloading and processing map data.
- **osm_distance.py**: Utilities for calculating distances and travel times on OSM graphs.
- **distance_matrix.py**: Tools for creating and manipulating distance matrices.

## Usage

These utilities provide infrastructure for the simulator's geographical operations:

1. **Map Data**: Downloading and processing map data from OpenStreetMap.
2. **Distance Calculations**: Computing distances and travel times between nodes.
3. **Matrix Generation**: Creating the necessary matrices for simulator operations.
4. **Geographic Utilities**: Tools for working with geographic coordinates and road networks.

Most of these utilities are used during the setup phase before running simulations, but some may be called during simulation execution. 