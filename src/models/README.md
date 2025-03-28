# Coverage Models

This directory contains various coverage models used in the ambulance relocation policies.

## Files

- **artm_model.py**: Implementation of the Adjusted Response Time Model (ARTM) for ambulance coverage.
- **dsm_model.py**: Implementation of the Double Standard Model (DSM) for ambulance coverage.
- **ertm_model.py**: Implementation of the Expected Response Time Model (ERTM) for ambulance coverage.
- **ertm_demand_model.py**: Extended ERTM model that incorporates demand patterns.

## Usage

These models are used by the simulator's relocation policies to determine optimal ambulance placement. Each model uses different approaches to maximize coverage:

1. **ARTM**: Adjusts coverage based on travel times and distance.
2. **DSM**: Uses a double-standard approach with two different coverage radii.
3. **ERTM**: Focuses on expected response times rather than binary coverage.
4. **ERTM Demand**: Incorporates historical demand patterns into the ERTM model.

When implementing new coverage models, follow the interface pattern established in these files for compatibility with the simulator. 