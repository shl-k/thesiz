# Ambulance Simulator Debugging Guide

This guide summarizes the issues we've identified and fixed in the ambulance simulator, along with best practices to maintain a stable simulation environment.

## Common Issues and Solutions

### 1. Type Consistency with Matrix Indices

**Problem**: Matrix indices were sometimes stored as numpy.float64 or other non-integer types, causing errors when accessing matrices.

**Solution**: 
- We added explicit integer type conversion in the Ambulance class methods for all location attributes
- Updated all matrix indexing operations to ensure integer values are used
- Added proper error handling for type conversion

### 2. Index Bounds Checking

**Problem**: Some node indices were out of bounds for the travel time and distance matrices.

**Solution**:
- Added bounds checking for all matrix accesses
- Implemented clamping of out-of-bounds indices to valid ranges
- Added fallback values for cases where valid indices cannot be determined

### 3. Path Finding Robustness

**Problem**: Path finding between nodes sometimes failed with NetworkX exceptions.

**Solution**:
- Enhanced the get_path method with comprehensive error handling
- Added fallback paths when shortest paths can't be found
- Included validation of node existence in the graph

### 4. Ambulance Base Validation

**Problem**: Base locations sometimes had invalid indices.

**Solution**:
- Updated initialize_ambulances to validate base node indices
- Added filter to exclude invalid base locations
- Implemented a fallback to create valid bases when none are provided

### 5. Call Handling

**Problem**: Call data sometimes lacked expected fields or had inconsistent formats.

**Solution**:
- Enhanced call data handling to check for multiple possible field names
- Added proper copying of call data to avoid modifying original data
- Implemented retry mechanism for pending calls

## Best Practices

1. **Type Safety**: Always ensure consistent types for matrix indices (always integers)
2. **Bounds Checking**: Validate indices are within matrix bounds before access
3. **Error Handling**: Add comprehensive try/except blocks with meaningful fallbacks
4. **Logging**: Include detailed logging to track ambulance movements and decisions
5. **Data Validation**: Validate input data (call data, base locations) before use
6. **Testing**: Use the provided test scripts to verify components independently:
   - `test_matrix.py`: Validates matrix access
   - `test_minimal_simulation.py`: Tests single-ambulance behavior
   - `test_multi_ambulance.py`: Tests dispatch policy with multiple ambulances
   - `test_policies.py`: Compares different dispatch and relocation policies

## Running the Simulator

The simulator can be run with:

```bash
python run_simulation.py --distance-matrix path/to/distance_matrix.npy \
                        --travel-time-matrix path/to/travel_time_matrix.npy \
                        --call-data path/to/calls.csv \
                        --num-ambulances 5 \
                        --policy nearest-static
```

## Adding New Policies

When adding new dispatch or relocation policies:

1. Extend the appropriate base class in `policies.py`
2. Register the policy in the `initialize_policies` method of `AmbulanceSimulator`
3. Add the policy configuration to the policies list in `run_simulation.py`

## Debugging Tips

1. Use `--verbose` flag to enable detailed logging
2. Check ambulance status using the AmbulanceStatus enum
3. Verify matrix shapes match the number of nodes in the graph
4. Inspect response times for reasonability (they shouldn't be zero or extremely large)
5. Consider using the step-by-step simulation for detailed debugging 