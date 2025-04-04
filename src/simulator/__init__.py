"""
Ambulance Simulator Package

This package provides a simulation environment for modeling emergency medical services
operations, including ambulance dispatch, relocation, and response to emergency calls.

The simulator can be used to evaluate different dispatch and relocation policies,
optimize ambulance placement, and analyze system performance metrics such as
response times.

Main Components:
- AmbulanceSimulator: Core simulator engine
- Ambulance: Representation of ambulance units
- AmbulanceStatus: States an ambulance can be in
- Policy classes: Decision-making for dispatch and relocation
- Coverage models: Mathematical models for ambulance placement
"""

from src.simulator.ambulance import Ambulance, AmbulanceStatus
from src.simulator.simulator import AmbulanceSimulator
from src.simulator.policies import (
    NearestDispatchPolicy, 
    StaticRelocationPolicy,
)

__version__ = '1.0.0'
__author__ = 'Shlok Patel'

# This allows "from src.simulator import *" to import the main classes
__all__ = [
    'AmbulanceSimulator',
    'Ambulance',
    'AmbulanceStatus',
    'NearestDispatchPolicy',
    'StaticRelocationPolicy'
] 