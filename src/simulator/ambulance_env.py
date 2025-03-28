import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import heapq
from src.simulator.simulator import AmbulanceSimulator
from src.simulator.ambulance import Ambulance, AmbulanceStatus
from src.rl.rl_agent import QAgent, SarsaAgent
import networkx as nx

# Define EventTuple locally since it's not exported from simulator
class EventTuple:
    def __init__(self, time, event_type, event_data, event_id):
        self.time = time
        self.event_type = event_type
        self.event_data = event_data
        self.event_id = event_id
    
    def __lt__(self, other):
        return self.time < other.time

class AmbulanceEnv:
    """
    A Gym-like environment wrapper for the ambulance simulator.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        call_data_path: str,
        distance_matrix: np.ndarray,
        num_ambulances: int,
        base_locations: List[int],
        hospital_node: int,
        index_to_node: Dict[int, Any] = None,
        node_to_index: Dict[Any, int] = None,
        scenario_days: int = 7,
        dispatch_policy: str = 'nearest',
        relocation_policy: str = 'static',
        coverage_model: str = None,
        coverage_params: Dict = None,
        time_resolution: str = 'second',
        service_time_mean: int = 900,  # 15 minutes in seconds
        hospital_transfer_time_mean: int = 2700,  # 45 minutes in seconds
        avg_speed: float = 8.33,  # m/s (30 km/h)
        verbose: bool = False
    ):
        """
        Initialize the ambulance environment.
        
        Args:
            graph: NetworkX graph of the road network
            call_data_path: Path to call data CSV file
            distance_matrix: Matrix of distances between nodes in meters
            num_ambulances: Number of ambulances to simulate
            base_locations: List of base location nodes
            hospital_node: Hospital node ID
            index_to_node: Mapping from matrix indices to node IDs
            node_to_index: Mapping from node IDs to matrix indices
            scenario_days: Number of days to simulate
            dispatch_policy: Policy for dispatching ambulances
            relocation_policy: Policy for relocating ambulances
            coverage_model: Model for calculating coverage
            coverage_params: Parameters for the coverage model
            time_resolution: Resolution for time tracking ('second')
            service_time_mean: Mean service time in seconds
            hospital_transfer_time_mean: Mean hospital transfer time in seconds
            avg_speed: Average speed in m/s (default 8.33 m/s = 30 km/h)
            verbose: Whether to print detailed logs
        """
        self.graph = graph
        self.call_data_path = call_data_path
        self.distance_matrix = distance_matrix
        self.num_ambulances = num_ambulances
        self.base_locations = base_locations
        self.hospital_node = hospital_node
        self.scenario_days = scenario_days
        self.time_resolution = time_resolution
        self.coverage_model = coverage_model
        self.coverage_params = coverage_params or {}
        self.verbose = verbose
        
        # State space parameters
        self.num_nodes = len(distance_matrix)
        self.num_statuses = len(AmbulanceStatus)
        
        # Add a pending call counter dictionary
        self.pending_call_counts = {}
        
        # Set service time and hospital transfer time means
        self.service_time_mean = service_time_mean
        self.hospital_transfer_time_mean = hospital_transfer_time_mean
        
        # Create simulator
        self._create_simulator(graph, call_data_path, distance_matrix, num_ambulances, base_locations, hospital_node, index_to_node, node_to_index, dispatch_policy, relocation_policy, coverage_model, coverage_params, time_resolution, service_time_mean, hospital_transfer_time_mean, avg_speed)
        
        # Track current decision type
        self.decision_type = None  # Can be 'dispatch', 'relocate', or None
        self.current_call = None
        self.idle_ambulance_id = None
        
    def _create_simulator(self, graph, call_data_path, distance_matrix, num_ambulances, base_locations, hospital_node, index_to_node, node_to_index, dispatch_policy, relocation_policy, coverage_model, coverage_params, time_resolution, service_time_mean, hospital_transfer_time_mean, avg_speed):
        """Create a new simulator instance."""
        self.simulator = AmbulanceSimulator(
            graph=graph,
            call_data_path=call_data_path,
            distance_matrix=distance_matrix,
            num_ambulances=num_ambulances,
            base_locations=base_locations,
            hospital_node=hospital_node,
            index_to_node=index_to_node,
            node_to_index=node_to_index,
            scenario_days=self.scenario_days,
            dispatch_policy=dispatch_policy,
            relocation_policy=relocation_policy,
            coverage_model=coverage_model,
            coverage_params=coverage_params,
            time_resolution=time_resolution,
            service_time_mean=service_time_mean,
            hospital_transfer_time_mean=hospital_transfer_time_mean,
            avg_speed=avg_speed,
            verbose=self.verbose
        )
        
        # Initialize additional attributes
        self.simulator.processed_event_count = 0
        self.simulator.calls = []
        self.simulator.dispatch_count = 0
        self.simulator.dispatch_times = []
        self.simulator.response_times = []
        
        # Set service time and hospital transfer time means in simulator
        self.simulator.service_time_mean = self.service_time_mean
        self.simulator.hospital_transfer_time_mean = self.hospital_transfer_time_mean
        
    def reset(self):
        """
        Reset the environment to the start of Day 1.
        
        Returns:
            Initial observation (state)
        """
        # Create a new simulator
        self._create_simulator(self.graph, self.call_data_path, self.distance_matrix, self.num_ambulances, self.base_locations, self.hospital_node, None, None, 'nearest', 'static', None, None, 'second', self.service_time_mean, self.hospital_transfer_time_mean, 8.33)
        
        # Initialize simulator time to start of day
        self.simulator.current_time = 0
        
        # Prepare the call queue
        self.simulator.prepare_call_queue()
        
        # Reset pending call counts
        self.pending_call_counts = {}
        
        # Debug: print number of events in queue
        if self.verbose:
            print(f"Reset: {len(self.simulator.events)} events in queue")
            if self.simulator.events:
                first_event = self.simulator.events[0]
                if hasattr(first_event, 'unpack'):
                    time, type_, data, _ = first_event.unpack()
                    print(f"First event: time={time}, type={type_}")
                    if type_ == 'new_call':
                        print(f"First call node: {data['node']}")
        
        # Initialize decision type to None - we'll determine this in step()
        self.decision_type = None
        self.current_call = None
        self.idle_ambulance_id = None
        
        # Fast forward to first decision point
        self._advance_to_next_decision()
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: Action to take
                If decision_type is 'dispatch', action is the ambulance ID to dispatch
                If decision_type is 'relocate', action is the base location to relocate to
                
        Returns:
            (next_state, reward, done, info)
        """
        reward = 0
        info = {}
        
        # Process action based on decision type
        if self.decision_type == 'dispatch':
            # Action is ambulance ID to dispatch
            if action is not None:
                # Dispatch ambulance
                call = self.current_call
                ambulance = next((amb for amb in self.simulator.ambulances if amb.id == action), None)
                
                if ambulance is None:
                    print(f"Warning: Ambulance {action} not found for dispatch")
                    reward = -100  # Penalty for invalid action
                else:
                    # Get path from ambulance to call
                    path = self.simulator.get_path(ambulance.location, call['node'])
                    travel_time = self.simulator.travel_time_matrix[ambulance.location][call['node']]
                    
                    # Update ambulance status
                    ambulance.dispatch_to_call(call, self.simulator.current_time, path, travel_time)
                    
                    # Schedule arrival at scene
                    event_id = self.simulator.event_id_counter
                    self.simulator.event_id_counter += 1
                    heapq.heappush(self.simulator.events, EventTuple(
                        self.simulator.current_time + travel_time,
                        'ambulance_arrival',
                        {'ambulance_id': action, 'call': call},
                        event_id
                    ))
                    
                    # Calculate response time and reward
                    response_time = travel_time
                    reward = -response_time  # Negative response time
                    
                    # Track metrics
                    self.simulator.dispatch_count += 1
                    self.simulator.dispatch_times.append(travel_time)
                    self.simulator.response_times.append(response_time)
                    
                    if self.verbose:
                        print(f"Time {self.simulator.current_time}: Dispatching ambulance {action} to call at {call['node']}")
                        print(f"Expected arrival at {self.simulator.current_time + travel_time:.1f} (travel time: {travel_time:.1f} seconds)")
            else:
                # No ambulance available, retry later
                event_id = self.simulator.event_id_counter
                self.simulator.event_id_counter += 1
                heapq.heappush(self.simulator.events, EventTuple(
                    self.simulator.current_time + 1,
                    'pending_call',
                    self.current_call,
                    event_id
                ))
                
                # Penalty for not dispatching
                reward = -10
        
        elif self.decision_type == 'relocate':
            # Action is the base location to relocate to
            if action is not None and self.idle_ambulance_id is not None:
                amb_id = self.idle_ambulance_id
                amb = next((a for a in self.simulator.ambulances if a.id == amb_id), None)
                
                if amb is None:
                    print(f"Warning: Ambulance {amb_id} not found for relocation")
                    reward = -100  # Penalty for invalid action
                elif action not in self.base_locations:
                    print(f"Warning: Invalid base location {action} for relocation")
                    reward = -100  # Penalty for invalid action
                else:
                    # Get path and travel time
                    path = self.simulator.get_path(amb.location, action)
                    travel_time = self.simulator.travel_time_matrix[amb.location][action]
                    
                    # Relocate ambulance
                    amb.relocate(action, path, travel_time, self.simulator.current_time)
                    
                    # Schedule arrival at base
                    event_id = self.simulator.event_id_counter
                    self.simulator.event_id_counter += 1
                    heapq.heappush(self.simulator.events, EventTuple(
                        self.simulator.current_time + travel_time,
                        'ambulance_base_arrival',
                        {'ambulance_id': amb_id},
                        event_id
                    ))
                    
                    # Small negative reward for travel time
                    reward = -travel_time * 0.1  # Scaled down compared to dispatch rewards
                    
                    if self.verbose:
                        print(f"Time {self.simulator.current_time}: Relocating ambulance {amb_id} to location {action}")
                        print(f"Expected arrival at {self.simulator.current_time + travel_time:.1f} (travel time: {travel_time:.1f} seconds)")
        
        # Reset decision type
        self.decision_type = None
        self.current_call = None
        self.idle_ambulance_id = None
        
        # Process events until next decision point
        done = self._advance_to_next_decision()
        
        # Get new state
        next_state = self._get_state()
        
        # Add info
        info['current_time'] = self.simulator.current_time
        info['event_counter'] = self.simulator.event_id_counter
        
        return next_state, reward, done, info
    
    def _advance_to_next_decision(self):
        """
        Advance the simulation until the next decision point.
        
        Returns:
            done: Whether the episode is done
        """
        # Debug: number of events at start
        if self.verbose:
            print(f"Advancing: {len(self.simulator.events)} events in queue")
            if self.simulator.events:
                first_event = self.simulator.events[0]
                if hasattr(first_event, 'unpack'):
                    time, type_, data, _ = first_event.unpack()
                    print(f"Next event: time={time}, type={type_}")
                    if 'node' in data:
                        print(f"Call node: {data['node']}")
        
        # Process events until we need a decision
        while self.simulator.events and self.decision_type is None:
            # Get next event
            event = heapq.heappop(self.simulator.events)
            
            # Unpack event - handle EventTuple or tuple
            if hasattr(event, 'unpack'):
                event_time, event_type, event_data, event_id = event.unpack()
            else:
                event_time, event_type, event_data, event_id = event
            
            # Debug: print event being processed
            if self.verbose:
                print(f"Processing event: time={event_time}, type={event_type}")
                if event_type in ['new_call', 'pending_call'] and 'node' in event_data:
                    print(f"Call node: {event_data['node']}")
            
            # Update simulation time
            self.simulator.current_time = event_time
            
            # Check if simulation is done - based on current day exceeding scenario days
            current_day = int(self.simulator.current_time / (24 * 3600)) + 1
            if current_day > self.simulator.scenario_days:
                if self.verbose:
                    print(f"Simulation done: current_day={current_day} > scenario_days={self.simulator.scenario_days}")
                return True
            
            # Process event based on type
            if event_type == 'new_call':
                # New call arrived, need to make a dispatch decision
                call = event_data
                if self.verbose:
                    print(f"New call arrived at node {call.get('node', 'unknown')}")
                
                # Add to calls list for tracking
                self.simulator.calls.append(call)
                
                # Initialize call ID for tracking pending calls
                call_id = call.get('id', str(event_id))
                self.pending_call_counts[call_id] = 0
                
                # Check if there are available ambulances
                available_ambulances = [amb for amb in self.simulator.ambulances 
                                       if amb.is_available(self.simulator.current_time)]
                
                if available_ambulances:
                    # Need to make a dispatch decision
                    self.decision_type = 'dispatch'
                    self.current_call = call
                    if self.verbose:
                        print(f"Dispatch decision needed for call at node {call.get('node', 'unknown')}")
                        print(f"Available ambulances: {[amb.id for amb in available_ambulances]}")
                    return False
                else:
                    # No ambulances available, queue call for later
                    if self.verbose:
                        print(f"No ambulances available for call at node {call.get('node', 'unknown')}, queueing as pending")
                    
                    event_id = self.simulator.event_id_counter
                    self.simulator.event_id_counter += 1
                    heapq.heappush(self.simulator.events, EventTuple(
                        self.simulator.current_time + 1,
                        'pending_call',
                        call,
                        event_id
                    ))
            
            elif event_type == 'pending_call':
                # Retry pending call
                call = event_data
                call_id = call.get('id', str(event_id))
                
                # Get current count or initialize to 0
                current_count = self.pending_call_counts.get(call_id, 0)
                
                # Increment count
                self.pending_call_counts[call_id] = current_count + 1
                
                if self.verbose:
                    print(f"Processing pending call at node {call.get('node', 'unknown')} (attempt {current_count + 1})")
                
                # Check if we've tried too many times (limit to 2 requeues)
                if current_count >= 2:
                    if self.verbose:
                        print(f"Call at node {call.get('node', 'unknown')} has been pending too long, dropping")
                    # Drop the call - don't requeue
                    continue
                
                # Check if there are available ambulances
                available_ambulances = [amb for amb in self.simulator.ambulances 
                                       if amb.is_available(self.simulator.current_time)]
                
                if available_ambulances:
                    # Need to make a dispatch decision
                    self.decision_type = 'dispatch'
                    self.current_call = call
                    if self.verbose:
                        print(f"Dispatch decision needed for pending call at node {call.get('node', 'unknown')}")
                        print(f"Available ambulances: {[amb.id for amb in available_ambulances]}")
                    return False
                else:
                    # No ambulances available, queue call for later if we haven't reached the limit
                    if self.verbose:
                        print(f"No ambulances available for pending call at node {call.get('node', 'unknown')}, re-queueing")
                    
                    event_id = self.simulator.event_id_counter
                    self.simulator.event_id_counter += 1
                    heapq.heappush(self.simulator.events, EventTuple(
                        self.simulator.current_time + 1,
                        'pending_call',
                        call,
                        event_id
                    ))
            
            # Other event types remain unchanged
            elif event_type == 'ambulance_arrival':
                # Ambulance arrives at call scene
                ambulance_id = event_data['ambulance_id']
                call = event_data['call']
                
                # Find ambulance
                ambulance = next((amb for amb in self.simulator.ambulances if amb.id == ambulance_id), None)
                
                if ambulance is None:
                    print(f"Warning: Ambulance {ambulance_id} not found for arrival")
                    continue
                
                # Calculate service time
                service_time = np.random.poisson(self.simulator.service_time_mean)
                
                # Update ambulance status
                ambulance.arrive_at_scene(self.simulator.current_time, service_time)
                
                # Schedule transport event
                event_id = self.simulator.event_id_counter
                self.simulator.event_id_counter += 1
                heapq.heappush(self.simulator.events, EventTuple(
                    self.simulator.current_time + service_time,
                    'ambulance_transport',
                    {'ambulance_id': ambulance_id, 'call': call},
                    event_id
                ))
            
            elif event_type == 'ambulance_transport':
                # Ambulance begins transport to hospital
                ambulance_id = event_data['ambulance_id']
                call = event_data['call']
                
                # Find ambulance
                ambulance = next((amb for amb in self.simulator.ambulances if amb.id == ambulance_id), None)
                
                if ambulance is None:
                    print(f"Warning: Ambulance {ambulance_id} not found for transport")
                    continue
                
                # Get path to hospital
                path = self.simulator.get_path(call['node'], self.simulator.hospital_node)
                transport_time = self.simulator.travel_time_matrix[call['node']][self.simulator.hospital_node]
                
                # Update ambulance status
                ambulance.begin_transport(self.simulator.hospital_node, path, transport_time, self.simulator.current_time)
                
                # Schedule hospital arrival event
                event_id = self.simulator.event_id_counter
                self.simulator.event_id_counter += 1
                heapq.heappush(self.simulator.events, EventTuple(
                    self.simulator.current_time + transport_time,
                    'ambulance_hospital_arrival',
                    {'ambulance_id': ambulance_id},
                    event_id
                ))
            
            elif event_type == 'ambulance_hospital_arrival':
                # Ambulance arrives at hospital
                ambulance_id = event_data['ambulance_id']
                
                # Find ambulance
                ambulance = next((amb for amb in self.simulator.ambulances if amb.id == ambulance_id), None)
                
                if ambulance is None:
                    print(f"Warning: Ambulance {ambulance_id} not found for hospital arrival")
                    continue
                
                # Calculate hospital transfer time
                transfer_time = np.random.poisson(self.simulator.hospital_transfer_time_mean)
                
                # Update ambulance status
                ambulance.arrive_at_hospital(self.simulator.current_time, transfer_time)
                
                # Schedule return to base event
                event_id = self.simulator.event_id_counter
                self.simulator.event_id_counter += 1
                heapq.heappush(self.simulator.events, EventTuple(
                    self.simulator.current_time + transfer_time,
                    'ambulance_return',
                    {'ambulance_id': ambulance_id},
                    event_id
                ))
            
            elif event_type == 'ambulance_return':
                # Ambulance begins return to base - need to make a relocation decision
                ambulance_id = event_data['ambulance_id']
                
                # Find ambulance
                ambulance = next((amb for amb in self.simulator.ambulances if amb.id == ambulance_id), None)
                
                if ambulance is None:
                    print(f"Warning: Ambulance {ambulance_id} not found for return")
                    continue
                
                # Need to make a relocation decision
                self.decision_type = 'relocate'
                self.idle_ambulance_id = ambulance_id
                if self.verbose:
                    print(f"Relocation decision needed for ambulance {ambulance_id}")
                return False
            
            elif event_type == 'ambulance_base_arrival':
                # Ambulance arrives at base
                ambulance_id = event_data['ambulance_id']
                
                # Find ambulance
                ambulance = next((amb for amb in self.simulator.ambulances if amb.id == ambulance_id), None)
                
                if ambulance is None:
                    print(f"Warning: Ambulance {ambulance_id} not found for base arrival")
                    continue
                
                # Update ambulance status
                ambulance.arrive_at_base(self.simulator.current_time)
                if self.verbose:
                    print(f"Ambulance {ambulance_id} arrived at base {ambulance.base_node}")
            
            # Increment processed event count
            self.simulator.processed_event_count += 1
        
        # If we ran out of events, the episode is done
        if not self.simulator.events:
            if self.verbose:
                print("No more events in queue, simulation is done")
            return True
        
        return False
    
    def _get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            state: A compact representation of the environment state
        """
        # Encode ambulance information
        ambulance_states = []
        for amb in self.simulator.ambulances:
            # Encode location and status
            ambulance_states.append((amb.location, amb.status.value))
        
        # Encode current call (if any)
        call_info = None
        if self.current_call is not None:
            call_info = self.current_call['node']
        
        # Encode time of day (0-86400 seconds)
        time_of_day = int(self.simulator.current_time) % (24 * 3600)
        
        # Return state as a tuple
        return (tuple(ambulance_states), call_info, time_of_day, self.decision_type)
    
    def get_available_actions(self):
        """
        Get the available actions for the current decision type.
        
        Returns:
            actions: List of available actions
        """
        if self.decision_type == 'dispatch':
            # Add debug printing to understand the problem
            if self.verbose:
                print("Debug get_available_actions:")
                for amb in self.simulator.ambulances:
                    print(f"  Amb {amb.id} status: {amb.status}, status.value: {amb.status.value}")
                    print(f"  Comparison: {amb.status} == {AmbulanceStatus.IDLE} -> {amb.status == AmbulanceStatus.IDLE}")
                    print(f"  Value comparison: {amb.status.value} == {AmbulanceStatus.IDLE.value} -> {amb.status.value == AmbulanceStatus.IDLE.value}")
            
            # Compare by value instead of direct enum comparison
            return [amb.id for amb in self.simulator.ambulances if amb.status.value == AmbulanceStatus.IDLE.value]
        
        elif self.decision_type == 'relocate':
            # Available actions are all base locations
            return self.base_locations
        
        return []
    
    def render(self, mode='human'):
        """
        Render the environment (optional).
        
        Args:
            mode: Rendering mode
        """
        if self.verbose:
            print(f"Current time: {self.simulator.current_time}")
            print(f"Decision type: {self.decision_type}")
            print("Ambulances:")
            for amb in self.simulator.ambulances:
                print(f"  ID: {amb.id}, Location: {amb.location}, Status: {amb.status}")
            if self.current_call is not None:
                print(f"Current call: Node {self.current_call['node']}")

    def get_current_day(self) -> int:
        """Get the current day of the simulation."""
        current_day = int(self.simulator.current_time / (24 * 3600)) + 1
        return current_day

    def get_time_of_day(self) -> int:
        """Get the time of day in seconds (0-86400)."""
        time_of_day = int(self.simulator.current_time) % (24 * 3600)
        return time_of_day 