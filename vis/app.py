from flask import Flask, render_template
from flask_socketio import SocketIO
import sys
from pathlib import Path
import time
import threading

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simulator.simulator import AmbulanceSimulator
import networkx as nx
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Initialize simulator (we'll need to load actual data later)
graph = nx.Graph()  
call_data = pd.DataFrame() 
simulator = AmbulanceSimulator(
    graph=graph,
    call_data=call_data,
    num_ambulances=3,
    base_location=241,
    hospital_node=1293
)

# Simulation control
simulation_running = False
simulation_thread = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global simulation_running
    simulation_running = False

@socketio.on('start_simulation')
def handle_start_simulation():
    global simulation_running, simulation_thread
    
    if not simulation_running:
        simulation_running = True
        # Initialize the simulation
        simulator.initialize()
        
        # Start simulation in a separate thread
        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.daemon = True
        simulation_thread.start()
        
        # Send initial state
        state = get_simulation_state()
        socketio.emit('simulation_state', state)

@socketio.on('pause_simulation')
def handle_pause_simulation():
    global simulation_running
    simulation_running = False
    socketio.emit('simulation_paused')

@socketio.on('step_simulation')
def handle_step_simulation():
    # Run a single step of the simulation
    time, event_type, payload = simulator.step()
    
    # Send updated state
    state = get_simulation_state()
    socketio.emit('simulation_state', state)

def run_simulation():
    """Run the simulation in a loop and emit updates."""
    global simulation_running
    
    while simulation_running:
        # Run a single step
        time, event_type, payload = simulator.step()
        
        # Send updated state
        state = get_simulation_state()
        socketio.emit('simulation_state', state)
        
        # Small delay to control simulation speed
        time.sleep(0.1)

def get_simulation_state():
    """Get the current state of the simulation for visualization."""
    # Get ambulance locations and convert to lat/lng
    ambulances = []
    for amb in simulator.ambulances:
        # For now, use dummy coordinates - we'll need to convert from graph nodes
        # to lat/lng coordinates in a real implementation
        lat, lng = get_coordinates_from_node(amb.location)
        
        ambulances.append({
            'id': amb.id,
            'location': {'lat': lat, 'lng': lng},
            'status': amb.status.value,
            'path': [get_coordinates_from_node(node) for node in amb.path] if amb.path else []
        })
    
    # Get active calls
    active_calls = []
    # This would be implemented based on your simulator's data structure
    
    # Get statistics
    stats = {
        'active_calls': len(active_calls),
        'available_ambulances': sum(1 for amb in simulator.ambulances if amb.status.value == 0),
        'avg_response_time': 0  # This would be calculated from your simulator's data
    }
    
    return {
        'ambulances': ambulances,
        'active_calls': active_calls,
        'statistics': stats
    }

def get_coordinates_from_node(node_id):
    """Convert a graph node ID to lat/lng coordinates."""
    # This is a placeholder - in a real implementation, you would
    # look up the actual coordinates from your graph data
    # For now, return dummy coordinates around San Francisco
    import random
    return (
        37.7749 + random.uniform(-0.01, 0.01),
        -122.4194 + random.uniform(-0.01, 0.01)
    )

if __name__ == '__main__':
    socketio.run(app, debug=True) 