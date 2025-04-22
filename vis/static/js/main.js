// Initialize map
mapboxgl.accessToken = 'YOUR_MAPBOX_ACCESS_TOKEN'; // You'll need to replace this with your token
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/streets-v12',
    center: [-122.4194, 37.7749], // San Francisco coordinates - adjust as needed
    zoom: 12
});

// Initialize socket connection
const socket = io();

// Store markers
const ambulanceMarkers = {};
const hospitalMarkers = {};
const callMarkers = {};

// Initialize markers when map loads
map.on('load', () => {
    // Add hospital marker
    const hospitalMarker = new mapboxgl.Marker({
        color: '#FF0000'
    })
    .setLngLat([-122.4194, 37.7749]) // Replace with actual hospital coordinates
    .addTo(map);
    hospitalMarkers['main'] = hospitalMarker;
});

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('simulation_state', (state) => {
    updateVisualization(state);
});

// Button event handlers
document.getElementById('startBtn').addEventListener('click', () => {
    socket.emit('start_simulation');
});

document.getElementById('pauseBtn').addEventListener('click', () => {
    socket.emit('pause_simulation');
});

document.getElementById('stepBtn').addEventListener('click', () => {
    socket.emit('step_simulation');
});

function updateVisualization(state) {
    // Update ambulance markers
    state.ambulances.forEach(amb => {
        if (!ambulanceMarkers[amb.id]) {
            // Create new marker
            ambulanceMarkers[amb.id] = new mapboxgl.Marker({
                color: getAmbulanceColor(amb.status)
            });
        }
        
        // Update marker position and color
        ambulanceMarkers[amb.id]
            .setLngLat([amb.location.lng, amb.location.lat])
            .addTo(map);
    });

    // Update statistics
    updateStatistics(state.statistics);
}

function getAmbulanceColor(status) {
    const colors = {
        0: '#00FF00', // IDLE - Green
        1: '#FFA500', // DISPATCHED - Orange
        2: '#FF0000', // ON_SCENE - Red
        3: '#0000FF', // TRANSPORT - Blue
        4: '#800080', // HOSPITAL - Purple
        5: '#FFFF00'  // RELOCATING - Yellow
    };
    return colors[status] || '#808080';
}

function updateStatistics(stats) {
    const statsContent = document.getElementById('statsContent');
    statsContent.innerHTML = `
        <p>Active Calls: ${stats.active_calls || 0}</p>
        <p>Available Ambulances: ${stats.available_ambulances || 0}</p>
        <p>Average Response Time: ${stats.avg_response_time || 0}s</p>
    `;
} 