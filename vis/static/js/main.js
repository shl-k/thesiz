// Initialize map
console.log('Initializing map...');

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, creating map...');
    const map = L.map('map').setView([40.3573, -74.6593], 14); // Princeton coordinates

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Initialize socket connection
    console.log('Initializing socket connection...');
    const socket = io();

    // Store markers
    const ambulanceMarkers = {};
    const callMarkers = {};
    
    // Store previous positions and animation states
    const prevPositions = {};
    const animationStates = {};
    
    // Debug logging
    function logDebug(message, data) {
        console.log(`[DEBUG] ${message}`, data || '');
    }

    // Add Leaflet.arrowhead plugin for directional arrows
    L.Polyline.Arrow = L.Polyline.extend({
        options: {
            arrowheads: [{
                frequency: 'endpoint',
                size: '15px',
                offset: '25%',
                fill: true
            }]
        }
    });

    L.polyline.arrow = function (latlngs, options) {
        return new L.Polyline.Arrow(latlngs, options);
    };

    // Add hospital marker with correct coordinates
    console.log('Adding hospital marker...');
    L.marker([40.340339, -74.623913], {
        icon: L.divIcon({
            className: 'hospital-marker',
            html: '🏥',
            iconSize: [25, 25]
        })
    }).addTo(map);
    
    // Add base location marker
    console.log('Adding base location marker...');
    L.marker([40.361395, -74.664879], {
        icon: L.divIcon({
            className: 'base-marker',
            html: '🚒',
            iconSize: [25, 25]
        })
    }).addTo(map);

    // Socket event handlers
    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('simulation_state', (state) => {
        console.log('Received simulation state:', state);
        updateVisualization(state);
    });
    
    socket.on('simulation_ready', (data) => {
        console.log('Simulation ready:', data);
        // Add a message to the log
        window.simulationLogs.push(data.message);
        // Update the statistics to show the new log
        updateStatistics({});
    });

    // Button event handlers
    const startBtn = document.getElementById('startBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const stepBtn = document.getElementById('stepBtn');

    if (startBtn) startBtn.addEventListener('click', () => {
        console.log('Start button clicked');
        socket.emit('start_simulation');
    });

    if (pauseBtn) pauseBtn.addEventListener('click', () => {
        console.log('Pause button clicked');
        socket.emit('pause_simulation');
    });

    if (stepBtn) stepBtn.addEventListener('click', () => {
        console.log('Step button clicked');
        socket.emit('step_simulation');
    });

    function updateVisualization(state) {
        logDebug('Updating visualization with state:', state);
        
        const routeStyle = (status) => {
            // colour legend matches server status codes
            return {
                0: "#4CAF50", // available
                1: "#FFC107", // responding
                2: "#FF5722", // on-scene
                3: "#F44336", // transporting
                4: "#9C27B0"  // at hospital
            }[status] || "blue";
        };
        
        // Update ambulance markers
        state.ambulances.forEach((amb) => {
            logDebug(`Processing ambulance ${amb.id}:`, amb);
            const newPos = [amb.location.lat, amb.location.lng];
            logDebug(`New position for ambulance ${amb.id}:`, newPos);

            if (!ambulanceMarkers[amb.id]) {
                logDebug(`Creating new marker for ambulance ${amb.id}`);
                // Create a custom icon with a larger size
                const icon = L.divIcon({
                    className: "ambulance-marker",
                    html: "🚑",
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                });
                
                ambulanceMarkers[amb.id] = L.marker(newPos, {
                    icon: icon,
                    zIndexOffset: 1000,
                }).addTo(map);
                
                logDebug(`Marker created for ambulance ${amb.id}`);
            }

            // draw / refresh the route polyline
            if (ambulanceMarkers[amb.id].route) {
                logDebug(`Removing existing route for ambulance ${amb.id}`);
                map.removeLayer(ambulanceMarkers[amb.id].route);
            }
            
            if (amb.path && amb.path.length > 1) {
                logDebug(`Drawing path for ambulance ${amb.id}:`, amb.path);
                // Create the path with arrows
                ambulanceMarkers[amb.id].route = L.polyline.arrow(amb.path, {
                    color: routeStyle(amb.status),
                    weight: 4,
                    opacity: 0.8,
                    dashArray: "5,10",
                    arrowheads: [{
                        frequency: '50px',
                        size: '15px',
                        offset: '25%',
                        fill: true
                    }]
                }).addTo(map);
                logDebug(`Path drawn for ambulance ${amb.id}`);

                // Initialize or update animation state
                if (!animationStates[amb.id]) {
                    logDebug(`Initializing animation state for ambulance ${amb.id}`);
                    animationStates[amb.id] = {
                        currentIndex: 0,
                        path: amb.path,
                        lastUpdate: Date.now()
                    };
                } else {
                    logDebug(`Updating animation state for ambulance ${amb.id}`);
                    animationStates[amb.id].path = amb.path;
                }
            }

            // Animate position along path
            if (amb.path && amb.path.length > 1) {
                const animState = animationStates[amb.id];
                const now = Date.now();
                const timeSinceLastUpdate = now - animState.lastUpdate;
                
                // Update position every 100ms
                if (timeSinceLastUpdate > 100) {
                    animState.lastUpdate = now;
                    
                    // Move to next point in path
                    if (animState.currentIndex < amb.path.length - 1) {
                        animState.currentIndex++;
                    } else {
                        animState.currentIndex = 0;
                    }
                    
                    // Update ambulance position
                    const currentPoint = amb.path[animState.currentIndex];
                    logDebug(`Moving ambulance ${amb.id} to:`, currentPoint);
                    ambulanceMarkers[amb.id].setLatLng(currentPoint);
                    
                    // Calculate rotation angle for ambulance icon
                    if (animState.currentIndex < amb.path.length - 1) {
                        const current = amb.path[animState.currentIndex];
                        const next = amb.path[animState.currentIndex + 1];
                        const angle = Math.atan2(next[1] - current[1], next[0] - current[0]) * 180 / Math.PI;
                        logDebug(`Rotating ambulance ${amb.id} to angle:`, angle);
                        
                        const iconElement = ambulanceMarkers[amb.id].getElement();
                        if (iconElement) {
                            iconElement.style.transform = `rotate(${angle}deg)`;
                        } else {
                            logDebug(`Could not find icon element for ambulance ${amb.id}`);
                        }
                    }
                }
            } else {
                // If no path, just update position directly
                logDebug(`Setting direct position for ambulance ${amb.id}:`, [amb.location.lat, amb.location.lng]);
                ambulanceMarkers[amb.id].setLatLng([amb.location.lat, amb.location.lng]);
            }
        });
        
        // Update call markers - simplified to only show active calls
        // First, remove calls that are no longer active
        Object.keys(callMarkers).forEach((callId) => {
            if (!state.active_calls.some((call) => call.id == callId)) {  // ← loose ≈
                map.removeLayer(callMarkers[callId]);
                delete callMarkers[callId];
            }
        });
        
        // Then add or update active calls
        state.active_calls.forEach(call => {
            if (!callMarkers[call.id]) {
                // Create new marker
                callMarkers[call.id] = L.marker([call.location.lat, call.location.lng], {
                    icon: L.divIcon({
                        className: 'call-marker',
                        html: '🚨',
                        iconSize: [25, 25]
                    })
                }).addTo(map);
            } else {
                // Update existing marker position
                callMarkers[call.id].setLatLng([call.location.lat, call.location.lng]);
            }
        });

        // Update statistics
        updateStatistics(state.statistics);
    }

    function updateStatistics(stats) {
        logDebug('Updating statistics:', stats);
        const statsContent = document.getElementById('statsContent');
        if (statsContent) {
            // Create a detailed log section
            let logHtml = '<div class="log-section">';
            logHtml += '<h4>Simulation Log</h4>';
            
            // Get the last 5 log entries from the console
            const logs = window.simulationLogs || [];
            logs.forEach(log => {
                logHtml += `<div class="log-entry">${log}</div>`;
            });
            
            logHtml += '</div>';
            
            statsContent.innerHTML = `
                <div class="stats-section">
                    <h4>Statistics</h4>
                    <p>Active Calls: ${stats.active_calls || 0}</p>
                    <p>Available Ambulances: ${stats.available_ambulances || 0}</p>
                    <p>Patients Served: ${stats.patients_served || 0}</p>
                    <p>Missed Calls: ${stats.missed_calls || 0}</p>
                </div>
                ${logHtml}
            `;
        }
    }
    
    // Store simulation logs
    window.simulationLogs = [];
    
    // Override console.log to capture logs
    const originalConsoleLog = console.log;
    console.log = function() {
        // Call the original console.log
        originalConsoleLog.apply(console, arguments);
        
        // Add to our logs array
        const logEntry = Array.from(arguments).join(' ');
        window.simulationLogs.push(logEntry);
        
        // Keep only the last 10 logs
        if (window.simulationLogs.length > 10) {
            window.simulationLogs.shift();
        }
    };
}); 