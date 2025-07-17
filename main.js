document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded and parsed');
    
    // Initialize map
    console.log('Initializing map...');
    const map = L.map('map').setView([37.7749, -122.4194], 12); // Default: San Francisco
    console.log('Map initialized with San Francisco view');

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
    console.log('Map tiles loaded');

    const drawnItems = new L.FeatureGroup().addTo(map);
    console.log('Feature group for drawn items created');

    const drawControl = new L.Control.Draw({
        edit: { featureGroup: drawnItems },
        draw: {
            polygon: {
                allowIntersection: false,
                showArea: true,
                shapeOptions: { color: 'green' }
            },
            rectangle: false,
            polyline: false,
            circle: false,
            marker: false
        }
    });
    map.addControl(drawControl);
    console.log('Draw control added to map');

    let currentPin = null;
    let currentPolygon = null;

    // Handle tab switching
    console.log('Setting up tab switching...');
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            console.log(`Tab ${button.dataset.tab} clicked`);
            
            // Remove active class from all buttons and tabs
            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            
            // Add active class to current button and tab
            button.classList.add('active');
            document.getElementById(`${button.dataset.tab}-tab`).classList.add('active');
            console.log(`Switched to ${button.dataset.tab} tab`);
        });
    });

    // Handle polygon drawing
    console.log('Setting up polygon drawing handler...');
    map.on('draw:created', function (e) {
        console.log('New polygon drawn');
        if (currentPolygon) {
            console.log('Removing previous polygon');
            drawnItems.removeLayer(currentPolygon);
        }
        
        const layer = e.layer;
        currentPolygon = layer;
        drawnItems.addLayer(layer);
        console.log('New polygon added to layer');
        
        const geojson = layer.toGeoJSON();
        console.log('Polygon converted to GeoJSON:', geojson);
        
        showLoadingIndicator();
        sendPolygonToBackend(geojson);
    });

    // Handle polygon editing
    console.log('Setting up polygon edit handler...');
    map.on('draw:edited', function(e) {
        console.log('Polygon edited');
        const layers = e.layers;
        layers.eachLayer(function(layer) {
            const geojson = layer.toGeoJSON();
            console.log('Edited polygon converted to GeoJSON:', geojson);
            
            showLoadingIndicator();
            sendPolygonToBackend(geojson);
        });
    });

    // On click – drop invisible box + styled pin
    console.log('Setting up map click handler...');
    map.on('click', function(e) {
        console.log(`Map clicked at lat: ${e.latlng.lat}, lng: ${e.latlng.lng}`);
        dropPinAndSendBox(e.latlng.lat, e.latlng.lng, 0.0045);
    });

    // GeoSearch autocomplete
    console.log('Setting up GeoSearch...');
    const provider = new window.GeoSearch.OpenStreetMapProvider();
    const searchControl = new window.GeoSearch.GeoSearchControl({
        provider: provider,
        style: 'bar',
        searchLabel: 'Search location...',
        autoComplete: true,
        autoCompleteDelay: 250,
        showMarker: false,
    });
    map.addControl(searchControl);
    console.log('GeoSearch control added to map');

    // Handle search results selection
    map.on('geosearch/showlocation', function(e) {
        const { location } = e;
        console.log(`Location selected: ${location.label} (${location.y}, ${location.x})`);
        dropPinAndSendBox(location.y, location.x, 0.0045);
    });

    // Custom pin and backend call
    function dropPinAndSendBox(lat, lng, deltaLat) {
        console.log(`Dropping pin at lat: ${lat}, lng: ${lng} with delta: ${deltaLat}`);
        const deltaLng = deltaLat / Math.cos(lat * Math.PI / 180);
        console.log(`Calculated deltaLng: ${deltaLng}`);

        if (currentPin) {
            console.log('Removing previous pin');
            map.removeLayer(currentPin);
        }

        // Custom Google-style marker
        const pinIcon = L.divIcon({
            className: 'custom-pin',
            iconSize: [22, 40],
            iconAnchor: [11, 40]
        });

        currentPin = L.marker([lat, lng], { icon: pinIcon }).addTo(map);
        console.log('New pin added to map');

        // Prepare invisible bounding box
        const bounds = [
            [lat - deltaLat, lng - deltaLng],
            [lat - deltaLat, lng + deltaLng],
            [lat + deltaLat, lng + deltaLng],
            [lat + deltaLat, lng - deltaLng],
            [lat - deltaLat, lng - deltaLng]
        ];
        console.log('Bounding box coordinates created:', bounds);

        // Create or update polygon
        if (currentPolygon) {
            console.log('Removing previous polygon');
            drawnItems.removeLayer(currentPolygon);
        }
        
        currentPolygon = L.polygon(bounds, {
            color: 'blue',
            fillOpacity: 0.2,
            weight: 2
        });
        
        drawnItems.addLayer(currentPolygon);
        const geojson = currentPolygon.toGeoJSON();
        console.log('Bounding box polygon converted to GeoJSON');
        
        showLoadingIndicator();
        sendPolygonToBackend(geojson);
    }

    // Send polygon to backend
    function sendPolygonToBackend(geojsonPolygon) {
        console.log('Sending polygon to backend:', geojsonPolygon);
        
        fetch('/predict-zoning', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(geojsonPolygon)
        })
        .then(response => {
            console.log('Received response from server:', response.status);
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Model Predictions:', data);
            hideLoadingIndicator();
            displayResults(data);
        })
        .catch(error => {
            console.error('🚨 Error:', error);
            hideLoadingIndicator();
            showError(error.message);
        });
    }

    // Display all model results in the panel
    function displayResults(data) {
        console.log('Displaying results in panel:', data);
        document.getElementById('results-panel').style.display = 'block';
        
        // Update zoning recommendation
        document.getElementById('zoning-value').textContent = data.zoning_recommendation || 'Not available';
        console.log('Updated zoning value');
        
        // Update suggestions list
        const suggestionsList = document.getElementById('suggestions-list');
        suggestionsList.innerHTML = '';
        
        if (data.improvement_suggestions && data.improvement_suggestions.length > 0) {
            console.log(`Adding ${data.improvement_suggestions.length} suggestions`);
            data.improvement_suggestions.forEach(suggestion => {
                const li = document.createElement('li');
                li.textContent = suggestion;
                suggestionsList.appendChild(li);
            });
        } else {
            console.log('No suggestions available');
            const li = document.createElement('li');
            li.textContent = 'No specific suggestions available.';
            suggestionsList.appendChild(li);
        }
        
        // Update air quality tab
        console.log('Updating air quality tab');
        updateAirQualityTab(data.air_pollution);
        
        // Update traffic tab
        console.log('Updating traffic tab');
        updateTrafficTab(data.traffic);
        
        // Update features list
        console.log('Updating features tab');
        updateFeaturesTab(data.features);
    }
    
    // Update air quality tab
    function updateAirQualityTab(airData) {
        console.log('Air quality data:', airData);
        const airQualityResults = document.getElementById('air-quality-results');
        
        if (!airData || airData.error) {
            console.log('Air quality data not available');
            airQualityResults.innerHTML = `<p class="no-data">Air quality prediction not available</p>`;
            if (airData && airData.error) {
                airQualityResults.innerHTML += `<p class="error-message">${airData.error}</p>`;
            }
            return;
        }
        
        // Assuming airData has structure with AQI and pollutant levels
        let html = `
            <div class="air-quality-card">
                <div class="aqi-value">${airData.aqi || 'N/A'}</div>
                <div class="aqi-label">Air Quality Index</div>
                <div class="aqi-category ${getAQIColorClass(airData.aqi)}">
                    ${getAQICategory(airData.aqi)}
                </div>
            </div>
            <div class="pollutant-levels">
                <h4>Pollutant Levels</h4>
                <div class="pollutant-grid">
        `;
        
        // Add pollutant details if available
        if (airData.pollutants) {
            console.log('Adding pollutant details');
            for (const [pollutant, value] of Object.entries(airData.pollutants)) {
                html += `
                    <div class="pollutant-item">
                        <div class="pollutant-name">${formatPollutantName(pollutant)}</div>
                        <div class="pollutant-value">${value.toFixed(2)}</div>
                    </div>
                `;
            }
        }
        
        html += `
                </div>
            </div>
        `;
        
        airQualityResults.innerHTML = html;
        console.log('Air quality tab updated');
    }
    
    // Update traffic tab
    function updateTrafficTab(trafficData) {
        console.log('Traffic data:', trafficData);
        const trafficResults = document.getElementById('traffic-results');
        
        if (!trafficData || trafficData.error) {
            console.log('Traffic data not available');
            trafficResults.innerHTML = `<p class="no-data">Traffic prediction not available</p>`;
            if (trafficData && trafficData.error) {
                trafficResults.innerHTML += `<p class="error-message">${trafficData.error}</p>`;
            }
            return;
        }
        
        // Assuming trafficData has structure with congestion levels and other metrics
        let html = `
            <div class="traffic-card">
                <div class="traffic-value ${getCongestionColorClass(trafficData.congestion_level)}">
                    ${trafficData.congestion_level || 'N/A'}
                </div>
                <div class="traffic-label">Congestion Level</div>
            </div>
            <div class="traffic-metrics">
                <h4>Traffic Metrics</h4>
                <div class="metrics-grid">
        `;
        
        // Add metrics details if available
        if (trafficData.metrics) {
            console.log('Adding traffic metrics details');
            for (const [metric, value] of Object.entries(trafficData.metrics)) {
                html += `
                    <div class="metric-item">
                        <div class="metric-name">${formatMetricName(metric)}</div>
                        <div class="metric-value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
                    </div>
                `;
            }
        }
        
        html += `
                </div>
            </div>
        `;
        
        trafficResults.innerHTML = html;
        console.log('Traffic tab updated');
    }
    
    // Update features tab
    function updateFeaturesTab(features) {
        console.log('Features data:', features);
        const featuresList = document.getElementById('features-list');
        featuresList.innerHTML = '';
        
        if (features) {
            console.log(`Adding ${Object.keys(features).length} features`);
            for (const [key, value] of Object.entries(features)) {
                const item = document.createElement('div');
                item.className = 'feature-item';
                
                const label = document.createElement('span');
                label.className = 'feature-label';
                label.textContent = formatFeatureName(key) + ':';
                
                const val = document.createElement('span');
                val.className = 'feature-value';
                val.textContent = formatFeatureValue(value);
                
                item.appendChild(label);
                item.appendChild(val);
                featuresList.appendChild(item);
            }
        } else {
            console.log('No features available');
            featuresList.innerHTML = '<p class="no-data">No feature data available</p>';
        }
        console.log('Features tab updated');
    }
    
    // Helper functions
    function formatFeatureName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    function formatFeatureValue(value) {
        if (typeof value === 'number') {
            return Number.isInteger(value) ? value : value.toFixed(2);
        }
        return value;
    }
    
    function formatPollutantName(name) {
        const pollutantMap = {
            'pm25': 'PM 2.5',
            'pm10': 'PM 10',
            'o3': 'Ozone (O₃)',
            'no2': 'NO₂',
            'so2': 'SO₂',
            'co': 'CO'
        };
        
        return pollutantMap[name.toLowerCase()] || name;
    }
    
    function formatMetricName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    function getAQIColorClass(aqi) {
        if (!aqi) return 'unknown';
        
        if (aqi <= 50) return 'good';
        if (aqi <= 100) return 'moderate';
        if (aqi <= 150) return 'sensitive';
        if (aqi <= 200) return 'unhealthy';
        if (aqi <= 300) return 'very-unhealthy';
        return 'hazardous';
    }
    
    function getAQICategory(aqi) {
        if (!aqi) return 'Unknown';
        
        if (aqi <= 50) return 'Good';
        if (aqi <= 100) return 'Moderate';
        if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
        if (aqi <= 200) return 'Unhealthy';
        if (aqi <= 300) return 'Very Unhealthy';
        return 'Hazardous';
    }
    
    function getCongestionColorClass(level) {
        if (!level) return 'unknown';
        
        level = level.toLowerCase();
        if (level.includes('low')) return 'good';
        if (level.includes('medium') || level.includes('moderate')) return 'moderate';
        if (level.includes('high')) return 'unhealthy';
        if (level.includes('severe')) return 'very-unhealthy';
        return 'unknown';
    }
    
    // Show loading indicator
    function showLoadingIndicator() {
        console.log('Showing loading indicator');
        document.getElementById('loading').style.display = 'flex';
        document.getElementById('results-content').style.display = 'none';
    }
    
    // Hide loading indicator
    function hideLoadingIndicator() {
        console.log('Hiding loading indicator');
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results-content').style.display = 'block';
    }
    
    // Show error in the results panel
    function showError(message) {
        console.error('Displaying error:', message);
        document.getElementById('results-panel').style.display = 'block';
        document.getElementById('loading').style.display = 'none';
        
        const content = document.getElementById('results-content');
        content.innerHTML = `
            <div class="error-message">
                <h3>Error</h3>
                <p>${message}</p>
                <p>Please try again or select a different area.</p>
            </div>
        `;
        content.style.display = 'block';
    }
});