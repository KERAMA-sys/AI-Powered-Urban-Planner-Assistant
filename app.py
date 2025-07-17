from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import logging
import webbrowser
from threading import Timer
from util.Feature_extraction import extract_all_features_from_polygon
from util.meta_model import rule_based_meta_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for frontend integration

@app.route('/')
def index():
    print("Rendering index.html template")
    return render_template('index.html')

@app.route('/predict-zoning', methods=['POST'])
def predict_zoning():
    """
    API endpoint to analyze a GeoJSON polygon and return zoning recommendations
    along with improvement suggestions and extracted features.
    """
    print("Received request to /predict-zoning")
    
    # Step 1: Validate input
    try:
        geojson = request.get_json()
        print(f"Received GeoJSON: {json.dumps(geojson)[:200]}...")  # Print first 200 chars
        
        # Extract coordinates from GeoJSON
        if geojson.get('type') == 'Feature':
            geometry = geojson.get('geometry', {})
        else:
            geometry = geojson
            
        if geometry.get('type') != 'Polygon':
            print(f"Invalid geometry type: {geometry.get('type')}")
            return jsonify({'error': 'Invalid geometry type. Expected Polygon.'}), 400
            
        polygon = geometry.get('coordinates', [])[0]  # First ring of coordinates
        
        if not polygon or not isinstance(polygon, list):
            print("Invalid or missing polygon coordinates")
            return jsonify({'error': 'Invalid or missing polygon coordinates'}), 400
            
        print(f"Extracted polygon with {len(polygon)} points")
    
    except Exception as e:
        print(f"Error parsing request data: {str(e)}")
        return jsonify({'error': f'Error parsing request data: {str(e)}'}), 400
    
    # Step 2: Extract features
    try:
        print("Starting feature extraction...")
        features = extract_all_features_from_polygon(polygon)
        print(f"Features extracted: {features}")
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500
    
    # Step 3: Meta-model decision
    try:
        print("Running meta-model...")
        zoning, suggestions = rule_based_meta_model(features)
        print(f"Zoning recommendation: {zoning}")
        print(f"Improvement suggestions: {suggestions}")
    except Exception as e:
        print(f"Meta-model failed: {str(e)}")
        return jsonify({'error': f'Meta-model failed: {str(e)}'}), 500
    
    # Step 4: Prepare air pollution data
    # Extract relevant pollution data from features if available
    air_pollution_data = {
        'aqi': 75,  # Default value
        'pollutants': {
            'pm25': 12.5,
            'pm10': 35.2,
            'o3': 0.045,
            'no2': 38.6,
            'co': 1.2,
            'so2': 8.3
        }
    }
    
    # If pollution data exists in features, use it
    if 'Pollution' in features:
        pollution = features['Pollution']
        if 'raw' in pollution:
            raw_pollution = pollution['raw']
            # Update with actual values if available
            if 'CO AQI Value' in raw_pollution:
                air_pollution_data['pollutants']['co'] = raw_pollution['CO AQI Value']
            if 'Ozone AQI Value' in raw_pollution:
                air_pollution_data['pollutants']['o3'] = raw_pollution['Ozone AQI Value']
            if 'NO2 AQI Value' in raw_pollution:
                air_pollution_data['pollutants']['no2'] = raw_pollution['NO2 AQI Value']
            if 'PM2.5 AQI Value' in raw_pollution:
                air_pollution_data['pollutants']['pm25'] = raw_pollution['PM2.5 AQI Value']
    
    # Step 5: Prepare traffic data
    traffic_data = {
        'congestion_level': 'Medium',  # Default value
        'metrics': {
            'average_speed': 28.5,
            'peak_hour_congestion': 'High',
            'public_transport_access': 'Good',
            'pedestrian_connectivity': 'Medium'
        }
    }
    
    # If traffic data exists in features, use it
    if 'Traffic' in features:
        traffic = features['Traffic']
        if 'raw' in traffic:
            raw_traffic = traffic['raw']
            # Update with actual values if available
            if 'Speed_Deviation' in raw_traffic:
                traffic_data['metrics']['average_speed'] = 30 - raw_traffic['Speed_Deviation']
            if 'Is_Peak_Hour' in raw_traffic:
                traffic_data['metrics']['peak_hour_congestion'] = 'High' if raw_traffic['Is_Peak_Hour'] == 1 else 'Low'
    
    # Step 6: Flatten and clean features for display
    display_features = {}
    
    # Process LandUse features
    if 'LandUse' in features:
        for key, value in features['LandUse'].items():
            display_features[key] = value
    
    # Add centroid information
    if 'centroid' in features:
        display_features['latitude'] = features['centroid']['lat']
        display_features['longitude'] = features['centroid']['lon']
    
    # Step 7: Return combined response
    response = {
        'zoning_recommendation': zoning,
        'improvement_suggestions': suggestions,
        'features': display_features,
        'air_pollution': air_pollution_data,
        'traffic': traffic_data
    }
    
    print("Returning complete response")
    return jsonify(response), 200

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1.0, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True)