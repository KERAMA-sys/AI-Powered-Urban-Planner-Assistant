import json
from .Feature_extraction import extract_all_features_from_polygon

# === Rule-based Meta-Model ===
def rule_based_meta_model(features):
    """
    Takes the unified features dict and returns a zoning recommendation
    along with improvement suggestions based on urban planning rules.
    """
    pollution = features['Pollution']['prediction_decoded']
    traffic_peak = features['Traffic']['raw']['Is_Peak_Hour']
    dist_to_water = features['LandUse']['dist_to_water_m']
    dist_to_road = features['LandUse']['dist_to_road_m']
    compactness = features['LandUse']['compactness']
    population_density = features['LandUse']['population_density']
    mean_slope = features['LandUse']['mean_slope']
    area = features['LandUse']['area']
    mean_elevation = features['LandUse']['mean_elevation']

    # Initial zoning decision
    if pollution in ['Unhealthy', 'Very Unhealthy', 'Hazardous']:
        zoning = 'Green Space / Ecological Reserve'
    elif traffic_peak and dist_to_road is not None and dist_to_road < 500:
        zoning = 'Commercial / Mixed-Use'
    elif area > 10000 and compactness and compactness > 0.7:
        zoning = 'Residential'
    elif population_density > 50000:
        zoning = 'Urban Core / High-Density Residential'
    elif dist_to_water is not None and dist_to_water < 500:
        zoning = 'Recreational / Waterfront Development'
    else:
        zoning = 'Mixed Use / Adaptive'

    # Improvement suggestions
    suggestions = []

    # Rule 1-5: Pollution + Traffic
    if pollution in ['Unhealthy', 'Very Unhealthy', 'Hazardous'] and traffic_peak:
        suggestions.append('Increase green cover and improve public transit to reduce emissions.')
    if pollution == 'Moderate' and dist_to_road < 1000:
        suggestions.append('Encourage rooftop gardens and optimize traffic flow near roads.')
    if pollution == 'Moderate' and traffic_peak:
        suggestions.append('Promote carpooling and urban forestry to improve air quality.')
    if pollution in ['Unhealthy', 'Very Unhealthy'] and compactness < 0.5:
        suggestions.append('Restrict industrial activities and encourage compact development.')
    if pollution == 'Moderate' and population_density > 30000:
        suggestions.append('Expand green spaces to balance high population density.')

    # Rule 6-10: Traffic + Land-Use
    if traffic_peak and dist_to_road > 1500:
        suggestions.append('Develop additional road connections and enhance public transit.')
    if traffic_peak and compactness < 0.3:
        suggestions.append('Promote mixed-use developments to reduce traffic congestion.')
    if traffic_peak and population_density > 50000:
        suggestions.append('Introduce congestion pricing and expand public transit options.')
    if dist_to_road < 500 and compactness > 0.7:
        suggestions.append('Encourage pedestrian-friendly infrastructure near dense areas.')
    if dist_to_road > 2000 and area > 20000:
        suggestions.append('Plan for arterial roads to connect large, isolated areas.')

    # Rule 11-15: Pollution + Land-Use
    if pollution in ['Unhealthy', 'Very Unhealthy'] and dist_to_water < 500:
        suggestions.append('Develop green buffers near water bodies to improve air quality.')
    if pollution == 'Moderate' and mean_slope > 0.1:
        suggestions.append('Implement slope stabilization and promote vegetation to reduce erosion.')
    if pollution in ['Unhealthy', 'Very Unhealthy'] and area > 20000:
        suggestions.append('Designate portions of large areas for ecological restoration.')
    if pollution == 'Moderate' and compactness > 0.7:
        suggestions.append('Encourage vertical construction to optimize land use.')
    if pollution == 'Moderate' and mean_elevation > 500:
        suggestions.append('Plan for high-altitude infrastructure to support sustainable development.')

    # Rule 16-20: Land-Use + Population Density
    if population_density < 1000 and compactness < 0.3:
        suggestions.append('Attract businesses and residents to underutilized areas.')
    if population_density > 50000 and compactness > 0.7:
        suggestions.append('Develop high-density housing to accommodate population growth.')
    if population_density > 50000 and dist_to_water < 500:
        suggestions.append('Encourage waterfront development for high-density areas.')
    if population_density < 1000 and area > 20000:
        suggestions.append('Plan for affordable housing projects in large, underutilized areas.')
    if population_density > 50000 and mean_slope > 0.1:
        suggestions.append('Implement slope stabilization measures in high-density areas.')

    # Rule 21-25: Slope + Elevation
    if mean_slope > 0.1 and mean_elevation > 500:
        suggestions.append('Ensure infrastructure accounts for steep slopes and high altitudes.')
    if mean_slope > 0.1 and dist_to_water < 500:
        suggestions.append('Promote terracing and vegetation near water bodies to stabilize slopes.')
    if mean_slope > 0.1 and compactness < 0.3:
        suggestions.append('Avoid large-scale construction on steep slopes to reduce risks.')
    if mean_elevation > 500 and area > 20000:
        suggestions.append('Plan for sustainable development in large, high-altitude areas.')
    if mean_elevation > 500 and dist_to_road > 2000:
        suggestions.append('Ensure road infrastructure is optimized for high-altitude conditions.')

    # Rule 26-30: Water Access + Recreation
    if dist_to_water > 2000 and area > 20000:
        suggestions.append('Improve water access infrastructure in large, isolated areas.')
    if dist_to_water < 500 and compactness > 0.7:
        suggestions.append('Develop recreational areas near dense, waterfront locations.')
    if dist_to_water < 500 and population_density > 50000:
        suggestions.append('Encourage waterfront tourism and recreation in high-density areas.')
    if dist_to_water > 2000 and mean_slope > 0.1:
        suggestions.append('Plan for sustainable water management in steep, isolated areas.')
    if dist_to_water > 2000 and pollution in ['Unhealthy', 'Very Unhealthy']:
        suggestions.append('Implement water conservation measures to address pollution concerns.')

    # Limit to 3 suggestions
    suggestions = suggestions[:3]

    return zoning, suggestions

# === Command-line Interface ===
if __name__ == '__main__':
    # Expect a GeoJSON-like polygon as JSON string input
    import sys
    polygon_json = sys.argv[1]
    polygon = json.loads(polygon_json)
    features = extract_all_features_from_polygon(polygon)
    zoning, suggestions = rule_based_meta_model(features)
    output = {
        'zoning_recommendation': zoning,
        'improvement_suggestions': suggestions
    }
    print(json.dumps(output, indent=2))
