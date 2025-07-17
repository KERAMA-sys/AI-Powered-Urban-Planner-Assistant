import os
import glob
import geopandas as gpd
import pandas as pd

# -------------------------------
# 1. CONFIGURATION & INPUT PATHS
# -------------------------------
input_dir = "geojson_with_elevation2"  # Folder containing your enriched GeoJSON files (one per city)
output_csv = "Consolidated_Land_Use_Data.csv"

# -------------------------------
# 2. DEFINE NLCD-to-ZONING MAPPING
# -------------------------------
# This dictionary maps NLCD land cover codes to zoning labels.
# Adjust these based on local planning standards.
nlcd_to_zoning = {
    11: "Open Water",           # Water bodies
    21: "Residential",          # Developed, Open Space (often residential or low-density)
    22: "Residential",          # Developed, Low Intensity
    23: "Commercial",           # Developed, Medium Intensity (often commercial/institutional)
    24: "Commercial",           # Developed, High Intensity (typically commercial/mixed-use)
    31: "Barren",               # Barren Land (potentially industrial or undeveloped)
    41: "Green Space",          # Deciduous Forest (parks/greenspace)
    42: "Green Space",          # Evergreen Forest
    43: "Green Space",          # Mixed Forest
    52: "Agricultural",         # Shrub/Scrub
    71: "Agricultural",         # Grassland/Herbaceous
    81: "Agricultural",         # Pasture/Hay
    82: "Agricultural",         # Cultivated Crops
    90: "Wetlands",             # Woody Wetlands
    95: "Wetlands"              # Emergent Herbaceous Wetlands
}

def infer_zoning(land_cover_code):
    """
    Given an NLCD land cover code, return the inferred zoning label.
    If not found in the mapping, return "Unknown".
    """
    return nlcd_to_zoning.get(land_cover_code, "Unknown")

# -------------------------------
# 3. DATA CONSOLIDATION & PREPROCESSING
# -------------------------------
# List to hold DataFrames from each GeoJSON
dfs = []

# Expecting file names like "CityName_USA_NLCD_with_elevation.geojson"
for filepath in glob.glob(os.path.join(input_dir, "*_USA_NLCD_with_elevation.geojson")):
    try:
        # Load the GeoJSON as a GeoDataFrame
        gdf = gpd.read_file(filepath)
        # (Optional) Ensure correct CRS (assuming the file uses CRS84/OGC:CRS84)
        # gdf = gdf.to_crs("EPSG:4326")
        
        # Convert properties (all columns except geometry) to a DataFrame
        df_city = pd.DataFrame(gdf.drop(columns="geometry"))
        
        # Add a "city" column from file name if not already present (optional)
        city_name = os.path.basename(filepath).split("_USA_NLCD_with_elevation")[0]
        if "city" not in df_city.columns:
            df_city["city"] = city_name
        
        dfs.append(df_city)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Combine DataFrames from all cities
consolidated_df = pd.concat(dfs, ignore_index=True)

# Drop extraneous columns (e.g., any city, fips or name columns if present) 
drop_cols = [col for col in consolidated_df.columns if col.lower() in ["city", "fips", "name"]]
consolidated_df = consolidated_df.drop(columns=drop_cols, errors='ignore')

# -------------------------------
# 4. ADD ZONING LABELS BASED ON NLCD LAND COVER
# -------------------------------
# Assume that the original "land_cover" column exists and holds the NLCD code (e.g., 21, 22, etc.)
# If your dataset already has one-hot encoded versions, you may need to recover the original code.
if "land_cover" in consolidated_df.columns:
    consolidated_df["zoning_label"] = consolidated_df["land_cover"].apply(infer_zoning)
else:
    print("Warning: 'land_cover' column not found. Cannot assign zoning labels.")

# -------------------------------
# 5. SAVE THE FINAL CONSOLIDATED DATASET
# -------------------------------
consolidated_df.to_csv(output_csv, index=False)
print(f"✅ Consolidated data with zoning labels saved to {output_csv}")

# For inspection, you can also print the first few rows and column names:
print("Dataset columns:", consolidated_df.columns.tolist())
print(consolidated_df.head())
