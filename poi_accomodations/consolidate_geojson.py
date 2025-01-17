import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from rapidfuzz import fuzz
import sys
from multiprocessing import Pool

country_crs_mapping = {
    'albania': 32634,
    'kosovo': 32634,
    'andorra': 32631,
    'latvia': 32635,
    'austria': 32633,
    'liechtenstein': 32632,
    'azores': 32626,
    'lithuania': 32635,
    'belarus': 32635,
    'luxembourg': 32631,
    'belgium': 32631,
    'macedonia': 32634,
    'bosnia-herzegovina': 32633,
    'malta': 32633,
    'bulgaria': 32635,
    'moldova': 32635,
    'croatia': 32633,
    'monaco': 32632,
    'cyprus': 32636,
    'montenegro': 32634,
    'czech-republic': 32633,
    'netherlands': 32631,
    'denmark': 32632,
    'norway': 32632,
    'estonia': 32635,
    'poland': 32634,
    'faroe-islands': 32629,
    'portugal': 32629,
    'finland': 32635,
    'romania': 32635,
    'france': 32631,  # Defaulting to Western France
    'russia': 32636,  # Defaulting to Western Russia
    'georgia': 32638,
    'serbia': 32634,
    'germany': 32632,  # Defaulting to Western Germany
    'slovakia': 32633,
    'greece': 32634,  # Defaulting to Western Greece
    'slovenia': 32633,
    'guernsey-jersey': 32630,
    'spain': 32630,  # Defaulting to Western Spain
    'hungary': 32634,
    'sweden': 32633,  # Defaulting to Southern Sweden
    'iceland': 32627,
    'switzerland': 32632,
    'ireland-and-northern-ireland': 32629,
    'turkey': 32636,
    'isle-of-man': 32630,
    'ukraine': 32636,
    'italy': 32632,  # Defaulting to Western Italy
    'united-kingdom': 32630
}

def clean_polygons(df, threshold=80):
    overlapping = gpd.sjoin(df, df, how='inner') # overlapping polygons
    overlapping = overlapping[overlapping.id_left != overlapping.id_right] # remove self joins
    overlapping = overlapping[overlapping.id_left < overlapping.id_right] # remove swapped orders e.g. (A,B) and (B,A)
    if "name_left" in overlapping.columns and "name_right" in overlapping.columns:
        overlapping["similarity_score"] = overlapping.apply(
            lambda row: fuzz.ratio(row["name_left"], row["name_right"]), axis=1
        )
        duplicates = overlapping[overlapping["similarity_score"] > threshold] # overlapping and similar name
    else:
        duplicates = overlapping

    duplicates["keep_id"] = duplicates.apply(
        lambda row: row["id_left"] if row["tag_count_left"] >= row["tag_count_right"] else row["id_right"],
        axis=1
    )
    to_remove = duplicates.apply(
        lambda row: row["id_right"] if row["keep_id"] == row["id_left"] else row["id_left"],
        axis=1
    )
    cleaned_df = df.loc[~df["id"].isin(to_remove)].copy()
    return cleaned_df

def clean_points(points, polygons):
    """
    Deduplicates the given DataFrame based on overlap with an existing polygon.
    Args:
        points (gpd.DataFrame): GeoDataFrame to deduplicate
        polygons (gpd.DataFrame): GeoDataFrame to base deduplication on
    Returns:
        gpd.DataFrame: Deduplicated GeoDataFrame
    """
    points_in_polygons = points.sjoin(polygons, how="inner", predicate="within")
    to_remove = points_in_polygons.index
    cleaned_df = points.loc[~points.index.isin(to_remove)].copy()
    return cleaned_df

def deduplicate_data(gdf):
    """
    Deduplicates the given DataFrame.
    Args:
        gdf (gpd.DataFrame): GeoDataFrame to deduplicate
    Returns:
        gpd.DataFrame: Deduplicated GeoDataFrame
    """
    country = gdf['country'].iloc[0].lower()  # Assumes all rows in gdf belong to one country
    target_epsg = country_crs_mapping.get(country, 3035)  # Default to EPSG:3035 for Europe

    # Transform GeoDataFrame to the appropriate CRS
    gdf = gdf.to_crs(epsg=target_epsg)
    
    gdf_points = gdf[gdf.geometry.type == "Point"].copy()
    gdf_polygons = gdf[gdf.geometry.type.isin(["LineString", "MultiPolygon"])].copy() # way or area
    polygons_clean = None
    points_clean = None

    # Process polygons if any exist
    if not gdf_polygons.empty:
        polygons_clean = clean_polygons(gdf_polygons)
        # Convert polygon centroids to points
        polygons_clean["geometry"] = polygons_clean.centroid
    
    # Process points if any exist
    if not gdf_points.empty:
        # If polygons_clean exists, use it for filtering; otherwise, skip filtering
        if polygons_clean is not None:
            points_clean = clean_points(gdf_points, polygons_clean)
        else:
            points_clean = gdf_points
        
    return polygons_clean, points_clean

def process_to_features(gdf):
    gdf = gdf.to_crs(4326)
    #gdf['lat'] = gdf['geometry'].y
    #gdf['long'] = gdf['geometry'].x
    if 'wikimedia_commons' in gdf.columns:
        gdf['has_wikimedia_commons'] = gdf['wikimedia_commons'].notnull().astype(int)
    else:
        gdf['has_wikimedia_commons'] = 0
        
    if 'wikipedia' in gdf.columns:
        gdf['has_wikipedia'] = gdf['wikipedia'].notnull().astype(int)
    else:
        gdf['has_wikipedia'] = 0
    
    #gdf = gdf.drop(labels=["geometry"], axis=1)
    #gdf["geometry"] = gpd.GeoSeries.from_wkt(gdf["geometry"])
    columns_to_keep = ['id', 'name', 'tag_count', 'geometry', 'has_wikimedia_commons', 'has_wikipedia', 'country', 'tag_key', 'tag_value'] # memory fix
    gdf = gdf.loc[:, [col for col in columns_to_keep if col in gdf.columns]]
    return gdf

def process_file(args):
    """
    Loads and deduplicates a single GeoJSON file.
    Args:
        file_path (str): Path to the GeoJSON file
    Returns:
        pd.DataFrame: Deduplicated DataFrame
    """
    file_path, file_name = args
    print(f"Processing {file_path}...")
    gdf = gpd.read_file(file_path)
    if gdf.empty:
        print(f"File {file_path} is empty, skipping...")
        return None
    country, key, value = extract_values_from_filename(file_name)
    gdf["tag_count"] = gdf.apply(
        lambda row: row.notna().sum() - 3, axis=1 # count tags other than id, name, geometry
    )
    gdf["country"] = country
    gdf["tag_key"] = key
    gdf["tag_value"] = value
    columns_to_keep = ['id', 'name', 'tag_count', 'geometry', 'wikimedia_commons', 'wikipedia', 'country', 'tag_key', 'tag_value'] # memory fix
    gdf = gdf.loc[:, [col for col in columns_to_keep if col in gdf.columns]]
    polygons, points = deduplicate_data(gdf)

    if polygons is not None and not polygons.empty and points is not None and not points.empty:
        polygons["geometry"] = polygons.centroid
        final_gdf = pd.concat([points, polygons], ignore_index=True)
        return process_to_features(final_gdf)
    
    if polygons is not None and not polygons.empty and points is None:
        print("empty points but polygons")
        return process_to_features(polygons)
    
    if polygons is None and points is not None and not points.empty:
        print("empty polygons but points")
        return process_to_features(points)
    
    if polygons is None and points is None:
        print("both empty")
        return None


def extract_metadata_from_filename(file_name):
    """
    Extract metadata (country, key, value) from the file name.
    Args:
        file_name (str): File name to parse.
    Returns:
        tuple: Extracted (country, key, value) as strings.
    """
    base_name = file_name.replace(".geojson", "")
    parts = base_name.split("-")
    
    # Ensure the file name has enough parts to extract meaningful metadata
    if len(parts) < 4:
        raise ValueError(f"File name {file_name} is not in the expected format.")
    
    country = parts[0]  # e.g., 'austria'
    key = parts[-2]     # e.g., 'craft'
    value = parts[-1]   # e.g., 'handicraft'
    
    return country, key, value

def extract_values_from_filename(filename: str) -> tuple[str, str, str]:
    """
    Given a filename of the format:

        <country>-latest-keys-<key>-<value>.geojson

    where:
        - country can have 1 or more words (joined by '-')
        - "latest" and "keys" are fixed markers and not extracted
        - key is exactly 1 word
        - value can have 1 or more words (joined by '-')

    This function returns a tuple of (country, key, value).

    Examples:
        "guernsey-jersey-latest-keys-amenity-exhibition-centre.geojson" 
            -> ("guernsey-jersey", "amenity", "exhibition-centre")
        "macedonia-latest-keys-building-military.geojson"
            -> ("macedonia", "building", "military")

    :param filename: The filename to parse (with or without the .geojson extension).
    :return: A tuple (country, key, value).
    """
    # Strip out the extension if present (e.g., .geojson)
    base_name = os.path.splitext(filename)[0]

    # Split on '-'
    parts = base_name.split('-')

    # Find the index of 'latest'; we assume 'keys' comes immediately after
    # This is a simplified assumption based on known format:
    #   [country (1..n parts)]-latest-keys-[key (1 part)]-[value (1..n parts)]
    try:
        latest_idx = parts.index('latest')
    except ValueError:
        raise ValueError(f"'latest' not found in filename: {filename}")

    # country is everything before 'latest'
    country_parts = parts[:latest_idx]
    if not country_parts:
        raise ValueError(f"No country name found before 'latest' in filename: {filename}")
    country = '-'.join(country_parts)

    # skip 'latest' and the next part should be 'keys'
    # so the key should be after "latest" and "keys"
    # i.e. at index `latest_idx + 2`
    if len(parts) < latest_idx + 3:
        raise ValueError(f"Filename does not have enough parts to extract key and value: {filename}")
    if parts[latest_idx + 1] != 'keys':
        raise ValueError(f"Expected 'keys' after 'latest' in filename: {filename}")

    key = parts[latest_idx + 2]

    # The remainder (if any) is the value (can have 1 or more words)
    value_parts = parts[latest_idx + 3:]
    if not value_parts:
        raise ValueError(f"No value part found in filename: {filename}")
    value = '-'.join(value_parts)

    return country, key, value

def consolidate_files(directory, processes):
    """
    Processes all files in a directory, adds metadata from filenames, 
    and consolidates them into a single DataFrame.
    Args:
        directory (str): Directory containing GeoJSON files.
    Returns:
        pd.DataFrame: Consolidated and deduplicated DataFrame with metadata.
    """
    files = [
        (os.path.join(directory, file_name), file_name)
        for file_name in os.listdir(directory)
        if file_name.endswith(".geojson")
    ]
    with Pool(processes=processes) as pool:
        results = pool.map(process_file, files)

    filtered_list = [x for x in results if x is not None]
    
    consolidated_df = gpd.GeoDataFrame(pd.concat(filtered_list, ignore_index=True), crs=filtered_list[0].crs if filtered_list else None)
    return consolidated_df

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <path> <out> <processes>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    processes = sys.argv[3]
    pd.options.mode.copy_on_write = True
    consolidated_data = consolidate_files(input_path, int(processes))
    print("Writing ", output_path)
    consolidated_data.to_file(output_path, driver="GeoJSON")