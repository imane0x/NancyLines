import osmnx as ox

def get_nancy_boundary():
    """Fetch the metropolitan boundary of Nancy."""
    gdf = ox.geocode_to_gdf("Métropole du Grand Nancy, France")
    polygon = gdf.iloc[0].geometry
    print("Nancy metropolitan boundary loaded.")
    return polygon
