import pandas as pd
import osmnx as ox
import networkx as nx
from osm_nancy.utils import nearest_nodes_distance
def get_street_name(tags, fallback_id):
    if not isinstance(tags, dict):
        tags = {}
    name = tags.get('name')
    if name:
        return name
    id_val = tags.get('id', fallback_id)
    return f"street_{id_val}"
    
def get_stop_name(tags, fallback_id):
    """Return the stop name if available, otherwise build a fallback name."""
    if not isinstance(tags, dict):
        tags = {}
    name = tags.get('name')
    if name:
        return name
    id_val = tags.get('id', fallback_id)
    return f"stop_{id_val}"
    
def compute_edges(G, stops_gdf, streets_gdf, buffer_m=20, max_distance=500):
    """
    Compute edges between stops based on road network distance and nearby streets.

    Parameters:
    - G: networkx graph of the street network (projected in meters)
    - stops_gdf: GeoDataFrame of stops with 'geometry' and 'tags' columns
    - streets_gdf: GeoDataFrame of streets with 'geometry' and 'tags' columns
    - buffer_m: buffer around stop to consider nearby streets (meters)
    - max_distance: maximum distance (meters) to consider a stop pair connected

    Returns:
    - edges_df: DataFrame with columns ['from', 'to', 'streets', 'distance_m']
    """
    # Reproject GeoDataFrames to match graph CRS 
    G_proj = ox.project_graph(G, to_crs=2154)
    stops = stops_gdf.to_crs(2154)
    streets = streets_gdf.to_crs(2154)
    # Ensure a 'name' column exists for streets
    if 'name' not in streets.columns:
        streets['name'] = streets.apply(lambda row: get_street_name(row['tags'], row['id']), axis=1)
    if 'name' not in stops.columns:
        stops['name'] = stops.apply(lambda row: get_stop_name(row['tags'], row['id']), axis=1)
    edges_data = []

    for i, stop1 in stops.iterrows():
        for j, stop2 in stops.iterrows():
            if stop1["id"] >= stop2["id"]:
                continue
            distance_m = nearest_nodes_distance(G_proj, stop1, stop2)
            # Only include stop pairs within max_distance
            if distance_m not None and 0 < distance_m  and distance_m <= max_distance:
                streets1 = streets[streets.geometry.buffer(buffer_m).intersects(stop1.geometry)]
                streets2 = streets[streets.geometry.buffer(buffer_m).intersects(stop2.geometry)]
                street_names = set(streets1["name"]).union(set(streets2["name"]))
                edges_data.append({
                    "from": stop1["name"],
                    "to": stop2["name"],
                    "streets": list(street_names),
                    "distance_m": distance_m
                })

    edges_df = pd.DataFrame(edges_data)
    return edges_df
