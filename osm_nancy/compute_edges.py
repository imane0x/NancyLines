import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString

def nearest_nodes_distance(G, stop1_geom, stop2_geom):
    """Compute shortest path length along road network between two stops."""
    node1 = ox.distance.nearest_nodes(G, stop1_geom.x, stop1_geom.y)
    node2 = ox.distance.nearest_nodes(G, stop2_geom.x, stop2_geom.y)
    try:
        return nx.shortest_path_length(G, node1, node2, weight="length")
    except nx.NetworkXNoPath:
        return None

def get_street_name(tags, fallback_id):
    if not isinstance(tags, dict):
        tags = {}
    name = tags.get('name')
    if name:
        return name
    id_val = tags.get('id', fallback_id)
    return f"street_{id_val}"

def get_stop_name(tags, fallback_id):
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
    """
    # Reproject to metric CRS
    G_proj = ox.project_graph(G, to_crs=2154)
    stops = stops_gdf.to_crs(2154)
    streets = streets_gdf.to_crs(2154)

    # Ensure a 'name' column exists
    if 'name' not in streets.columns:
        streets['name'] = streets.apply(lambda row: get_street_name(row['tags'], row['id']), axis=1)
    if 'name' not in stops.columns:
        stops['name'] = stops.apply(lambda row: get_stop_name(row['tags'], row['id']), axis=1)

    # Compute nearest nodes once
    stops['nearest_node'] = stops.geometry.apply(lambda geom: ox.distance.nearest_nodes(G_proj, geom.x, geom.y))

    edges_data = []

    for stop1 in stops.iterrows():
        for stop2 in stops.iterrows():
            if stop1["id"] >= stop2["id"]:
                continue

            distance_m = nearest_nodes_distance(G_proj, stop1.geometry, stop2.geometry)
            if distance_m is None or distance_m <= 0 or distance_m > max_distance:
                continue

            route = nx.shortest_path(G_proj, stop1['nearest_node'], stop2['nearest_node'], weight="length")

            street_names = []
            for u, v in zip(route[:-1], route[1:]):
                edge_data = G_proj.get_edge_data(u, v)
                if edge_data:
                    first_edge = list(edge_data.values())[0]  # take first parallel edge
                    name = first_edge.get('name', f"street_{u}_{v}")
                    if isinstance(name, list):
                        street_names.extend(name)
                    else:
                        street_names.append(name)
            street_names = list(set(street_names))  # unique

            edges_data.append({
                "from": stop1["name"],
                "to": stop2["name"],
                "streets": street_names,
                "distance_m": distance_m
            })

    return pd.DataFrame(edges_data)
