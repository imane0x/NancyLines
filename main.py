from osm_nancy.fetch_boundary import get_nancy_boundary
from osm_nancy.fetch_streets import fetch_streets
from osm_nancy.fetch_stops import fetch_stops
from osm_nancy.build_graph import load_graph
from osm_nancy.compute_edges import compute_edges
from osm_nancy.utils import nearest_nodes_distance

def main():
    nancy_polygon = get_nancy_boundary()
    streets_gdf = fetch_streets(nancy_polygon)
    stops_gdf = fetch_stops(nancy_polygon)
    G = load_graph()
    
    edges_df = compute_edges(G, stops_gdf, streets_gdf, nearest_nodes_distance)
    edges_df.to_json("data/bus_stop_edges_streets.json", orient="records", force_ascii=False)
    print("JSON saved to data/bus_stop_edges_streets.json")

if __name__ == "__main__":
    main()
