from osm_nancy.fetch_boundary import get_nancy_boundary
from osm_nancy.fetch_streets import fetch_streets
from osm_nancy.fetch_stops import fetch_stops
from osm_nancy.build_graph import load_graph
from osm_nancy.compute_edges import compute_edges
from osm_nancy.utils import nearest_nodes_distance
from osm_nancy.process_bus_relations import process_bus_relation
from osm_nancy.generate_pairs import generate_stop_pairs

def main():
    polygon = get_nancy_boundary()
    graph = load_graph()
    bus_relations = fetch_bus_relations(polygon)
    streets =fetch_streets(polygon)
    stops = get_stops_with_streets(polygon, bus_relations)
    all_bus_data = process_bus_relation(bus_relations, stops, streets)
    pairs_data = generate_stop_pairs(all_bus_data)


if __name__ == "__main__":
    main()
