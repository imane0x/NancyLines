import osmnx as ox

def load_graph():
    """Load the driving network graph for Nancy."""
    place = "Métropole du Grand Nancy, France"
    G = ox.graph_from_place(place, network_type="drive")
    print("✅ OSM road network loaded.")
    return G
