import osmnx as ox
import networkx as nx

def nearest_nodes_distance(G, stop1, stop2):
    """Compute shortest path length along road network between two stops."""
    node1 = ox.distance.nearest_nodes(G, stop1.geometry.x, stop1.geometry.y)
    node2 = ox.distance.nearest_nodes(G, stop2.geometry.x, stop2.geometry.y)
    try:
        return nx.shortest_path_length(G, node1, node2, weight="length")
    except nx.NetworkXNoPath:
        return None
