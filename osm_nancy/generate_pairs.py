from typing import List, Dict, Any


def get_street_lists(stop: dict) -> List[str]:
    """Convert stop['streets'] to a list of street names."""
    streets = stop.get('streets')
    if not streets:
        return []
    if isinstance(streets, list):
        return streets
    return [streets]


def get_ways_between_stops(stop1: dict, stop2: dict, street_names: List[str]) -> List[str]:
    """
    Return the ordered list of streets between two stops.
    Args:
        stop1, stop2: stop dictionaries containing 'streets'.
        street_names: list of street names along the bus line.

    Returns:
        List of street names between stop1 and stop2.
    """
    street1_list = get_street_lists(stop1)
    street2_list = get_street_lists(stop2)
    # Find first matching street for each stop
    s1 = next((s for s in street1_list if s in street_names), None)
    s2 = next((s for s in street2_list if s in street_names), None)
    if s1 is None or s2 is None:
        return []

    idx1 = street_names.index(s1)
    idx2 = street_names.index(s2)
    # Ensure correct order
    if idx1 <= idx2:
        ways_slice = street_names[idx1:idx2 + 1]
    else:
        ways_slice = street_names[idx2:idx1 + 1]

    # Remove duplicates while preserving order
    seen = set()
    unique_ways = [w for w in ways_slice if (w not in seen and not seen.add(w))]
    return unique_ways


def generate_stop_pairs(all_bus_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate a list of stop-to-stop pairs with streets and line info."""
    pairs_data = []

    for element in all_bus_data:
        stops = element.get('stops', [])
        streets_list = [w['name'] for w in element.get('ways', [])]
        for stop1, stop2 in zip(stops[:-1], stops[1:]):
            pair_info = {
                "from": stop1.get('name'),
                "to": stop2.get('name'),
                "distance": None,
                "ways": get_ways_between_stops(stop1, stop2, streets_list),
                "line": element.get('line')
            }
            pairs_data.append(pair_info)

    return pairs_data
