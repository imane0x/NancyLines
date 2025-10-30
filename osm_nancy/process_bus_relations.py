import pandas as pd
from typing import List, Dict, Any

def get_member_stop_info(member: dict, stop_df: pd.DataFrame) -> dict:
    """Process a stop member and return updated member with coordinates, name, and streets."""
    stop_info = stop_df[stop_df['id'] == member['ref']]

    if stop_info.empty:
      return None

    stop_coordinates = tuple(stop_info['geometry'].values[0])
    stop_name = stop_info['name'].values[0]
    streets_info = stop_info['streets'].values[0]

    member.update({
        'coordinates': list(stop_coordinates),
        'name': stop_name,
        'streets': streets_info
    })
    return member


def get_member_way_info(member: dict, streets_df: pd.DataFrame) -> dict:
    """Process a way member and return updated member with coordinates and name."""
    ways_info = streets_df[streets_df['id'] == member['ref']]
    if not ways_info.empty:
        ways_coordinates = list(ways_info['geometry'].iloc[0].coords)
        way_name = ways_info['name'].iloc[0]
    else:
        ways_coordinates = None
        way_name = f"street_{member['ref']}"

    member.update({
        'coordinates': ways_coordinates,
        'name': way_name
    })
    return member


def process_bus_relation(bus_relations: dict, stop_list: pd.DataFrame, streets: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert raw bus_relations into structured bus data with stops and ways."""
    all_bus_data = []

    for element in bus_relations.get('elements', []):
        line_name = element.get('tags', {}).get('name')
        print(line_name)
        if not line_name:
            continue

        bus_data = {
            "line": line_name,
            "stops": [],
            "ways": []
        }

        for member in element.get('members', []):
            if member.get('role') == 'stop':
                updated_member = get_member_stop_info(member, stop_list)
                if updated_member:
                    bus_data['stops'].append(updated_member)

            elif member.get('type') == 'way' and member.get('role') != 'platform':
                updated_member = get_member_way_info(member, streets)
                bus_data['ways'].append(updated_member)

        all_bus_data.append(bus_data)

    return all_bus_data
