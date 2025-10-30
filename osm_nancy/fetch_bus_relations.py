def fetch_bus_relations(nancy_polygon):
    """Fetch all bus route relations in Nancy metropolitan area."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:120];
    area["name"="Métropole du Grand Nancy"]->.searchArea;
    (
      relation["type"="route"]["route"="bus"](area.searchArea);
    );
    out body;
    >;
    out skel qt;
    """

    response = requests.get(overpass_url, params={"data": overpass_query})
    response.raise_for_status()
    data = response.json()
    return data
