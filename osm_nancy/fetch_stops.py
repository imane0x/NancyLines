import requests
import geopandas as gpd
from shapely.geometry import Point

def fetch_stops(nancy_polygon):
    """Fetch and filter all bus stops in Nancy metropolitan area."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:120];
    area["name"="Métropole du Grand Nancy"]->.searchArea;
    (
      node["highway"="bus_stop"](area.searchArea);
      node["public_transport"="platform"]["bus"~"."](area.searchArea);
    );
    out body;
    """

    response = requests.get(overpass_url, params={"data": overpass_query})
    data = response.json()

    stops = []
    for el in data["elements"]:
        if el["type"] == "node":
            stops.append({
                "id": el["id"],
                "geometry": Point(el["lon"], el["lat"]),
                "tags": el.get("tags", {})
            })

    gdf = gpd.GeoDataFrame(stops, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.intersects(nancy_polygon)]
    print(f"✅ {len(gdf)} bus stops inside Nancy boundary.")
    return gdf
