import requests
import geopandas as gpd
from shapely.geometry import LineString

def fetch_streets(nancy_polygon):
    """Fetch and filter all OSM bus route streets in the Nancy metropolitan area."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:180];
    area["name"="Métropole du Grand Nancy"]->.searchArea;
    relation["route"="bus"](area.searchArea);
    >;
    out geom;
    """

    response = requests.get(overpass_url, params={"data": overpass_query})
    data = response.json()

    streets = []
    for el in data["elements"]:
        if el["type"] == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            streets.append({
                "id": el["id"],
                "geometry": LineString(coords),
                "tags": el.get("tags", {})
            })

    gdf = gpd.GeoDataFrame(streets, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.intersects(nancy_polygon)]
    print(f" {len(gdf)} streets inside Nancy boundary.")
    return gdf
