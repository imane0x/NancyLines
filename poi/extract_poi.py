import argparse

import osmnx as ox
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


def main(args):
    tags = {
        "name": True,
    }

    pois = ox.features_from_place(args.city, tags=tags)
    pois = pois.reset_index()

    city_polygon = ox.geocode_to_gdf(args.city).iloc[0].geometry

    pois["lon"] = pois.geometry.centroid.x
    pois["lat"] = pois.geometry.centroid.y

    pois = pois[pois.apply(lambda x: city_polygon.contains(Point(x['lon'], x['lat'])), axis=1)]
    pois = pois[pois["highway"].isnull() | pois["highway"].isin([None, "bus_stop", "pedestrian", "rest_area", "steps"])]
    pois = pois[pois["railway"].isnull()] 
    pois = pois[pois["information"] != "guidepost"] 
    pois = pois[pois["waterway"].isnull()]
    pois = pois[pois["man_made"] != "pipeline"]
    pois = pois[pois["public_transport"] != "platform"]
    pois = pois[pois["power"].isnull()]
    pois = pois[~pois["amenity"].isin(["atm", "charging_station"])]

    pois = pois.dropna(axis=1, how='all')

    # [print(poi["name"]) for i, poi in pois[pois["amenity"] != "atm"].iterrows()]

    def get_type(row):
        output = ""
        for fields in [
            "amenity", #
            "shop", #
            "tourism", #
            "leisure", #
            "historic", #
            "highway", #
            # "type", 
            "landuse",
            # "information", #subtype
            # "shelter", 
            # "bench", 
            # "bus", 
            "cuisine", #
            "healthcare", # 
            # "bin", 
            # "school:FR", #subtype
            "office", #
            "building", #
            "craft", #
            "boundary",
            "emergency",
            "social_facility", 
        ]:
            if pd.notna(row[fields]) and (row[fields] not in [None, "yes", "no"]):
                output += fields + "==" + str(row[fields]) + " | "
                # output += fields
        if output != "":
            return output[:-3]
        else:
            return "other"
        
    pois["type"] = pois.apply(get_type, axis=1)

    from collections import Counter
    # name_counts = Counter({field:float(pois[field][pois["type"] == "other"].count()) for field in pois.columns})
    name_counts = Counter(pois["type"][[" | " in t and "craft" in t for t in pois["type"]]])
    for count in name_counts.most_common(1000):
        print(count)

    result = pois[["name", "lat", "lon", "type"]].copy()

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, row in result.iterrows():
            ax.text(row["lon"], row["lat"], row["name"], fontsize=5, alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title(f"Points of Interest in {args.city}")
    plt.show()
    plt.savefig("pois_map.png")

    result = result.sort_values(by=["name"])
    result.to_csv("pois.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract POIs from OpenStreetMap for a given city.")
    parser.add_argument("--city", type=str, default="Nancy, France", help="City name to extract POIs from.")
    args = parser.parse_args()
    main(args) 