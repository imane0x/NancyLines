import argparse
import requests

import osmnx as ox


def main(args):
    left, bottom, right, top = ox.geocode_to_gdf(args.city).iloc[0].geometry.bounds

    empty = False
    page_num = 0
    with open("traces.gpx", 'w') as f:
        while not empty:
            gpx = str(requests.get(f"https://api.openstreetmap.org/api/0.6/trackpoints?bbox={left},{bottom},{right},{top}&page={page_num}").content)

            print(f"Reading page {page_num}, len:{len(gpx)}")

            splits = gpx.split('\\n')
            for index, item in enumerate(splits):
                if index == 0:
                    f.write(str(item)[2:] + '\n')
                else:
                    if len(item) > 1:
                        f.write(str(item) + '\n')

            if "<trk>" not in gpx:
                empty = True

            page_num += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GPS Traces from OpenStreetMap for a given city.")
    parser.add_argument("--city", type=str, default="Nancy, France", help="City name to extract GPS Traces from.")
    args = parser.parse_args()
    main(args) 