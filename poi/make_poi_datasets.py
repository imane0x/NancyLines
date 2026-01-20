import argparse
import random
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datasets import Dataset, concatenate_datasets
import shapely as sp
import geopandas as gpd
from shapely import wkt


def min_nodes_between(graph, id1, id2):
    if id1 == id2:
        return 0

    queue = deque([(id1, 0)])
    visited = {id1}

    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == id2:
                return dist
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


def get_target_id_at_node_distance(graph, source_id, node_distance):
    queue = deque([(source_id, 0)])
    visited = {source_id}

    while queue:
        node, dist = queue.popleft()
        if dist == node_distance:
            return node
        nodes_to_visit = graph[node]
        random.shuffle(nodes_to_visit)
        for neighbor in nodes_to_visit:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


def sample_train_pair_id(pois_dataset, distance_matrix, temperature, source_id=None):
    if source_id is None:
        source_id = random.randint(0, len(pois_dataset) - 1)

    probs = np.exp(-distance_matrix[source_id] / temperature)
    probs /= probs.sum()

    target_id = np.random.choice(len(pois_dataset), p=probs)

    return source_id, target_id


def sample_test_pair_id(pois_dataset, train_graph, node_distance):
    target_id = -1
    for source_id in random.sample(range(len(pois_dataset)), len(pois_dataset)):
        target_id = get_target_id_at_node_distance(train_graph, source_id, node_distance)
        print(source_id, target_id)
        if target_id != -1:
            return source_id, target_id
    
    raise Exception(f"No pair found for node distance {node_distance}")


def get_relation(source_geo, target_geo):
    relation = "separated"
    if (
        target_geo.geom_type in ["Polygon", "LineString"] and 
        target_geo.buffer(1e-5).contains(source_geo)
    ):
        relation = "in"
        
    if (
        source_geo.geom_type in ["Polygon", "LineString"] and 
        source_geo.buffer(1e-5).contains(target_geo)
    ):
        if relation == "in":
            relation = "in and contains"
        else:
            relation = "contains"

    if (
        (
            source_geo.geom_type == "Polygon" and 
            target_geo.geom_type == "Polygon" and 
            sp.area(sp.intersection(source_geo, target_geo)) > 0.1*max(sp.area(source_geo), sp.area(target_geo)) and 
            relation == "separated"
        )
            or
        (
            set([source_geo.geom_type, target_geo.geom_type]) in [set(["Polygon", "LineString"]), set(["LineString"])] and
            not sp.is_empty(sp.intersection(source_geo, target_geo)) and 
            relation == "separated"
        )
    ):
        relation = "intersects"

    return relation


def get_size_relation(source_geo, target_geo):
    if set([source_geo.geom_type, target_geo.geom_type]) == set(["Polygon"]):
        if sp.area(source_geo) > 1.1*sp.area(target_geo):
            return "bigger"
        elif sp.area(source_geo) < 0.9*sp.area(target_geo):
            return "smaller"
        else:
            return "same"
    return None


def build_pair(pois_dataset, distance_matrix, source_id, target_id, is_test=False):
    source = pois_dataset.iloc[source_id]
    target = pois_dataset.iloc[target_id]

    plt.plot([source["x"], target["x"]], [source["y"], target["y"]], c=("red" if is_test else "blue"), alpha=0.2, linewidth=0.1)


    return {
        "source": source["name"], 
        "source_osm_id": source["id"], 
        "source_id": source_id, 

        "target": target["name"],
        "target_osm_id": target["id"], 
        "target_id": target_id, 

        "source_loc": (float(source["x"]), float(source["y"])), 
        "target_loc": (float(target["x"]), float(target["y"])),

        "distance": distance_matrix[source_id, target_id],
        "angle": np.degrees(np.arctan2(target["y"] - source["y"], target["x"] - source["x"])),
        "size":  get_size_relation(source["geometry"], target["geometry"]),
        "relation": get_relation(source["geometry"], target["geometry"]),
    }


def get_mcqa_dataset(dataset, mcqa_type, max_distance=None, pois_dataset=None, distance_matrix=None, temperature=None):
    mcqa_dataset = []
    for i, sample in dataset.iterrows():
        if mcqa_type == "cardinal_direction" and sample["relation"] == "separated":
            propositions = [
                f'À l\'est de "{sample["source"]}"',
                f'Au nord de "{sample["source"]}"',
                f'Au sud de "{sample["source"]}"',
                f'À l\'ouest de "{sample["source"]}"',
            ]
            if -45 <= sample["angle"] < 45:
                answer = propositions[0]
            elif 45 <= sample["angle"] < 135:
                answer = propositions[1]
            elif -135 <= sample["angle"] < -45:
                answer = propositions[2]
            else:
                answer = propositions[3]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "question": f'"{sample["target"]}" se situe:',
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))
        elif mcqa_type == "cardinal_direction_numeric" and sample["relation"] == "separated":
            answer = int(90-sample["angle"]) % 360
            propositions = [answer]
            for i in range(3):
                while len(propositions) <= i+1:
                    new_proposition = int(np.random.uniform(0, 360))
                    print(new_proposition, propositions)
                    if all([new_proposition <= proposition - 45  or new_proposition >= proposition + 45 for proposition in propositions]):
                        propositions.append(new_proposition)

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "question": f'Quelle est l\'azimut de "{sample["target"]}" depuis "{sample["source"]}"?',
                "propositions": [str(p) + " degrées" for p in propositions],
                "answer": str(answer) + " degrées",
            } | dict(sample))
    
        elif mcqa_type == "proximity" and sample["relation"] == "separated":
            propositions = [
                "Très proche",
                "Proche",
                "Loin",
                "Très loin",
            ]
            if sample["distance"]*max_distance < 200:
                answer = propositions[0]
            elif sample["distance"]*max_distance < 600:
                answer = propositions[1]
            elif sample["distance"]*max_distance < 2000:
                answer = propositions[2]
            else:
                answer = propositions[3]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "question": f'Quelle est la distance entre "{sample["target"]}" et "{sample["source"]}"?',
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))
    
        elif mcqa_type == "proximity_numeric" and sample["relation"] == "separated":
            answer = int(sample["distance"]*max_distance)
            propositions = [answer]
            n_above = np.random.randint(4)
            for i in range(n_above):
                while len(propositions) <= i+1:
                    new_proposition = int(np.random.uniform(1.25, 4)*answer)
                    print(new_proposition, propositions)
                    if all([new_proposition <= 0.75*proposition or new_proposition >= 1.25*proposition for proposition in propositions]):
                        propositions.append(new_proposition)
            for i in range(3-n_above):
                while len(propositions) <= i+1+n_above:
                    new_proposition = int(np.random.uniform(0.01, 0.75)*answer)
                    print(new_proposition, propositions)
                    if all([new_proposition <= 0.75*proposition or new_proposition >= 1.25*proposition for proposition in propositions]):
                        propositions.append(new_proposition)

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "question": f'Quelle est la distance entre "{sample["target"]}" et "{sample["source"]}"?',
                "propositions": [str(p) + " mètres" for p in propositions],
                "answer": str(answer) + " mètres",
            } | dict(sample))

        elif mcqa_type == "inclusion":
            propositions = [
                f'"{sample["source"]}" se trouve dans "{sample["target"]}"',
                f'"{sample["source"]}" contient "{sample["target"]}"',
                f'"{sample["source"]}" se trouve dans "{sample["target"]}" et "{sample["source"]}" contient "{sample["target"]}"',
                "Aucune des autres réponses",
            ]
            if sample["relation"] == "in":
                answer = propositions[0]
            elif sample["relation"] == "contains":
                answer = propositions[1]
            elif sample["relation"] == "in and contains":
                answer = propositions[2]
            else:
                answer = propositions[3]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "question": "Laquelle de ces affirmations est vraie ?",
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))

        elif mcqa_type == "size" and sample["size"] is not None:
            propositions = [
                f'"{sample["source"]}" est plus grand que "{sample["target"]}"',
                f'"{sample["source"]}" est plus petit que "{sample["target"]}"',
                f'"{sample["source"]}" et "{sample["target"]}" font à peu près la même taille',
            ]
            if sample["size"] == "bigger":
                answer = propositions[0]
            elif sample["size"] == "smaller":
                answer = propositions[1]
            else:
                answer = propositions[2]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "question": "Laquelle de ces affirmations est vraie ?",
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))  

        elif mcqa_type == "closest":
            propositions = [sample["target"]]
            min_distance = sample["distance"]
            answer = propositions[-1]

            for i in range(3):
                source_id, target_id = sample_train_pair_id(pois_dataset, distance_matrix, temperature*(1.75**i), source_id=sample["source_id"])
                distance = distance_matrix[source_id, target_id]
                propositions.append(pois_dataset.iloc[target_id]["name"])
                if distance < min_distance:
                    min_distance = distance
                    answer = propositions[-1]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "question": f'Laquelle de ces propositions est la plus proche de "{sample["source"]}" ?',
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))   

    return Dataset.from_list(mcqa_dataset)


def build_graph(dataset, direction):
    edges = set([(sample["source_id"], sample["target_id"]) for k, sample in dataset.iterrows()])
    graph = defaultdict(list)

    for a, b in edges:
        if direction in ["forward", "both"]:
            graph[a].append(b)
        if direction in ["backward", "both"]:
            if direction == "both" or (b, a) not in edges:
                graph[b].append(a)

    return graph


def build_test_data(pois_dataset, distance_matrix, graph, n_test_sample, max_node_dist):
    test_dataset = []
    seen_pairs = []
    while len(test_dataset) != n_test_sample:
        for node_distance in range(1,max_node_dist+1):
            already_sampled = True
            while already_sampled:
                source_id, target_id = sample_test_pair_id(pois_dataset, graph, node_distance)
                if (source_id, target_id) not in seen_pairs:
                    already_sampled = False
                    seen_pairs.append((source_id, target_id))

                pair = build_pair(pois_dataset, distance_matrix, source_id, target_id, is_test=True)
                pair["node_distance"] = node_distance
                test_dataset.append(pair)

    return pd.DataFrame(test_dataset)


def main(args):
    pois_dataset = pd.read_csv("pois.csv")
    pois_dataset['geometry'] = pois_dataset['geometry'].apply(wkt.loads)
    pois_dataset = gpd.GeoDataFrame(pois_dataset, crs='epsg:4326')
    if args.n_pois == -1:
        args.n_pois = len(pois_dataset)
    pois_dataset = pois_dataset.sample(n=args.n_pois, random_state=0)

    length = geodesic((0, pois_dataset["lon"].max() - pois_dataset["lon"].min()), (0,0)).meters
    height = geodesic((pois_dataset["lat"].max() - pois_dataset["lat"].min(), 0), (0,0)).meters
    pois_dataset["x"] = (pois_dataset["lon"] - pois_dataset["lon"].min()) / (pois_dataset["lon"].max() - pois_dataset["lon"].min())*length
    pois_dataset["y"] = (pois_dataset["lat"] - pois_dataset["lat"].min()) / (pois_dataset["lat"].max() - pois_dataset["lat"].min())*height
    print(f"Area size: {length} m x {height} m")

    coords = pois_dataset[["x", "y"]].values
    distance_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    max_distance = distance_matrix.max()
    distance_matrix /= max_distance
    np.fill_diagonal(distance_matrix, np.inf)
    print(f"Max distance between POIs: {max_distance} m")

    train_dataset = []
    seen_pairs = []
    for _ in range(args.n_train_sample):
        already_sampled = True
        while already_sampled:
            source_id, target_id = sample_train_pair_id(pois_dataset, distance_matrix, args.temperature)
            if (source_id, target_id) not in seen_pairs:
                already_sampled = False
                seen_pairs.append((source_id, target_id))

        pair = build_pair(pois_dataset, distance_matrix, source_id, target_id)
        train_dataset.append(pair)

    train_dataset = pd.DataFrame(train_dataset)
    train_mcqa_dataset = concatenate_datasets([
        # get_mcqa_dataset(train_dataset, mcqa_type="cardinal_direction"),
        # get_mcqa_dataset(train_dataset, mcqa_type="cardinal_direction_numeric"),
        get_mcqa_dataset(train_dataset, mcqa_type="proximity", max_distance=max_distance),
        # get_mcqa_dataset(train_dataset, mcqa_type="proximity_numeric", max_distance=max_distance),
        # get_mcqa_dataset(train_dataset, mcqa_type="inclusion"),
        # get_mcqa_dataset(train_dataset, mcqa_type="closest", pois_dataset=pois_dataset, distance_matrix=distance_matrix, temperature=args.temperature),
    ])
    train_mcqa_dataset.save_to_disk("geoLLM_train_dataset")
    train_mcqa_dataset.to_json("geoLLM_train_dataset/train.jsonl", lines=True, orient="records")

    test_mcqa_dataset = concatenate_datasets([
        get_mcqa_dataset( # Symmetrie / Reciprocité
            build_test_data(
                pois_dataset, distance_matrix, 
                build_graph(train_dataset[train_dataset["relation"] == "separated"], direction="backward"), 
                n_test_sample=10, max_node_dist=1
            ), 
            mcqa_type="proximity", max_distance=max_distance,
         ),
    ])
    test_mcqa_dataset.save_to_disk("geoLLM_test_dataset")
    test_mcqa_dataset.to_json("geoLLM_test_dataset/test.jsonl", lines=True, orient="records")

    plt.savefig("city_graph.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make pois datasets for training/testing.")
    parser.add_argument("--n_pois", type=int, default=-1, help="Number of pois to use (-1 to use every pois).")
    parser.add_argument("--n_train_sample", type=int, default=10000, help="Number of samples to generate for train.")
    parser.add_argument("--n_test_sample", type=int, default=2000, help="Number of samples to generate for test.")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for sampling.")
    parser.add_argument("--max_node_dist", type=int, default=1, help="Maximum node distance to evaluate.")

    args = parser.parse_args()
    main(args) 