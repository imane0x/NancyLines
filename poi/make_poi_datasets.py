import argparse
import random
from collections import defaultdict, deque
from itertools import combinations, product

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datasets import Dataset, concatenate_datasets
import shapely as sp
import geopandas as gpd
from shapely import wkt


def get_target_id_at_node_distance(graph, source_id, node_distance, n_node=1, return_visited=False):
    queue = deque([(source_id, 0, True)])
    visited = {source_id}
    output_nodes = []

    while queue:
        node, dist, is_parent_unique = queue.popleft()
        if dist == node_distance and is_parent_unique:
            if n_node == 1:
                return node
            else:
                output_nodes.append(node)
            if len(output_nodes) == n_node:
                return output_nodes
        elif dist > node_distance:
            if n_node == -1:
                return output_nodes
            else:
                return -1
        if type(graph) == type([]):
            if dist == node_distance:
                return -1
            else:
                nodes_to_visit = graph[dist][node]
        else:
            nodes_to_visit = graph[node]
        random.shuffle(nodes_to_visit)
        for neighbor, is_curr_unique in nodes_to_visit:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1, is_parent_unique and is_curr_unique))
    if return_visited:
        return visited
    return -1


def sample_train_pair_id(pois_dataset, distance_matrix, temperature, source_id=None):
    if source_id is None:
        source_id = random.randint(0, len(pois_dataset) - 1)

    probs = np.exp(-distance_matrix[source_id] / temperature)
    probs /= probs.sum()

    # plt.scatter(pois_dataset["x"], pois_dataset["y"], c=probs)
    # plt.colorbar()
    # plt.show()
    # plt.savefig("heatmap.png")

    target_id = np.random.choice(len(pois_dataset), p=probs)

    return source_id, target_id


def sample_test_pair_id(pois_dataset, train_graph, node_distance, n_node=1, return_visited=False):
    target_id = -1
    for source_id in random.sample(range(len(pois_dataset)), len(pois_dataset)):
        if return_visited:
            visited = get_target_id_at_node_distance(train_graph, source_id, node_distance, n_node=n_node, return_visited=return_visited)
            target_ids = list(
                set(
                    [k for k,v in train_graph.items() if any([is_unique for target_id, is_unique in v])] +
                    [target_id for k,v in train_graph.items() for target_id, is_unique in v if is_unique]
                )
                -
                set(
                    visited
                )
            )
            if len(target_ids) != 0:
                return source_id, random.choice(target_ids)
        else:
            target_id = get_target_id_at_node_distance(train_graph, source_id, node_distance, n_node=n_node)
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

    plt.plot([source["x"], target["x"]], [source["y"], target["y"]], c=("red" if is_test else "blue"), alpha=(0.3 if is_test else 0.2), linewidth=(0.15 if is_test else 0.1))

    return {
        "source": source["name"], 
        "source_osm_id": source["id"], 
        "source_id": source_id, 

        "target": target["name"],
        "target_osm_id": target["id"], 
        "target_id": target_id, 

        "is_unique": source["is_unique"] and target["is_unique"],

        "source_loc": (float(source["x"]), float(source["y"])), 
        "target_loc": (float(target["x"]), float(target["y"])),

        "distance": distance_matrix[source_id, target_id],
        "angle": np.degrees(np.arctan2(target["y"] - source["y"], target["x"] - source["x"])),
        "size":  get_size_relation(source["geometry"], target["geometry"]),
        "relation": get_relation(source["geometry"], target["geometry"]),
    }


def build_graph(dataset, direction):
    edges = set(zip(dataset["source_id"], dataset["target_id"], dataset["is_unique"]))
    graph = defaultdict(list)

    for a, b, is_unique in edges:
        if direction in ["forward", "both", "forward_only"]:
            if direction != "forward_only" or (b, a) not in edges:
                graph[a].append((b, is_unique))
        if direction in ["backward", "both"]:
            if direction == "both" or (b, a) not in edges:
                graph[b].append((a, is_unique))

    return graph


def build_test_data_depth(pois_dataset, distance_matrix, graph, n_test_sample, node_dist, return_visited=False):
    test_dataset = []
    seen_pairs = []
    while len(test_dataset) < n_test_sample:
        already_sampled = True
        while already_sampled:
            source_id, target_id = sample_test_pair_id(pois_dataset, graph, node_dist, return_visited=return_visited)
            if (source_id, target_id) not in seen_pairs:
                already_sampled = False
                seen_pairs.append((source_id, target_id))
                pair = build_pair(pois_dataset, distance_matrix, source_id, target_id, is_test=True)
                pair["node_distance"] = node_dist
                test_dataset.append(pair)

    return pd.DataFrame(test_dataset)


def build_test_data_width(pois_dataset, distance_matrix, graph, n_test_sample, width):
    test_dataset = []
    seen_pairs = []
    while len(test_dataset) < n_test_sample:
        already_sampled = True
        while already_sampled:
            source_id, target_ids = sample_test_pair_id(pois_dataset, graph, node_distance=1, n_node=width)
            if (source_id, tuple(target_ids)) not in seen_pairs:
                already_sampled = False
                seen_pairs.append((source_id, tuple(target_ids)))
                sample = {}
                for i, target_id in enumerate(target_ids):
                    pair = build_pair(pois_dataset, distance_matrix, source_id, target_id, is_test=True)
                    sample.update({(f"{k}_pair_{i}"):v for k,v in pair.items()})
                test_dataset.append(sample)

    return pd.DataFrame(test_dataset)


def build_test_data_cayley_menger(pois_dataset, distance_matrix, graph, n_test_sample):
    non_connected_dataset = build_test_data_depth(pois_dataset, distance_matrix, graph, n_test_sample=1000, node_dist=2)
    test_dataset = []
    seen_pairs = []
    for i, sample in non_connected_dataset.iterrows():
        a_neighbors = get_target_id_at_node_distance(graph, sample["source_id"], node_distance=1, n_node=-1)
        b_neighbors = get_target_id_at_node_distance(graph, sample["target_id"], node_distance=1, n_node=-1)
        if a_neighbors != -1 and b_neighbors != -1:
            for neighbor1, neighbor2 in combinations(list(set(a_neighbors) & set(b_neighbors)), 2):
                if (
                    (neighbor1 in [target_id for target_id, is_unique in graph[neighbor2]] or neighbor2 in [target_id for target_id, is_unique in graph[neighbor1]]) and 
                    (sample["source_id"], sample["target_id"]) not in seen_pairs
                ):
                    test_dataset.append(sample)
                    seen_pairs.append((sample["source_id"], sample["target_id"]))
                    if len(test_dataset) == n_test_sample:
                        return pd.DataFrame(test_dataset)

    raise Exception("No Cayley-Menger configuration")


def angle_to_cardinality(angle):
        if -45 <= angle < 45:
            return "est"
        elif 45 <= angle < 135:
            return "nord"
        elif -135 <= angle < -45:
            return "sud"
        else:
            return "ouest"


def get_mcqa_dataset(dataset, mcqa_type, generalization_type=None, max_distance=None, negation=None, uncertainty=None):
    print(f"Started task {mcqa_type} {generalization_type}")
    mcqa_dataset = []
    for i, sample in dataset.iterrows():
        if mcqa_type == "cardinality" and sample["relation"] == "separated":
            if negation == "double":
                propositions = [
                    f'"{sample["target"]}" ne se situe pas à l\'est de "{sample["source"]}"',
                    f'"{sample["target"]}" ne se situe pas au nord de "{sample["source"]}"',
                    f'"{sample["target"]}" ne se situe pas au sud de "{sample["source"]}"',
                    f'"{sample["target"]}" ne se situe pas à l\'ouest de "{sample["source"]}"',
                ]
            else:
                propositions = [
                    f'À l\'est de "{sample["source"]}"',
                    f'Au nord de "{sample["source"]}"',
                    f'Au sud de "{sample["source"]}"',
                    f'À l\'ouest de "{sample["source"]}"',
                ]
            cardinality = angle_to_cardinality(sample["angle"])
            if cardinality == "est":
                answer = propositions[0]
            elif cardinality == "nord":
                answer = propositions[1]
            elif cardinality == "sud":
                answer = propositions[2]
            else:
                answer = propositions[3]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": "Laquelle de ces affirmations est fausse ?" if negation == "double" else f'"{sample["target"]}" se situe:',
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))
        
        elif mcqa_type == "cardinality_numeric" and sample["relation"] == "separated":
            if negation == "anti-symmetry":
                angle = round(90-sample["angle"]) % 360
                symmetric_angle = (angle+180) % 360
                question = f'Est-ce que "{sample["target"]}" est à {symmetric_angle} degrés d\'azimut de "{sample["source"]} ?"'
                propositions = [
                    "Oui",
                    "Non",
                ]
                answer = propositions[1]
            else:
                question = f'Quelle est l\'azimut de "{sample["target"]}" depuis "{sample["source"]}"?'
                answer = round(90-sample["angle"]) % 360
                propositions = [answer]
                for i in range(3):
                    while len(propositions) <= i+1:
                        new_proposition = round(np.random.uniform(0, 360))
                        if all([new_proposition <= proposition - 45  or new_proposition >= proposition + 45 for proposition in propositions]):
                            propositions.append(new_proposition)
                propositions = [str(p) + " degrées" for p in propositions]
                answer = str(answer) + " degrées"

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": question,
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))
    
        elif mcqa_type == "proximity" and sample["relation"] == "separated":
            if negation == "disjonction":
                if sample["distance"]*max_distance < 200:
                    fake_answer = "loin"
                elif sample["distance"]*max_distance < 500:
                    fake_answer = "loin"
                elif sample["distance"]*max_distance < 2000:
                    fake_answer = "proche"
                else:
                    fake_answer = "proche"
                question = f'"{sample["target"]}" est {fake_answer} de "{sample["source"]} ?"'
                propositions = [
                    "Vrai",
                    "Faux",
                ]
                answer = propositions[1]

            else:
                if negation == "double":
                    question = "Laquelle de ces affirmations est fausse ?"
                    propositions = [
                        f'"{sample["target"]}" n\'est pas très proche de "{sample["source"]}"',
                        f'"{sample["target"]}" n\'est pas proche de "{sample["source"]}"',
                        f'"{sample["target"]}" n\'est pas loin de "{sample["source"]}"',
                        f'"{sample["target"]}" n\'est pas très loin de "{sample["source"]}"',
                    ]
                else:
                    question = f'À quelle distance se trouve "{sample["target"]}" par rapport à "{sample["source"]}"?'
                    propositions = [
                        "Très proche",
                        "Proche",
                        "Loin",
                        "Très loin",
                    ]
                if sample["distance"]*max_distance < 200:
                    answer = propositions[0]
                elif sample["distance"]*max_distance < 500:
                    answer = propositions[1]
                elif sample["distance"]*max_distance < 2000:
                    answer = propositions[2]
                else:
                    answer = propositions[3]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": question,
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))
    
        elif mcqa_type == "proximity_numeric" and sample["relation"] == "separated":
            answer = round(sample["distance"]*max_distance)
            propositions = [answer]
            n_above = np.random.randint(4)
            for i in range(n_above):
                while len(propositions) <= i+1:
                    new_proposition = round(np.random.uniform(1.25, 4)*answer)
                    if all([new_proposition <= 0.75*proposition or new_proposition >= 1.25*proposition for proposition in propositions]):
                        propositions.append(new_proposition)
            for i in range(3-n_above):
                while len(propositions) <= i+1+n_above:
                    new_proposition = round(np.random.uniform(0.01, 0.75)*answer)
                    if all([new_proposition <= 0.75*proposition or new_proposition >= 1.25*proposition for proposition in propositions]):
                        propositions.append(new_proposition)

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": f'À quelle distance se trouve "{sample["target"]}" par rapport à "{sample["source"]}"?',
                "propositions": [str(p) + " mètres" for p in propositions],
                "answer": str(answer) + " mètres",
            } | dict(sample))

        elif mcqa_type == "inclusion" and (negation != "double" or sample["relation"] != "in and contains"):
            if negation == "double":
                propositions = [
                    f'"{sample["source"]}" ne se trouve pas dans "{sample["target"]}"',
                    f'"{sample["source"]}" ne contient pas "{sample["target"]}"',
                    f'"{sample["source"]}" contient ou se trouve dans "{sample["target"]}"',
                ]
            else:
                propositions = [
                    f'"{sample["source"]}" se trouve dans "{sample["target"]}"',
                    f'"{sample["source"]}" contient "{sample["target"]}"',
                    f'"{sample["source"]}" contient et se trouve dans "{sample["target"]}"',
                    "Aucune des autres réponses",
                ]
            if sample["relation"] == "in":
                answer = propositions[0]
            elif sample["relation"] == "contains":
                answer = propositions[1]
            elif sample["relation"] == "in and contains" or negation == "double":
                answer = propositions[2]
            else:
                answer = propositions[3]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": "Laquelle de ces affirmations est fausse ?" if negation == "double" else "Laquelle de ces affirmations est vraie ?",
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))

        elif mcqa_type == "size" and sample["size"] is not None and (negation != "disjonction" or sample["size"] in ["bigger", "smaller"]):
            if negation == "disjonction":
                question = f'"{sample["source"]}" est plus {"petit" if sample["size"] == "bigger" else "grand"} que "{sample["target"]} ?"'
                propositions = [
                    "Vrai",
                    "Faux",
                ]
                answer = propositions[1]
            elif negation == "anti-symmetry":
                question = f'Est-ce que "{sample["source"]}" est plus {"petit" if sample["size"] == "bigger" else "grand"} que "{sample["target"]} ?"'
                propositions = [
                    "Oui",
                    "Non",
                ]
                answer = propositions[1]
            else:
                question = f'"{sample["source"]}" est plus grand, plus petit ou bien de la même taille que "{sample["target"]}"'
                propositions = [
                    "Plus grand",
                    "Plus petit",
                    "De la même taille",
                ]
                if sample["size"] == "bigger":
                    answer = propositions[0]
                elif sample["size"] == "smaller":
                    answer = propositions[1]
                else:
                    answer = propositions[2]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": question,
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))  

        elif mcqa_type.startswith("distance"):
            propositions = [sample[f"target_pair_{i}"] for i in range(3)]
            if mcqa_type == "distance_closest":
                answer = propositions[np.argmin([sample[f"distance_pair_{i}"] for i in range(3)])]
            else:
                answer = propositions[np.argmax([sample[f"distance_pair_{i}"] for i in range(3)])]

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": f'Laquelle de ces propositions est la plus {"proche" if mcqa_type == "distance_closest" else "loin"} de "{sample["source_pair_0"]}" ?',
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))

        elif mcqa_type.startswith("direction"):
            propositions = [sample[f"target_pair_{i}"] for i in range(3)]
            if mcqa_type == "direction_eastern":
                answer = propositions[np.argmin([sample[f"angle_pair_{i}"] for i in range(3)])]
            elif mcqa_type == "direction_western":
                answer = propositions[np.argmax([abs(sample[f"angle_pair_{i}"]) for i in range(3)])]
            else:
                raise Exception(f"{mcqa_type} not supported")

            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": f'Laquelle de ces propositions est la plus à l\'{"est" if mcqa_type == "direction_eastern" else "ouest"} de "{sample["source_pair_0"]}" ?',
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))
        
        elif mcqa_type == "unicity_proximity":
            propositions = [
                "Oui",
                "Non",
            ]
            answer = propositions[1]
            mcqa_dataset.append({
                "mcqa_type": mcqa_type,
                "generalization_type": generalization_type,
                "question": f'Est-ce que "{sample["source"]}" est le plus proche de "{sample["target"]}"?',
                "propositions": propositions,
                "answer": answer,
            } | dict(sample))

    if uncertainty is not None:
        for sample in mcqa_dataset:
            proposition = "Je ne sais pas"
            sample["propositions"].append(proposition)
            if uncertainty == "yes":
                sample["answer"] = proposition

    for sample in mcqa_dataset:
        propositions = sample["propositions"]
        random.shuffle(propositions)
        sample["propositions"] = {chr(ord('A') + i):proposition for i, proposition in enumerate(propositions)}
        sample["answer_letter"] = [i for i, proposition in sample["propositions"].items() if proposition == sample["answer"]][0]

    print(f"Finished task {mcqa_type} {generalization_type}")
    return Dataset.from_list(mcqa_dataset)


def main(args):
    random.seed(0)
    np.random.seed(0)

    pois_dataset = pd.read_csv("pois.csv")
    pois_dataset['geometry'] = pois_dataset['geometry'].apply(wkt.loads)
    pois_dataset = gpd.GeoDataFrame(pois_dataset, crs='epsg:4326')
    if args.n_pois == -1:
        args.n_pois = len(pois_dataset)
    pois_dataset = pois_dataset.sample(n=args.n_pois, random_state=0)

    length = geodesic((pois_dataset["lat"].mean(), pois_dataset["lon"].min()), (pois_dataset["lat"].mean(), pois_dataset["lon"].max())).meters,
    height = geodesic((pois_dataset["lat"].min(), pois_dataset["lon"].mean()), (pois_dataset["lat"].max(), pois_dataset["lon"].mean())).meters,

    pois_dataset["x"] = (pois_dataset["lon"] - pois_dataset["lon"].min()) / (pois_dataset["lon"].max() - pois_dataset["lon"].min())*length
    pois_dataset["y"] = (pois_dataset["lat"] - pois_dataset["lat"].min()) / (pois_dataset["lat"].max() - pois_dataset["lat"].min())*height
    print(f"Area size: {length} m x {height} m")

    coords = pois_dataset[["x", "y"]].values
    distance_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    max_distance = distance_matrix.max()
    distance_matrix /= max_distance
    np.fill_diagonal(distance_matrix, np.inf)
    print(f"Max distance between POIs: {max_distance} m")

    print([sample for i,sample in pois_dataset.iterrows() if sample["x"] == min(list(pois_dataset["x"]))])
    print([sample for i,sample in pois_dataset.iterrows() if sample["x"] == max(list(pois_dataset["x"]))])
    print([sample for i,sample in pois_dataset.iterrows() if sample["y"] == min(list(pois_dataset["y"]))])
    print([sample for i,sample in pois_dataset.iterrows() if sample["y"] == max(list(pois_dataset["y"]))])

    train_dataset = []
    seen_pairs = []

    while len(train_dataset) < args.n_train_sample:
        for temperature in args.temperatures:
            already_sampled = True
            while already_sampled:
                source_id, target_id = sample_train_pair_id(pois_dataset, distance_matrix, temperature)
                if (source_id, target_id) not in seen_pairs:
                    already_sampled = False
                    seen_pairs.append((source_id, target_id))
                    pair = build_pair(pois_dataset, distance_matrix, source_id, target_id)
                    train_dataset.append(pair)

    train_dataset = pd.DataFrame(train_dataset)
    train_mcqa_dataset = concatenate_datasets([
        get_mcqa_dataset(train_dataset, mcqa_type="cardinality"),
        get_mcqa_dataset(train_dataset, mcqa_type="cardinality_numeric"),
        get_mcqa_dataset(train_dataset, mcqa_type="proximity", max_distance=max_distance),
        get_mcqa_dataset(train_dataset, mcqa_type="proximity_numeric", max_distance=max_distance),
        get_mcqa_dataset(train_dataset, mcqa_type="inclusion"),
    ])
    train_mcqa_dataset.save_to_disk("geoLLM_train_dataset")
    train_mcqa_dataset.to_json("geoLLM_train_dataset/train.jsonl", lines=True, orient="records")
   
    test_mcqa_dataset = concatenate_datasets(   
        [
            # Symmetrie / Reciprocité
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset[mask], direction="backward"), 
                n_test_sample=10, node_dist=1), mcqa_type=mcqa_type, generalization_type="Symmetry/Reciprocity", max_distance=max_distance
            )
            for mask, mcqa_type in [
                (train_dataset["relation"] == "separated", "proximity"),
                (train_dataset["relation"] == "separated", "proximity_numeric"),
                (train_dataset["relation"] == "separated", "cardinality"),
                (train_dataset["relation"] == "separated", "cardinality_numeric"),
                (train_dataset["relation"].isin(["in", "contains"]), "inclusion"),
                (train_dataset["size"].isin(["bigger", "smaller"]), "size"),
            ]
        ] +
        [
            # Transitivité
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset[mask], direction=direction), 
                n_test_sample=10, node_dist=node_dist), mcqa_type=mcqa_type, generalization_type=f"Transitivity x{node_dist-1} {direction}"
            )
            for (mask, mcqa_type), node_dist, direction in list(product([
                ((train_dataset["relation"] == "separated") & (train_dataset["angle"].map(lambda x: angle_to_cardinality(x) == random.choice(["nord", "sud", "est", "ouest"]))), "cardinality"),
                (train_dataset["relation"].isin(["in", "contains"]), "inclusion"),
                (train_dataset["size"].isin(["bigger", "smaller"]), "size"),
            ],[2,4],["forward", "backward"]))
        ] +
        [
            # Geometrie Euclidienne 
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset[train_dataset["relation"] == "separated"], direction=direction), 
                n_test_sample=10, node_dist=2), mcqa_type=mcqa_type, generalization_type=f"Euclidian Geometry {direction}", max_distance=max_distance
            )
            for mcqa_type, direction in list(product(["proximity_numeric", "cardinality_numeric"],["forward", "backward"]))
        ] +
        [
            # Width
            get_mcqa_dataset(build_test_data_width(pois_dataset, distance_matrix,
                build_graph(train_dataset[mask], direction=direction), 
                n_test_sample=10, width=5), mcqa_type=mcqa_type, generalization_type="Width"
            )
            for mask, mcqa_type, direction in [
                (train_dataset["relation"] == "separated", "distance_closest", "forward"),
                (train_dataset["relation"] == "separated", "distance_furtherest", "backward"),
                (train_dataset["relation"] == "separated", "direction_eastern", "forward"),
                (train_dataset["relation"] == "separated", "direction_western", "backward"),
            ]
        ] +
        [
            # Double negation
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset[mask], direction=direction), 
                n_test_sample=10, node_dist=1), mcqa_type=mcqa_type, generalization_type=f"Double negation {direction}", negation="double", max_distance=max_distance
            )
            for (mask, mcqa_type), direction in list(product([
                (train_dataset["relation"] == "separated", "cardinality"),
                (train_dataset["relation"] == "separated", "proximity"),
            ], ["forward", "backward"]))
        ] +
        [
            # Monotonie stricte (needs to delete the test sample from the train)
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset[mask], direction=direction), 
                n_test_sample=10, node_dist=1), mcqa_type="size", generalization_type=f"Strict Monotony {direction}"
            )
            for mask, direction in list(product([
                train_dataset["relation"] == "in",
                train_dataset["relation"] == "contains",
            ], ["forward", "backward"]))
        ] +
        [
            # Incertitude 
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix, 
                build_graph(train_dataset[mask], direction="both"), 
                n_test_sample=10, node_dist=1+10000000*(uncertainty=="yes"), return_visited=(uncertainty=="yes")), mcqa_type=mcqa_type, generalization_type=f"Uncertainty {uncertainty}", uncertainty=uncertainty, max_distance=max_distance
            )
            for (mask, mcqa_type), uncertainty in list(product([
                (train_dataset["relation"].isin(["in","contains"]), "inclusion"),
                (train_dataset["relation"] == "separated", "proximity"),
            ], ["yes", "no"]))
        ] +
        [
            # Disjonction 
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset[mask], direction="forward"), 
                n_test_sample=10, node_dist=1), mcqa_type=mcqa_type, generalization_type="Disjunction", negation="disjonction", max_distance=max_distance
            )
            for mask, mcqa_type in [
                (train_dataset["size"].isin(["bigger", "smaller"]), "size"),
                (train_dataset["relation"] == "separated", "proximity"),
            ]
        ] +
        [
            # Anti-symmetrie 
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset[mask], direction="backward"), 
                n_test_sample=10, node_dist=1), mcqa_type=mcqa_type, generalization_type="Anti-symmetry", negation="anti-symmetry",
            )
            for mask, mcqa_type in [
                (train_dataset["size"].isin(["bigger", "smaller"]), "size"),
                (train_dataset["relation"] == "separated", "cardinality_numeric"),
            ]
        ] +
        [
            # Unicity (needs to add uniquely near in train and check test is not the nearest)
            get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,
                build_graph(train_dataset, direction="both"), 
                n_test_sample=10, node_dist=1), mcqa_type=mcqa_type, generalization_type="Unicity"
            )
            for mcqa_type in ["unicity_proximity"]
        ] +
        # [
        #     # Transitivité inter features
        #     get_mcqa_dataset(build_test_data_depth(pois_dataset, distance_matrix,[
        #             build_graph(train_dataset[train_dataset["relation"] == "in"], direction="forward"),
        #             build_graph(train_dataset[train_dataset["relation"] == "separated"], direction="forward"),
        #         ], n_test_sample=10, node_dist=2), mcqa_type="cardinality",
        #     )
        # ] +
        [
            # Cayley-Menger determinant (il faut remove les cardinality de ces points)
            get_mcqa_dataset(build_test_data_cayley_menger(pois_dataset, distance_matrix,
                build_graph(train_dataset[train_dataset["relation"] == "separated"], direction=direction),
                n_test_sample=10), mcqa_type="proximity_numeric", generalization_type=f"Cayley-Menger determinant {direction}", max_distance=max_distance,
            )
            for direction in ["forward", "backward", "both", "both"]
        ]
    )#.shuffle(seed=0)

    test_mcqa_dataset.save_to_disk("geoLLM_test_dataset")
    test_mcqa_dataset.to_json("geoLLM_test_dataset/test.jsonl", lines=True, orient="records")

    plt.savefig("city_graph.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make pois datasets for training/testing.")
    parser.add_argument("--n_pois", type=int, default=-1, help="Number of pois to use (-1 to use every pois).")
    parser.add_argument("--n_train_sample", type=int, default=50000, help="Number of samples to generate for train.")
    parser.add_argument("--n_test_sample", type=int, default=10, help="Number of samples to generate for test.")
    parser.add_argument("--temperatures", type=float, nargs='+', default=[0.01]*20+[0.05]*3+[0.1], help="Temperatures for sampling.")

    args = parser.parse_args()
    main(args) 