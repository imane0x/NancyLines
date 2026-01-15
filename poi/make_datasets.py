import argparse
import random
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datasets import Dataset


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


def sample_train_pair_id(pois_dataset, distance_matrix, temperature):
    source_id = random.randint(0, len(pois_dataset) - 1)

    probs = np.exp(-distance_matrix[source_id] / temperature)
    probs /= probs.sum()

    target_id = np.random.choice(len(pois_dataset), p=probs)

    return source_id, target_id


def sample_test_pair_id(pois_dataset, train_graph, node_distance):
    target_id = -1
    while target_id == -1:
        source_id = random.randint(0, len(pois_dataset) - 1)
        target_id = get_target_id_at_node_distance(train_graph, source_id, node_distance)

    return source_id, target_id


def build_pair(pois_dataset, distance_matrix, source_id, target_id, is_test=False):
    source = pois_dataset.iloc[source_id]
    target = pois_dataset.iloc[target_id]

    plt.plot([source["x"], target["x"]], [source["y"], target["y"]], c=("red" if is_test else "blue"), alpha=0.2, linewidth=0.1)

    return {
        "source": source["name"], 
        "source_id": source_id, 
        "target": target["name"],
        "target_id": target_id, 
        "source_loc": (float(source["x"]), float(source["y"])), 
        "target_loc": (float(target["x"]), float(target["y"])),
        "distance": distance_matrix[source_id, target_id],
        "angle": np.degrees(np.arctan2(target["y"] - source["y"], target["x"] - source["x"])),
    }


def add_mcqa(dataset, mcqa_type="cardinal_direction", max_distance=None):
    questions = []
    answers = []
    propositions = []
    for i, row in dataset.iterrows():
        if mcqa_type == "cardinal_direction":
            questions.append(f'"{row["target"]}" se situe:')
            propositions.append([
                f'Au nord de "{row["source"]}"',
                f'Au sud de "{row["source"]}"',
                f'À l\'ouest de "{row["source"]}"',
                f'À l\'est de "{row["source"]}"',
            ])
            if -45 <= row["angle"] < 45:
                answers.append(f'À l\'est de "{row["source"]}"')
            elif 45 <= row["angle"] < 135:
                answers.append(f'Au nord de "{row["source"]}"')
            elif -135 <= row["angle"] < -45:
                answers.append(f'Au sud de "{row["source"]}"')
            else:
                answers.append(f'À l\'ouest de "{row["source"]}"')
    
        elif mcqa_type == "proximity":
            questions.append(f'Quelle est la proximité de "{row["target"]}" par rapport à "{row["source"]}"?')
            propositions.append([
                "Très proche",
                "Proche",
                "Loin",
                "Très loin",
            ])
            if row["distance"]*max_distance < 200:
                answers.append("Très proche")
            elif row["distance"]*max_distance < 600:
                answers.append("Proche")
            elif row["distance"]*max_distance < 2000:
                answers.append("Loin")
            else:
                answers.append("Très loin")

    dataset[f"{mcqa_type}_question"] = questions
    dataset[f"{mcqa_type}_propositions"] = propositions
    dataset[f"{mcqa_type}_answer"] = answers

    return dataset


def main(args):
    pois_dataset = pd.read_csv("pois.csv")
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

    # plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    train_dataset = []
    for _ in range(args.n_train_sample):
        source_id, target_id = sample_train_pair_id(pois_dataset, distance_matrix, args.temperature)
        pair = build_pair(pois_dataset, distance_matrix, source_id, target_id)
        train_dataset.append(pair)

    train_dataset = pd.DataFrame(train_dataset)
    train_dataset = add_mcqa(train_dataset, mcqa_type="cardinal_direction")
    train_dataset = add_mcqa(train_dataset, mcqa_type="proximity", max_distance=max_distance)
    train_dataset.to_csv("train_dataset.csv", index=False)
    hf_dataset = Dataset.from_pandas(train_dataset)
    hf_dataset.save_to_disk("geoLLM_train_dataset")


    train_edges = [(row["source_id"], row["target_id"]) for k, row in train_dataset.iterrows()]
    train_graph_directed = defaultdict(list)
    train_graph_undirected = defaultdict(list)
    for a, b in train_edges:
        train_graph_directed[a].append(b)
        train_graph_undirected[a].append(b)
        train_graph_undirected[b].append(a)

    test_dataset = []
    while len(test_dataset) != args.n_test_sample:
        for is_directed, train_graph in enumerate([train_graph_undirected, train_graph_directed]):
            for node_distance in range(1,args.max_node_dist+1):
                source_id, target_id = sample_test_pair_id(pois_dataset, train_graph, node_distance)
                pair = build_pair(pois_dataset, distance_matrix, source_id, target_id, is_test=True)
                pair["node_distance"] = node_distance
                pair["directed"] = is_directed == 1
                test_dataset.append(pair)

    # random way
    # all_pairs_ids = [(i, j) for i in range(len(pois_dataset)) for j in range(len(pois_dataset))]
    # random.shuffle(all_pairs_ids)
    # for i, j in all_pairs_ids:
    #     pair = build_pair(pois_dataset, distance_matrix, i, j)
    #     pair["node_distance"] = min_nodes_between(train_graph, i, j)
    #     print(pair["node_distance"])
    #     if 0 < pair["node_distance"] < 8:
    #         test_dataset.append(pair)
    #     if len(test_dataset) >= args.n_test_sample:
    #         break

    test_dataset = pd.DataFrame(test_dataset)
    test_dataset = add_mcqa(test_dataset, mcqa_type="cardinal_direction")
    test_dataset = add_mcqa(test_dataset, mcqa_type="proximity", max_distance=max_distance)
    test_dataset.to_csv("test_dataset.csv", index=False)
    hf_dataset = Dataset.from_pandas(test_dataset)
    hf_dataset.save_to_disk("geoLLM_test_dataset")

    plt.savefig("test.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make datasets for training/testing.")
    parser.add_argument("--n_pois", type=int, default=1000, help="Number of pois to use (-1 to use every pois).")
    parser.add_argument("--n_train_sample", type=int, default=1000, help="Number of samples to generate for train.")
    parser.add_argument("--n_test_sample", type=int, default=250, help="Number of samples to generate for test.")
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for sampling.")
    parser.add_argument("--max_node_dist", type=int, default=5, help="Maximum node distance to evaluate.")

    args = parser.parse_args()
    main(args) 