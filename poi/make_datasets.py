import argparse
import random
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datasets import Dataset


def min_nodes_between(edges, id1, id2):
    if id1 == id2:
        return 0

    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

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


def sample_pair_id(df, distance_matrix, temperature):
    source_id = random.randint(0, len(df) - 1)

    probs = np.exp(-distance_matrix[source_id] / temperature)
    probs /= probs.sum()

    target_id = np.random.choice(len(df), p=probs)

    return source_id, target_id


def build_pair(df, distance_matrix, source_id, target_id):
    source = df.iloc[source_id]
    target = df.iloc[target_id]

    plt.plot([source["x"], target["x"]], [source["y"], target["y"]], c='blue', alpha=0.3, linewidth=0.3)

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
                f"Au nord de {row['source']}",
                f"Au sud de {row['source']}",
                f"À l'ouest de {row['source']}",
                f"À l'est de {row['source']}",
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
    df = pd.read_csv("pois.csv")

    length = geodesic((0, df["lon"].max() - df["lon"].min()), (0,0)).meters
    height = geodesic((df["lat"].max() - df["lat"].min(), 0), (0,0)).meters
    df["x"] = (df["lon"] - df["lon"].min()) / (df["lon"].max() - df["lon"].min())*length
    df["y"] = (df["lat"] - df["lat"].min()) / (df["lat"].max() - df["lat"].min())*height
    print(f"Area size: {length} m x {height} m")

    coords = df[["x", "y"]].values
    distance_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    max_distance = distance_matrix.max()
    distance_matrix /= max_distance
    np.fill_diagonal(distance_matrix, np.inf)
    print(f"Max distance between POIs: {max_distance} m")

    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

    train_dataset = []
    for _ in range(args.n_train_sample):
        source_id, target_id = sample_pair_id(df, distance_matrix, args.temperature)
        train_dataset.append(build_pair(df, distance_matrix, source_id, target_id))

    train_dataset = pd.DataFrame(train_dataset)
    train_dataset = add_mcqa(train_dataset, mcqa_type="cardinal_direction")
    train_dataset = add_mcqa(train_dataset, mcqa_type="proximity", max_distance=max_distance)
    train_dataset.to_csv("train_dataset.csv", index=False)
    hf_dataset = Dataset.from_pandas(train_dataset)
    hf_dataset.push_to_hub("GLauzza/geoLLM_train_dataset", private=True)

    test_dataset = []
    all_pairs_ids = [(i, j) for i in range(len(df)) for j in range(len(df))]
    random.shuffle(all_pairs_ids)
    for i, j in all_pairs_ids:
        pair = build_pair(df, distance_matrix, i, j)
        pair["node_distance"] = min_nodes_between([(row["source_id"], row["target_id"]) for k, row in train_dataset.iterrows()], i, j)
        print(pair["node_distance"])
        if 0 < pair["node_distance"] < 8:
            test_dataset.append(pair)
        if len(test_dataset) >= args.n_test_sample:
            break

    test_dataset = pd.DataFrame(test_dataset)
    test_dataset = add_mcqa(test_dataset, mcqa_type="cardinal_direction")
    test_dataset = add_mcqa(test_dataset, mcqa_type="proximity", max_distance=max_distance)
    test_dataset.to_csv("test_dataset.csv", index=False)
    hf_dataset = Dataset.from_pandas(test_dataset)
    hf_dataset.push_to_hub("GLauzza/geoLLM_test_dataset", private=True)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make datasets for training/testing.")
    parser.add_argument("--n_train_sample", type=int, default=10000, help="Number of samples to generate for train.")
    parser.add_argument("--n_test_sample", type=int, default=100, help="Number of samples to generate for test.")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for sampling.")
    args = parser.parse_args()
    main(args) 