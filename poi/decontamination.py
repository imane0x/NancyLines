import json
import argparse

import numpy as np

def get_filter_fun(sample):
    if "Strict Monotony" in sample["generalization_type"]:
        return lambda x: not((x["mcqa_type"] == "size") and (x["source"] == sample["source"]) and (x["target"] == sample["target"]))
    elif "Cayley-Menger" in sample["generalization_type"]:
        return lambda x: not(("cardinality" in x["mcqa_type"]) and ((x["source"] == sample["source"]) or (x["source"] == sample["target"]) or (x["target"] == sample["source"]) or (x["target"] == sample["target"])))
    else:
        return lambda x: True


def main(args):
    train = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train.append(json.loads(line))
    train = np.array(train)
    test = []
    with open(args.test_path, 'r', encoding='utf-8') as f:
        for line in f:
            test.append(json.loads(line))
    test = np.array(test)

    for i, sample in enumerate(test):
        filter_fun = get_filter_fun(sample)
        len_before = len(train)
        train = train[np.vectorize(filter_fun)(train)]
        print(f"filtered {len_before-len(train)} train samples for test sample {i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="remove contaminated samples from the jsonl.")
    parser.add_argument("--train_path", type=str, default="geoLLM_train_dataset/train.jsonl", help="path to the train dataset.")
    parser.add_argument("--test_path", type=str, default="geoLLM_test_dataset/test_filtered_shuffled.jsonl", help="path to the test dataset.")
    parser.add_argument("--output_path", type=str, default="geoLLM_train_dataset/train_decontaminated.jsonl", help="path to the uncontaminated dataset.")

    args = parser.parse_args()
    main(args) 