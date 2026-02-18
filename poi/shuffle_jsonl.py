import argparse
import random

import numpy as np


def main(args):
    with open(args.file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if lines[-1][-1] != "\n":
        lines[-1] += "\n"

    random.shuffle(lines)

    with open(".".join(args.file_path.split(".")[:-1]) + "_shuffled.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shuffle jsonl.")
    parser.add_argument("--file_path", type=str, default="geoLLM_test_dataset/test.jsonl", help="path to the json.")

    args = parser.parse_args()
    main(args) 