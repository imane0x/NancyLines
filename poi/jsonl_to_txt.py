import json
import argparse

import numpy as np


def main(args):
    data = []
    with open(args.file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    text = ""
    splits = np.array([i * len(data) // len(args.names) for i in range(len(args.names))])
    for i, sample in enumerate(data):
        if i in splits:
            text += "\n\n\n" + args.names[np.where(splits == i)[0][0]] + "-"*20 + "\n"
        text += f'{i}: {sample["question"]}\n'
        text += "\n".join([i + ": " + proposition for i, proposition in sample["propositions"].items() if proposition is not None])
        if args.show_answer:
            text += f"\nRéponse {sample['answer_letter']}: {sample['answer']}"
        text += "\n\n"

    with open(".".join(args.file_path.split(".")[:-1]) + ".txt", 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jsonl to txt.")
    parser.add_argument("--file_path", type=str, default="geoLLM_test_dataset/test.jsonl", help="path to the json.")
    parser.add_argument("--show_answer", action="store_true", help="whether to show the answer.")
    parser.add_argument("--names", nargs='+', type=str, default=["GABRIEL", "IMANE", "ESTELLE", "MALO", "YAYA", "IMED"], help="names of the readers.")

    args = parser.parse_args()
    main(args) 