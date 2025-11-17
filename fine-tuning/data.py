from datasets import Dataset, concatenate_datasets, load_dataset
import json
from typing import List, Dict

def load_train_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ds = Dataset.from_dict({"text": data})
    return ds

def split_and_add_eos(dataset: Dataset, tokenizer, test_size: float = 0.1):
    ds = dataset.shuffle(seed=42)
    split = ds.train_test_split(test_size=test_size)
    eos = tokenizer.eos_token or ""
    def add_eos(example):
        return {"text": example["text"] + eos}
    split = split.map(add_eos)
    return split

def load_eval_datasets(paths: List[str]):
    ds_list = []
    for p in paths:
        d = load_dataset(p)['train']
        ds_list.append(d)
    return concatenate_datasets(ds_list).shuffle(seed=42)

def load_eval_dataset_bus_stopa(paths: List[str]):
    # ds_list = []
    # for p in paths:
    d = load_dataset(paths[0])['train']
        #ds_list.append(d)
    return d.shuffle(seed=42)
