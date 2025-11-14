import torch
import os

MAX_SEQ_LENGTH = 128
DTYPE = torch.bfloat16
LOAD_IN_4BIT = True

MODEL_PATH = "/lustre/fsn1/projects/rech/knb/umq83db/Models/Qwen3-0.6B"

WANDB_DIR = os.environ.get("WANDB_DIR", "./wandb")
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ["WANDB_DIR"] = WANDB_DIR
os.environ["WANDB_MODE"] = "offline"

CACHE_DIR = "~/.cache/huggingface/datasets/"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR

TRAIN_JSON = "/lustre/fsn1/projects/rech/knb/umq83db/Datasets/train.json"

EVAL_DATA_PATHS = {
    "easy": "/lustre/fsn1/projects/rech/knb/umq83db/Datasets/eval_data_easy",
    "skip_1": "/lustre/fsn1/projects/rech/knb/umq83db/Datasets/eval_data_skip_one_stop",
    "skip_2": "/lustre/fsn1/projects/rech/knb/umq83db/Datasets/eval_data_skip_two_stops",
    "skip_3": "/lustre/fsn1/projects/rech/knb/umq83db/Datasets/eval_data_skip_three_stops",
    "pair_line": "/lustre/fsn1/projects/rech/knb/umq83db/Datasets/eval_data_pair_line"
}
