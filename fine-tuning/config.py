from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class ModelConfig:
    path: str
    max_seq_length: int = 128

@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    logging_steps: int = 200
    eval_steps: int = 500
    save_steps: int = 1000

@dataclass
class DataConfig:
    train_json: str
    eval_paths: List[str]
    eval_bus_path: List[str]

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    wandb: dict

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        wandb=raw.get("wandb", {}),
    )
