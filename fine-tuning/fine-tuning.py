import os
import wandb
import torch
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from transformers import AutoTokenizer
from .config import load_config
from .data import load_train_dataset, split_and_add_eos, load_eval_datasets
from .eval import evaluate_custom_streets, evaluate_tiny_mmlu
from .callbacks import BusLineEvalCallback

def main(cfg_path: str):
    cfg = load_config(cfg_path)

    # WandB setup (offline by default)
    wandb_dir = os.environ.get("WANDB_DIR", "./wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    os.environ["WANDB_MODE"] = cfg.wandb.get("mode", "offline")
    wandb.init(project=cfg.wandb.get("project", "qwen-finetune"))

    # load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.path,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=True,
    )

    # Data
    train_ds = load_train_dataset(cfg.data.train_json)
    split = split_and_add_eos(train_ds, tokenizer)
    eval_ds = load_eval_datasets(cfg.data.eval_paths)

    # Example callbacks & training args
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=UnslothTrainingArguments(
            dataset_text_field="text",
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            num_train_epochs=cfg.training.num_train_epochs,
            learning_rate=cfg.training.learning_rate,
            logging_strategy="steps",
            logging_steps=cfg.training.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            save_strategy="steps",
            save_steps=cfg.training.save_steps,
            save_total_limit=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to="wandb",
            output_dir="./tmp_checkpoints",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    )

    bus_lines_list = []  # fill or load from file
    trainer.add_callback()
    trainer.train()
    model.save_pretrained("./qwen3b-fft-final")
    tokenizer.save_pretrained("./qwen3b-fft-final")

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)
