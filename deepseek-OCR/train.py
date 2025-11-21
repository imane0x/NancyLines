import yaml
from torch.utils.data import DataLoader

from src.dataset import GreenPinDataset, collate_fn
from src.model_utils import load_model_and_tokenizer
from src.trainer import LLMTrainer

def main():

    # Load config
    cfg = yaml.safe_load(open("configs/default.yaml"))

    model, tokenizer, _ = load_model_and_tokenizer(cfg["model_path"])

    dataset = GreenPinDataset(
        cfg["train_manifest"],
        cfg["projector_dir"],
        tokenizer,
        max_text_len=cfg["max_text_len"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )

    trainer = LLMTrainer(
        model,
        dataloader,
        lr=cfg["learning_rate"],
        epochs=cfg["epochs"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save("outputs/finetuned-model")

if __name__ == "__main__":
    main()
