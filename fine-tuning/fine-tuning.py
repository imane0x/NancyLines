import os
import wandb
import torch
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
#from transformers import AutoTokenizer
from config import load_config
from data import load_train_dataset, split_and_add_eos, load_eval_datasets, load_eval_dataset_bus_stops
#from eval import evaluate_custom_streets, evaluate_tiny_mmlu
from callbacks import UnifiedEvalCallback

def main(cfg_path: str):
    cfg = load_config("/lustre/fsn1/projects/rech/knb/umq83db/NancyLines/fine-tuning/config.yaml")

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
    street_val_data = load_eval_datasets(cfg.data.eval_paths)
    bus_val_data = load_eval_dataset_bus_stops(cfg.data.eval_bus_path)

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

    bus_lines_list = ['Brabois express : Nancy Gare → Vandœuvre Brabois Santé',
 'Brabois express : Vandœuvre Brabois Santé → Nancy Gare',
 'Bus 10 : Frouard Munch/Champigneulles Rouchotte → Nancy Place Carnot',
 'Bus 10 : Nancy Place Carnot → Frouard Munch/Champigneulles Rouchotte',
 'Bus 11 : Saulxures Lorraine → Vandoeuvre Roberval',
 'Bus 11 : Tomblaine Maria Deraismes → Vandoeuvre Roberval',
 'Bus 11 : Vandoeuvre Roberval → Saulxures Lorraine',
 'Bus 11 : Vandoeuvre Roberval → Tomblaine Maria Deraismes',
 'Bus 12 : Heillecourt → Malzéville Savlons',
 'Bus 12 : Malzéville Savlons → Heillecourt',
 'Bus 13 : Dommartemont → Maxéville Écoparc',
 'Bus 13 : Maxéville Écoparc → Dommartemont',
 'Bus 15 : Essey Porte Verte → Nancy Place Carnot',
 'Bus 15 : Nancy Place Carnot → Essey Porte Verte',
 'Bus 16 : Malzéville Margeville → Villers Clairlieu',
 'Bus 16 : Malzéville Pixerécourt → Villers Clairlieu',
 'Bus 16 : Malzéville → Villers Clairlieu',
 'Bus 16 : Villers Clairlieu → Malzéville',
 'Bus 16 : Villers Clairlieu → Malzéville Margeville',
 'Bus 16 : Villers Clairlieu → Malzéville Pixerécourt',
 'Bus 17 : Ludres Marvingt → Villers Campus Sciences',
 'Bus 17 : Villers Campus Sciences → Ludres Marvingt',
 'Bus 20 : Art-sur-Meurthe → Nancy Gare',
 'Bus 20 : Nancy Gare → Art-sur-Meurthe',
 'Bus 21 : Fléville → Nancy Gare',
 'Bus 21 : Ludres Marvingt → Nancy Gare',
 'Bus 21 : Nancy Gare → Fléville',
 'Bus 21 : Nancy Gare → Ludres Marvingt',
 'Bus 22 : Essey Porte Verte → Saint-Max Gérard Barrois',
 'Bus 22 : Saint-Max Gérard Barrois → Essey Porte Verte',
 'Bus 30 : Laneuveville Gare → Villers Campus Sciences',
 'Bus 30 : Villers Campus Sciences → Laneuveville Gare',
 'Bus 32 : Essey La Fallée → Nancy Jean Lamour',
 'Bus 32 : Nancy Jean Lamour → Essey La Fallée',
 'Bus 33 : Jarville Gabriel Fauré → Nancy Gare - République',
 'Bus 33 : Nancy Gare - République → Jarville Gabriel Fauré',
 'Citadine 1 : Nancy',
 'Citadine 2 : Vandœuvre Roberval ↔ Vandœuvre Roberval',
 'Corol 1 : Laxou Plateau de Haye → Laxou Plateau de Haye',
 'Corol 2 : Laxou Plateau de Haye → Laxou Plateau de Haye',
 'Pleine Lune : Gare - Poirel => Gare - Poirel',
 'R410 Direct : Toul Gare Routière => Nancy République',
 'Tempo 2 : Laneuveville Centre → Laxou Sapinière',
 'Tempo 2 : Laxou Sapinière → Laneuveville Centre',
 'Tempo 3 : Seichamps Haie Cerlin → Villers Campus Sciences',
 'Tempo 3 : Villers Campus Sciences → Seichamps Haie Cerlin',
 'Tempo 4 : Houdemont Porte Sud → Laxou Champ le Boeuf',
 'Tempo 4 : Laxou Champ le Boeuf → Houdemont Porte Sud',
 'Tempo 5 : Maxéville Meurthe-Canal → Vandœuvre Roberval',
 'Tempo 5 : Vandœuvre Roberval → Maxéville Meurthe-Canal']
    
    trainer.add_callback(UnifiedEvalCallback(tokenizer, bus_val_data, bus_lines_list, street_val_data))
    trainer.train()
#    model.save_pretrained("./qwen3b-fft-final")
 #   tokenizer.save_pretrained("./qwen3b-fft-final")

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)
