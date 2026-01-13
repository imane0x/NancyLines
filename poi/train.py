import argparse
import random
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk, Dataset
import wandb
from vllm import LLM

from utils import format_mcqa
from eval_callback import EvalCallback


def get_tokens(tokenizer, dataset):
    texts = []
    for sample in dataset:
        for mcqa_type in ["cardinal_direction", "proximity"]:
            for _ in range(4):
                user_input, assistant_output = format_mcqa(sample, mcqa_type)
                messages = [
                    {
                        "role": "system", 
                        "content": (
                            "You are a helpful assistant that helps people find geographic information "
                            "about points of interest in the city of Nancy, France. "
                            "You answer MCQA by outputing only the correct letter."
                        )
                    },
                    {
                        "role": "user", 
                        "content": user_input
                    },
                    {
                        "role": "assistant", 
                        "content": assistant_output
                    },
                ]
                text = tokenizer.apply_chat_template(messages, add_eos_token=True, tokenize=False) 
                texts.append(text)
    random.shuffle(texts)

    print("max_length", max([tokenizer(text, return_length=True)["length"] for text in texts]))
    training_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=320)
    training_tokens["labels"] = training_tokens["input_ids"].clone()
    return Dataset.from_dict(training_tokens)


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    vllm_model = LLM(args.model, enable_prefix_caching=True, gpu_memory_utilization=0.2, max_model_len=320, seed=0)

    with wandb.init(
        dir=os.environ["SCRATCH"] + "/wandb", 
        entity="G-lauzzanaa", 
        project="poi", 
        # resume="must"
        resume="never"
    ):
        training_args = TrainingArguments(
            per_device_train_batch_size=64,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,

            save_strategy = "steps",
            save_steps=30,
            eval_strategy = "steps",
            eval_steps=30,
            logging_steps = 1,
            eval_on_start=True,

            warmup_steps=30,
            num_train_epochs=1,
            learning_rate=2e-4,
            lr_scheduler_type = "cosine",

            optim = "adamw_torch_fused",
            bf16=True,

            output_dir="./finetuned_model",
            logging_dir="./finetuned_model/logs",
            report_to="wandb",
            seed=0,
        )
        train_dataset = load_from_disk(args.train_dataset)
        training_tokens = get_tokens(tokenizer, train_dataset)

        eval_dataset = load_from_disk(args.eval_dataset)
        eval_tokens = get_tokens(tokenizer, eval_dataset)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_tokens,
            eval_dataset=eval_tokens,
            tokenizer=tokenizer,
        )
        
        trainer.add_callback(EvalCallback(tokenizer, vllm_model, eval_dataset, eval_steps=5))

        trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "poi"
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--model", type=str, default=f"{os.environ['DSDIR']}/HuggingFace_Models/Qwen/Qwen3-4B-Instruct-2507", help="Model name or path to train.")
    parser.add_argument("--train_dataset", type=str, default="geoLLM_train_dataset", help="Path to the training dataset.")
    parser.add_argument("--eval_dataset", type=str, default="geoLLM_test_dataset", help="Path to the validation dataset.")
    args = parser.parse_args()
    main(args) 