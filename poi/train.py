import argparse
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset


def get_training_tokens(tokenizer, dataset):
    texts = []
    for sample in dataset:
        for mcqa_type in ["cardinal_direction", "proximity"]:
            for _ in range(4):
                choices = sample[f"{mcqa_type}_propositions"]
                random.shuffle(choices)
                letters = [chr(ord('A') + i) for i in range(len(choices))]
                answer_letter = letters[choices.index(sample[f"{mcqa_type}_answer"])]
                user_input = sample[f"{mcqa_type}_question"] + "".join([f"\n{letter}: {choice}" for choice, letter in zip(choices, letters)])
                assistant_output = answer_letter + ": " + sample[f"{mcqa_type}_answer"]
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that helps people find geographic information about points of interest in the city of Nancy, France."},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_output},
                ]
                text = tokenizer.apply_chat_template(messages, add_eos_token=True, tokenize=False) 
                texts.append(text)
    random.shuffle(texts)
    training_tokens = tokenizer(texts[:100], return_tensors="pt", padding=True, truncation=True, max_length=512)
    training_tokens["labels"] = training_tokens["input_ids"].clone()
    return Dataset.from_dict(training_tokens)


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="./finetuned_model",
        save_total_limit=1,
        save_steps=20,
    )
    dataset = load_dataset(args.dataset_path)
    training_tokens = get_training_tokens(tokenizer, dataset["train"])
    print(training_tokens)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_tokens,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path to train.")
    parser.add_argument("--dataset_path", type=str, default="GLauzza/geoLLM_train_dataset", help="Path to the training dataset.")
    args = parser.parse_args()
    main(args) 