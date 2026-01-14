import argparse
import random
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk, Dataset
import wandb
from vllm import LLM

from utils import format_mcqa
from eval_callback import EvalCallback


def get_tokens(tokenizer, dataset, n_permutation=4):
    conversational_dataset = []
    for sample in dataset:
        for mcqa_type in ["cardinal_direction", "proximity"]:
            for _ in range(n_permutation):
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
                conversational_dataset.append(messages)
    random.shuffle(conversational_dataset)

    return Dataset.from_list(conversational_dataset)


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = prepare_model_for_kbit_training(model)
    # lora_config = LoraConfig(
    #     r=256,
    #     lora_alpha=512,
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, lora_config)

    vllm_model = LLM(args.model, enable_prefix_caching=True, gpu_memory_utilization=0.2, max_model_len=320, seed=0)

    with wandb.init(
        dir=os.environ["SCRATCH"] + "/wandb", 
        entity="G-lauzzanaa", 
        project="poi", 
        # resume="must"
        resume="never"
    ):
        training_args = SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,

            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=20,
            logging_steps=1,
            eval_on_start=True,
            assistant_only_loss=True,

            warmup_steps=40,
            num_train_epochs=5,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",

            optim="adamw_torch_fused",
            bf16=True,

            output_dir="./finetuned_model",
            logging_dir="./finetuned_model/logs",
            report_to="wandb",
            seed=0,
        )
        train_dataset = load_from_disk(args.train_dataset)
        training_tokens = get_tokens(tokenizer, train_dataset, n_permutation=4)

        eval_dataset = load_from_disk(args.eval_dataset)
        eval_tokens = get_tokens(tokenizer, eval_dataset, n_permutation=1)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=training_tokens,
            eval_dataset=eval_tokens,
            processing_class=tokenizer,
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