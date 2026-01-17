import argparse
import random
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk, Dataset
import wandb

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
                text = tokenizer.apply_chat_template(messages, add_eos_token=True, tokenize=False) 
                prompt = text.split("</think>\n\n")[0] + "</think>\n\n"
                completion = text.split("</think>\n\n")[-1]
                conversational_dataset.append({"prompt": prompt, "completion": completion})

    random.shuffle(conversational_dataset)
    return Dataset.from_list(conversational_dataset)


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # lora_config = LoraConfig(
    #     r=256,
    #     lora_alpha=512,
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    # )

    with wandb.init(
        dir=os.environ["SCRATCH"] + "/wandb", 
        entity="G-lauzzanaa", 
        project="poi", 
        # resume="must"
        resume="never"
    ):
        training_args = SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=128,

            save_strategy="steps",
            save_steps=300,
            eval_strategy="steps",
            eval_steps=50,
            logging_steps=1,
            eval_on_start=True,

            warmup_steps=200,
            num_train_epochs=1,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",

            optim="adamw_torch_fused",
            bf16=True,

            output_dir="./finetuned_model",
            logging_dir="./finetuned_model/logs",
            report_to="wandb",
            seed=0,
        )
        train_dataset = load_from_disk(args.train_dataset)
        training_tokens = get_tokens(tokenizer, train_dataset, n_permutation=50)

        eval_dataset = load_from_disk(args.eval_dataset)
        eval_tokens = get_tokens(tokenizer, eval_dataset, n_permutation=1)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=training_tokens,
            eval_dataset=eval_tokens,
            processing_class=tokenizer,
            # peft_config=lora_config,
        )
        
        trainer.add_callback(EvalCallback(tokenizer, eval_dataset, eval_steps=50))

        trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "poi"
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--model", type=str, default=f"{os.environ['DSDIR']}/HuggingFace_Models/Qwen/Qwen3-4B-Instruct-2507", help="Model name or path to train.")
    parser.add_argument("--train_dataset", type=str, default="geoLLM_train_dataset", help="Path to the training dataset.")
    parser.add_argument("--eval_dataset", type=str, default="geoLLM_test_dataset", help="Path to the validation dataset.")
    args = parser.parse_args()
    main(args) 