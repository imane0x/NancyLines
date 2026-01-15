import time
import re

import wandb
import numpy as np
import torch
from tqdm import tqdm
from transformers import TrainerCallback
from datasets import concatenate_datasets

from utils import format_mcqa


@torch.no_grad()
class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, eval_steps):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.letters_token_ids = tokenizer([chr(ord('A') + i) for i in range(26)], add_special_tokens=False, padding=False)["input_ids"]


    def on_step_end(self, args, state, control, **kwargs):
        time_start = time.time()
        step = state.global_step
        if (step-1) % self.eval_steps != 0:
            return control
        
        model = kwargs["model"]
        model.eval()

        print(f"[Step {step}] Running evaluation...")
        print(self.letters_token_ids)

        input_texts = []
        assistant_ouputs = []
        corrects = []

        for mcqa_type in ["cardinal_direction", "proximity"]:
            for sample in self.eval_dataset:
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
                ]
                text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                tokens = self.tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **tokens,
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )                

                pred_probs = outputs["scores"][0][0]
                pred_tokens = self.tokenizer.decode(outputs["sequences"][0][-1], skip_special_tokens=True)

                pred_output = "None"
                best_probs = float('-inf')
                for i, token_id in enumerate(self.letters_token_ids):
                    letter_probs = pred_probs[token_id[0]]
                    if letter_probs > best_probs:
                        pred_output = chr(ord('A') + i)
                        best_probs = letter_probs

                print(f"\nMax logprob:{pred_output}\nPred:{pred_tokens}\nGold:{assistant_output}")

                if pred_output.lower().strip() == assistant_output.lower().strip():
                    corrects.append(1)
                else:
                    corrects.append(0)


        acc = sum(corrects) / len(corrects)
        results =  {
            "val_accuracy": acc,
            "step": step,
            "runtime": time.time() - time_start,
        }

        for node_distance in self.eval_dataset.unique("node_distance"):
            corrects_node_distance = []
            for sample, correct in zip(concatenate_datasets([self.eval_dataset, self.eval_dataset]), corrects):
                if sample["node_distance"] == node_distance:
                    corrects_node_distance.append(correct)

            acc = sum(corrects_node_distance) / len(corrects_node_distance)
            results [f"val_accuracy_(distance={node_distance})"] = acc

        wandb.log(results)
        model.train()

        return control
