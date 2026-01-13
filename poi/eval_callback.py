import wandb
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import TrainerCallback
from vllm import SamplingParams
from datasets import concatenate_datasets

from utils import format_mcqa


@torch.no_grad()
class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, vllm_model, eval_dataset, eval_steps):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.vllm_model = vllm_model
        self.eval_steps = eval_steps
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0, logprobs=20)
        self.letters_token_ids = tokenizer([chr(ord('A') + i) for i in range(26)], add_special_tokens=False, padding=False)["input_ids"]


    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if (step-1) % self.eval_steps != 0:
            return control
        
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
                input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                input_texts.append(input_text)
                assistant_ouputs.append(assistant_output)

        vllm_outputs = self.vllm_model.generate(input_texts, self.sampling_params)

        for vllm_output, assistant_ouput in zip(vllm_outputs, assistant_ouputs):
            # Exact match
            # pred_output = vllm_output.outputs[0].text

            pred_log_probs = vllm_output.outputs[0].logprobs[0]
            pred_output = "None"
            best_log_probs = float('-inf')
            for i, token_id in enumerate(self.letters_token_ids):
                if token_id[0] in pred_log_probs:
                    letter_log_probs = pred_log_probs[token_id[0]].logprob
                    if letter_log_probs > best_log_probs:
                        pred_output = chr(ord('A') + i)
                        best_log_probs = letter_log_probs

            print(f"\nPred:{pred_output}\nGold:{assistant_ouput}")

            if pred_output.lower().strip() == assistant_output.lower().strip():
                corrects.append(1)
            else:
                corrects.append(0)


        acc = sum(corrects) / len(corrects)
        results =  {
            "val_accuracy": acc,
            "step": step
        }
        wandb.log(results)

        for node_distance in self.eval_dataset.unique("node_distance"):
            corrects_node_distance = []
            for sample, correct in zip(concatenate_datasets([self.eval_dataset, self.eval_dataset]), corrects):
                if sample["node_distance"] == node_distance:
                    corrects_node_distance.append(correct)

            acc = sum(corrects_node_distance) / len(corrects_node_distance)
            results =  {
                f"val_accuracy_(distance={node_distance})": acc,
                "step": step
            }
            wandb.log(results)

        return control
