import wandb
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import TrainerCallback
from vllm import SamplingParams

from utils import format_mcqa


@torch.no_grad()
class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, vllm_model, eval_dataset, eval_steps):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.vllm_model = vllm_model
        self.eval_steps = eval_steps
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=256)


    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.eval_steps != 0:
            return control
        
        print(f"[Step {step}] Running evaluation...")

        n_correct = 0

        input_texts = []
        assistant_ouputs = []

        for sample in self.eval_dataset:
            for mcqa_type in ["cardinal_direction", "proximity"]:
                user_input, assistant_output = format_mcqa(sample, mcqa_type)
                messages = [
                    {
                        "role": "system", 
                        "content": (
                            "You are a helpful assistant that helps people find geographic information "
                            "about points of interest in the city of Nancy, France. "
                            "You answer MCQA by outputing the correct letter followed by the full answer."
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
            pred_output = vllm_output.outputs[0].text.strip()
            if pred_output.lower().strip() == assistant_output.lower().strip():
                n_correct += 1

        acc = n_correct / (len(self.eval_dataset)*2)
        results =  {
            "val_accuracy": acc,
            "step": step
        }

        wandb.log(results)

        return control
