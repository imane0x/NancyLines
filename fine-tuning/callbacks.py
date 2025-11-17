import re
import numpy as np
from tqdm import tqdm
import torch
from transformers import TrainerCallback

class BusLineEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, val_data, bus_lines_list, eval_steps=500, patience=3):
        self.tokenizer = tokenizer
        self.val_data = val_data
        self.bus_lines_list = bus_lines_list
        self.eval_steps = eval_steps
        self.patience = patience
        self.best_f1 = -float("inf")
        self.waiting_steps = 0

    def extract_bus_lines(self, text):
        found_lines = [line for line in self.bus_lines_list if line.lower() in text.lower()]
        return set(found_lines)

    def clean_prediction(self, question, answer):
        if question in answer:
            answer = answer.replace(question, "")
        answer = re.sub(r'^[\n:?\-_]+', '', answer)
        answer = answer.strip().rstrip(".;,")
        return answer

    def compute_metrics(self, true_lines, pred_lines):
        if len(pred_lines) == 0 and len(true_lines) == 0:
            return 1.0, 1.0, 1.0, 1.0
        if len(pred_lines) == 0 or len(true_lines) == 0:
            return 0.0, 0.0, 0.0, 0.0
        correct = len(true_lines & pred_lines)
        precision = correct / len(pred_lines) if len(pred_lines) > 0 else 0.0
        recall = correct / len(true_lines) if len(true_lines) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        exact_match = 1.0 if true_lines == pred_lines else 0.0
        return precision, recall, f1, exact_match

    @torch.no_grad()
    def evaluate_bus_lines(self, model, step):
        model.eval()
        precisions, recalls, f1s, exact_matches = [], [], [], []
        for ex in tqdm(self.val_data, desc=f"Evaluating bus lines"):
            question = ex["question"]
            true_answer = ex["answer"]
            inputs = self.tokenizer(question, return_tensors="pt", truncation=True).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                use_cache=False
            )
            pred_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_answer = self.clean_prediction(question, pred_answer)
            true_lines = self.extract_bus_lines(true_answer)
            pred_lines = self.extract_bus_lines(pred_answer)
            p, r, f, em = self.compute_metrics(true_lines, pred_lines)
            precisions.append(p); recalls.append(r); f1s.append(f); exact_matches.append(em)
        results = {
            "bus_line_precision": float(np.mean(precisions)),
            "bus_line_recall": float(np.mean(recalls)),
            "bus_line_f1": float(np.mean(f1s)),
            "bus_line_exact_match": float(np.mean(exact_matches)),
            "step": step
        }
        model.train()
        return results

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            step = state.global_step
            results = self.evaluate_bus_lines(model, step)
            try:
                import wandb; wandb.log(results)
            except Exception:
                pass
            current_f1 = results['bus_line_f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.waiting_steps = 0
            else:
                self.waiting_steps += 1
                if self.waiting_steps >= self.patience:
                    control.should_training_stop = True
        return control
