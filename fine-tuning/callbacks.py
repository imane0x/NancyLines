import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import TrainerCallback



@torch.no_grad()
def evaluate_bus_lines(model, tokenizer, bus_val_data, bus_lines_list, step):
    print(f"[Step {step}] Running BusLineEval...")

    model.eval()
    precisions, recalls, f1s, ems = [], [], [], []

    for ex in tqdm(bus_val_data, desc="BusEval"):
        q = ex["question"]
        true_answer = ex["answer"]

        # --- Generate prediction
        inp = tokenizer(q, return_tensors="pt").to(model.device)
        out_ids = model.generate(
            **inp,
            max_new_tokens=200,
            do_sample=False
        )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # --- Clean pred
        if q in pred:
            pred = pred.replace(q, "")
        pred = pred.strip()

        # --- Extract bus line names
        true_lines = {b for b in bus_lines_list if b.lower() in true_answer.lower()}
        pred_lines = {b for b in bus_lines_list if b.lower() in pred.lower()}

        # --- Metrics
        if len(true_lines) == 0 and len(pred_lines) == 0:
            p = r = f = em = 1.0
        else:
            correct = len(true_lines & pred_lines)
            p = correct / len(pred_lines) if pred_lines else 0.0
            r = correct / len(true_lines) if true_lines else 0.0
            f = (2*p*r/(p+r)) if (p+r) else 0.0
            em = 1.0 if (true_lines == pred_lines) else 0.0

        precisions.append(p); recalls.append(r); f1s.append(f); ems.append(em)

    model.train()

    return {
        "bus_precision": float(np.mean(precisions)),
        "bus_recall": float(np.mean(recalls)),
        "bus_f1": float(np.mean(f1s)),
        "bus_em": float(np.mean(ems)),
        "step": step
    }


@torch.no_grad()
def evaluate_custom_streets(model, tokenizer, val_data, step):
    print(f"[Step {step}] Running custom street evaluation...")

    model.eval()
    total = 0
    correct = 0

    for ex in tqdm(val_data, desc="StreetEval"):
        q = ex["question"]
        true = ex["answer"]

        inp = tokenizer(q, return_tensors="pt").to(model.device)
        out_ids = model.generate(
            **inp,
            max_new_tokens=150,
            do_sample=False
        )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        if pred.lower() == true.lower():
            correct += 1
        total += 1

    model.train()

    acc = correct / total if total > 0 else 0.0
    return {
        "val_accuracy": acc,
        "step": step
    }



@torch.no_grad()
def evaluate_tiny_mmlu(model, tokenizer, step):
    print(f"[Step {step}] Running tinyMMLU...")

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    model.eval()

    try:
        eval_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size="auto"
        )

        results = evaluator.simple_evaluate(
            model=eval_model,
            tasks=["tinyMMLU"],
            batch_size="auto"
        )

        acc = results["results"]["tinyMMLU"]["acc_norm,none"]
        return {"tinyMMLU_acc": acc, "step": step}

    finally:
        model.train()



class UnifiedEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        bus_val_data,
        bus_lines_list,
        street_val_data,
        eval_steps=1000,
        patience=3,
        metric_name="bus_f1"  # Which metric determines early stopping
    ):
        self.tokenizer = tokenizer
        self.bus_val_data = bus_val_data
        self.bus_lines_list = bus_lines_list
        self.street_val_data = street_val_data

        self.eval_steps = eval_steps
        self.patience = patience
        self.metric_name = metric_name

        self.best_metric = -float("inf")
        self.waiting_steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.eval_steps != 0:
            return control

        model = kwargs["model"]

        # 1. Bus line eval
        bus_results = evaluate_bus_lines(
            model, self.tokenizer, self.bus_val_data, self.bus_lines_list, step
        )

        # 2. Custom streets eval
        street_results = evaluate_custom_streets(
            model, self.tokenizer, self.street_val_data, step
        )

        # 3. tinyMMLU eval
        tiny_results = evaluate_tiny_mmlu(model, self.tokenizer, step)

        # MERGE ALL RESULTS
        results = {}
        results.update(bus_results)
        results.update(street_results)
        results.update(tiny_results)

        # ---- Log to WandB if available
        try:
            import wandb
            wandb.log(results)
        except:
            pass

        # ---- Select metric to track
        current_metric = results.get(self.metric_name, None)
        if current_metric is None:
            print(f"[Step {step}] ERROR: metric '{self.metric_name}' not found in {results}")
            return control

        print(f"[Step {step}] Current {self.metric_name} = {current_metric:.4f}")

        # ---- Early stopping
        if current_metric > self.best_metric:
            print(f"   NEW BEST!")
            self.best_metric = current_metric
            self.waiting_steps = 0
        else:
            self.waiting_steps += 1
            print(f"   No improvement. Patience {self.waiting_steps}/{self.patience}")

            if self.waiting_steps >= self.patience:
                print(f"  Early stopping activated!")
                control.should_training_stop = True

        return control
