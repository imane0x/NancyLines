import re
import numpy as np
from tqdm import tqdm
import torch

def compare_streets(true_list: str, pred_list: str):
    true_streets = {s.strip().lower() for s in re.split(r",|;", true_list) if s.strip()}
    pred_streets = {s.strip().lower() for s in re.split(r",|;", pred_list) if s.strip()}
    correct = len(true_streets & pred_streets)
    total_pred = len(pred_streets)
    total_true = len(true_streets)
    precision = correct / total_pred if total_pred else 0.0
    recall = correct / total_true if total_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1

@torch.no_grad()
def evaluate_custom_streets(model, tokenizer, val_data, step=None, device=None):
    model.eval()
    precisions, recalls, f1s = [], [], []
    phrases_to_remove = (
        r"(?:en empruntant|le parcours suivant\s*:|passe par|"
        r"traverse principalement|en passant par|emprunte les rues suivantes\s*:)"
    )
    for ex in tqdm(val_data, desc="Evaluating streets"):
        question = ex["question"]
        true_answer = ex["answer"]
        inputs = tokenizer(question, return_tensors="pt", truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50, use_cache=False)
        pred_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if question in pred_answer:
            pred_answer = pred_answer.replace(question, "")
        match = re.search(phrases_to_remove, pred_answer, flags=re.IGNORECASE)
        if match:
            pred_answer = pred_answer[match.end():]
        pred_answer = re.sub(r'^[\n:?\-_]+', '', pred_answer)
        pred_answer = pred_answer.strip().rstrip(".;,")
        p, r, f = compare_streets(true_answer, pred_answer)
        precisions.append(p); recalls.append(r); f1s.append(f)
    result = {
        "streets_precision": float(np.mean(precisions)),
        "streets_recall": float(np.mean(recalls)),
        "streets_f1": float(np.mean(f1s)),
    }
    try:
        import wandb; wandb.log({**result, "step": step})
    except Exception:
        pass
    model.train()
    return result
