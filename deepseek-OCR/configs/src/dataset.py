import os
import json
import torch
from torch.utils.data import Dataset

class GreenPinDataset(Dataset):
    def __init__(self, manifest_path, projector_dir, tokenizer, max_text_len=64):
        self.items = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
        self.projector_dir = projector_dir
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # image projector embeddings
        img_embeds = torch.load(os.path.join(self.projector_dir, item["proj_file"]))
        img_embeds = img_embeds.squeeze(0)

        # prompt → tokens
        prompt_ids = self.tokenizer(
            item["prompt"],
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_text_len
        ).input_ids.squeeze(0)

        # answer → tokens
        gt_ids = self.tokenizer(
            item["answer"],
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.squeeze(0)

        return {
            "img_embeds": img_embeds,
            "input_ids": prompt_ids,
            "gt_ids": gt_ids,
        }


def collate_fn(batch, pad_token_id):
    B = len(batch)

    max_text_len = max(len(b["input_ids"]) for b in batch)
    max_gt_len   = max(len(b["gt_ids"]) for b in batch)
    max_img_len  = max(b["img_embeds"].size(0) for b in batch)
    D_proj = batch[0]["img_embeds"].size(1)

    input_ids = torch.full((B, max_text_len), pad_token_id, dtype=torch.long)
    gt_ids    = torch.full((B, max_gt_len), pad_token_id, dtype=torch.long)
    img_embs  = torch.zeros((B, max_img_len, D_proj), dtype=torch.float)

    for i, b in enumerate(batch):
        input_ids[i, :len(b["input_ids"])] = b["input_ids"]
        gt_ids[i, :len(b["gt_ids"])]       = b["gt_ids"]
        img_embs[i, :b["img_embeds"].size(0)] = b["img_embeds"]

    return {
        "input_ids": input_ids,
        "gt_ids": gt_ids,
        "img_embeds": img_embs
    }
