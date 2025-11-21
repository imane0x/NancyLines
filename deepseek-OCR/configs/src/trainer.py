import torch
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

class LLMTrainer:
    def __init__(self, model, dataloader, lr, epochs, tokenizer):
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.epochs = epochs
        self.tokenizer = tokenizer
        self.accelerator = Accelerator()

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )
        steps = self.epochs * len(self.dataloader)
        self.scheduler = get_scheduler("linear", self.optimizer, 0, steps)

        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

    def train(self):
        self.model.train()
        progress = tqdm(range(self.epochs * len(self.dataloader)),
                        disable=not self.accelerator.is_local_main_process)

        for epoch in range(self.epochs):
            for batch in self.dataloader:

                img_embeds = batch["img_embeds"].to(self.accelerator.device)
                input_ids  = batch["input_ids"].to(self.accelerator.device)
                gt_ids     = batch["gt_ids"].to(self.accelerator.device)

                text_embeds = self.model.model.embed_tokens(input_ids)

                B, K = img_embeds.size(0), img_embeds.size(1)
                T = text_embeds.size(1)
                A = gt_ids.size(1)

                max_len = max(T, A)
                if max_len > T:
                    text_embeds = torch.cat(
                        [text_embeds,
                         torch.zeros((B, max_len - T, text_embeds.size(2)),
                                      device=text_embeds.device)],
                        dim=1
                    )

                input_embeds = torch.cat([img_embeds, text_embeds], dim=1)

                # labels
                gt_padded = torch.full((B, max_len), -100,
                                       dtype=gt_ids.dtype,
                                       device=gt_ids.device)
                gt_padded[:, :A] = gt_ids
                labels = torch.cat(
                    [torch.full((B, K), -100, device=gt_ids.device), gt_padded],
                    dim=1
                )

                outputs = self.model(inputs_embeds=input_embeds, labels=labels)
                loss = outputs.loss

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress.update(1)

            self.accelerator.print(f"Epoch {epoch} | Loss = {loss.item():.4f}")

    def save(self, output_dir):
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
