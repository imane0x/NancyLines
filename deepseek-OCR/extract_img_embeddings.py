import os
import torch
from transformers import AutoModel, AutoTokenizer

def extract_embeddings(
    model_path,
    image_dir,
    output_dir,
    prompt="<image>\nFree OCR.",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
    ).to(device).eval()

    if device.type == "cuda":
        model = model.to(torch.bfloat16)

    os.makedirs(output_dir, exist_ok=True)

    # Prepare list of images
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Hook storage
    projector_outputs = []

    def hook_fn(module, _, output):
        projector_outputs.append(output.detach().cpu())

    # Hook registration
    inner = model.get_model()
    hook = inner.projector.register_forward_hook(hook_fn)

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        projector_outputs.clear()

        model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=img_path,
            output_path="/tmp/",
            eval_mode=True,
        )

        if len(projector_outputs) == 0:
            print(f"[WARNING] No projector output for {img_path}")
            continue

        # Save
        name = os.path.splitext(os.path.basename(img_path))[0]
        outfile = os.path.join(output_dir, f"{name}.pt")
        torch.save(projector_outputs[0], outfile)
        print(f"Saved → {outfile}")

    hook.remove()


if __name__ == "__main__":
    extract_embeddings(
        model_path="/lustre/fsn1/projects/rech/knb/umq83db/Models/DeepSeek-OCR-Latest-BF16.I64"",
        image_dir="/lustre/fsn1/projects/rech/knb/umq83db/Datasets/nancy_screenshots/",
        output_dir="lustre/fsn1/projects/rech/knb/umq83db/Datasets/projector/",
    )
