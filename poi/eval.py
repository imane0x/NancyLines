import json
import argparse
import os

import numpy as np
import datasets
from transformers import VoxtralForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch


def main(args):
    data = []
    with open(args.file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    dataset = datasets.Dataset.from_list(data)
    
    random_baseline = np.mean([1/sum([proposition is not None for proposition in sample["propositions"].values()]) for sample in dataset])
    print(f"Random baseline accuracy: {random_baseline:.4f}")

    if args.model_type == "audio-lm":
        model = VoxtralForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model_path)
        tokenizer = processor.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    letters_token_ids = tokenizer([chr(ord('A') + i) for i in range(26)], add_special_tokens=False, padding=False)["input_ids"]

    accuracy = 0.0
    for sample in dataset:
        if args.model_type == "audio-lm":
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "audio",
                    "path": os.path.join(os.path.dirname(args.file_path), sample["audio"]),
                },],
            }]
            inputs = processor.apply_chat_template(conversation, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        else:
            conversation = [{
                "role": "system",
                "content": f"Tu es un model qui répond à des questions de géographie. Réponds uniquement par une lettre majuscule correspondant à la bonne réponse ({', '.join([str(k) for k,v in sample['propositions'].items()])}).",
            },
            {
                "role": "user",
                "content": sample["question"] + "\n" + "\n".join([f"{k}: {v}" for k,v in sample["propositions"].items() if v is not None]),
            }]
            inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True).to(model.device)

        logits = model.generate(inputs, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True).scores[0]

        pred_output = "None"
        best_probs = float('-inf')
        for i, token_id in enumerate(letters_token_ids[:sum([v is not None for v in sample["propositions"].values()])]):
            letter_probs = logits[0, token_id[0]]
            print(f"Letter {chr(ord('A') + i)}: {letter_probs.item():.4f}")
            if letter_probs > best_probs:
                pred_output = chr(ord('A') + i)
                best_probs = letter_probs

        if pred_output == sample["answer_letter"]:
            accuracy += 1.0

        print(f"Question: {sample['question']}\nPropositions: {sample['propositions']}\nExpected answer: {sample['answer_letter']}\nPredicted answer: {pred_output}\n\n")

    print(f"\nAccuracy: {accuracy / len(dataset):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a model on a jsonl dataset.")
    parser.add_argument("--file_path", type=str, default="geoLLM_test_dataset/test.jsonl", help="path to the json to evaluate.")
    parser.add_argument("--model_path", type=str, default="mistralai/Voxtral-Mini-3B-2507", help="path to the model to evaluate.")
    parser.add_argument("--model_type", type=str, default="audio-lm", help="llm or audio-lm.")

    args = parser.parse_args()
    main(args) 