from transformers import AutoTokenizer, AutoModel

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, use_safetensors=True)

    llm = model.get_model()
    if hasattr(llm, "embed_tokens"):
        H = llm.embed_tokens.embedding_dim
    elif hasattr(llm, "transformer") and hasattr(llm.transformer, "wte"):
        H = llm.transformer.wte.embedding_dim
    else:
        raise ValueError("Cannot find LLM embedding layer")

    # Remove unused blocks
    model.model.sam_model = None
    model.model.vision_model = None
    model.model.projector = None

    return model, tokenizer, H
