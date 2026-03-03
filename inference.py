import torch
import pandas as pd
import json
from utils import chunk_text, extract_json, find_span_in_chunk, build_messages_fs, build_messages_ft
import config
from model import get_tokenizer, load_base_model, load_lora_model
from data import load_val_grouped  # needed for get_predictions() when txt_path contains "val"


def predict_file(model, tokenizer, fileid, text, few_shot=False):
    """Run inference on a single document and return a deduplicated list of predicted spans.
    
    Chunks the document, runs the model on each chunk, parses the JSON output,
    and recovers character offsets via str.find(). Spans appearing in multiple
    chunks due to stride overlap are deduplicated via set conversion at the end.

    Args:
        model: loaded HuggingFace model (base or LoRA)
        tokenizer: corresponding tokenizer
        fileid: document identifier used as the fileid field in output
        text: full document text
        few_shot: if True uses few-shot prompt, otherwise uses fine-tuned zero-shot prompt

    Returns:
        list of dicts with keys: fileid, start, end, tag
    """
    predictions = []
    chunks      = chunk_text(text)

    for idx, (chunk_start, chunk_end, chunk_text_) in enumerate(chunks):
        print(f"Chunk {idx+1}/{len(chunks)}")

        # Select prompt format based on inference mode
        messages    = build_messages_fs(chunk_text_) if few_shot else build_messages_ft(chunk_text_)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs       = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        # Greedy decoding for deterministic outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,  # sufficient since output is substrings of 1500-char chunk
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens, not the prompt
        decoded = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        spans   = extract_json(decoded)

        for span in spans:
            if "tag" not in span or "text" not in span:
                continue
            # Recover local offset within chunk via exact string match
            local_idx, matched_text = find_span_in_chunk(span["text"], chunk_text_)
            if local_idx == -1:
                continue  # span not found in chunk, likely a hallucination
            # Convert local chunk offset to global document offset
            predictions.append({
                "fileid": fileid,
                "start":  local_idx + chunk_start,
                "end":    local_idx + chunk_start + len(matched_text),
                "tag":    span["tag"]
            })

    # Deduplicate spans that appear in multiple overlapping chunks
    return [dict(t) for t in {tuple(sorted(d.items())) for d in predictions}]


def run_inference(file_path, output_path):
    """Run fine-tuned model on a single unannotated MMD file and save predictions to JSON.
    Uses the file path as the fileid in the output."""
    tokenizer = get_tokenizer()
    model     = load_lora_model()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Running inference on: {file_path}")
    preds = predict_file(model, tokenizer, file_path, text, few_shot=False)
    df    = pd.DataFrame(preds)[["fileid", "start", "end", "tag"]]
    df.to_json(output_path)
    print(f"Saved {len(df)} spans to {output_path}")


def get_predictions(txt_path, output_path, few_shot=False):
    """Run inference on a list of fileids and save predictions to JSON.

    For validation files, fileids are loaded from val.json to ensure they match
    the keys in file_contents.json. For test files, fileids are read from txt_path directly.

    Args:
        txt_path: path to a .txt file listing fileids, one per line
        output_path: path to save the predictions JSON
        few_shot: if True loads base model, otherwise loads LoRA fine-tuned model
    """
    tokenizer = get_tokenizer()
    model     = load_base_model() if few_shot else load_lora_model()
    model.eval()

    # Val fileids must come from val.json keys to match file_contents.json exactly
    # Test fileids can be read directly from the txt file
    if "val" in txt_path:
        val_grouped = load_val_grouped()
        fileids     = list(val_grouped.keys())
    else:
        with open(txt_path) as f:
            fileids = [line.strip() for line in f if line.strip()]

    with open("file_contents.json") as f:
        fc = json.load(f)

    all_preds = []
    for fileid in fileids:
        print(f"\nInference: {fileid}")
        preds = predict_file(model, tokenizer, fileid, fc[fileid], few_shot=few_shot)
        all_preds.extend(preds)
        print(f"  -> {len(preds)} spans")

    # Final deduplication across all files
    unique = [dict(t) for t in {tuple(sorted(d.items())) for d in all_preds}]
    df     = pd.DataFrame(unique)[["fileid", "start", "end", "tag"]]
    df.to_json(output_path)
    print(f"\nSaved {len(df)} predictions to {output_path}")
    return all_preds