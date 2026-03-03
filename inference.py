import torch
import pandas as pd
import json
from utils import chunk_text, extract_json, find_span_in_chunk, build_messages_fs, build_messages_ft
import config
from model import get_tokenizer, load_base_model, load_lora_model


def predict_file(model, tokenizer, fileid, text, few_shot=False):
    predictions = []
    chunks = chunk_text(text)

    for idx, (chunk_start, chunk_end, chunk_text_) in enumerate(chunks):
        print(f"Chunk {idx+1}/{len(chunks)}")
        
        if few_shot:
            messages = build_messages_fs(chunk_text_)
        else:
            messages = build_messages_ft(chunk_text_)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        spans = extract_json(decoded)

        for span in spans:
            if "tag" not in span or "text" not in span:
                continue
            local_idx, matched_text = find_span_in_chunk(span["text"], chunk_text_)
            if local_idx == -1:
                continue
            predictions.append({
                "fileid": fileid,
                "start": local_idx + chunk_start,
                "end": local_idx + chunk_start + len(matched_text),
                "tag": span["tag"]
            })

    return [dict(t) for t in {tuple(sorted(d.items())) for d in predictions}]


def run_inference(file_path, output_path):
    tokenizer = get_tokenizer()
    model = load_lora_model()
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Running inference on: {file_path}")
    preds = predict_file(model, tokenizer, file_path, text, few_shot=False)
    df = pd.DataFrame(preds)[["fileid", "start", "end", "tag"]]
    df.to_json(output_path)
    print(f"Saved {len(df)} spans to {output_path}")


def get_predictions(txt_path, output_path, few_shot=False):
    """Run inference on fileids listed in txt_path and save predictions."""
    tokenizer = get_tokenizer()
    model = load_lora_model() if not few_shot else load_base_model()
    model.eval()

    if "val" in txt_path:
        val_grouped = load_val_grouped()
        fileids = list(val_grouped.keys())
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

    unique = [dict(t) for t in {tuple(sorted(d.items())) for d in all_preds}]
    df = pd.DataFrame(unique)[["fileid", "start", "end", "tag"]]
    df.to_json(output_path)
    print(f"\nSaved {len(df)} predictions to {output_path}")
    return all_preds