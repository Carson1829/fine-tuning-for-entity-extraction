import torch
from utils import chunk_text, extract_json, find_span_in_chunk, build_messages_fs, build_messages_ft
import config


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

def run_few_shot(output_path="few_shot_val_predictions.json"):
    """Run few-shot prompting on the validation set and save predictions."""
    tokenizer = get_tokenizer()
    model load_base_model()

    val_grouped = load_val_grouped()
    all_preds = []

    for fileid, data in val_grouped.items():
        print(f"\n=== Few-shot: {fileid[:70]} ===")
        preds = predict_file(model, tokenizer, fileid, data["text"], few_shot=True)
        all_preds.extend(preds)
        print(f"{len(preds)} spans extracted")

    if all_preds:
        df = pd.DataFrame(all_preds)[["fileid", "start", "end", "tag"]]
        df.to_json(output_path)
        print(f"\nSaved {len(df)} few-shot predictions to {output_path}")
    else:
        print("No predictions extracted.")

    return all_preds

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

def get_val_predictions(output_path="val_predictions_finetuned.json"):
    """Run fine-tuned model on validation set and save predictions."""
    tokenizer = get_tokenizer()
    model = load_lora_model()

    val_grouped = load_val_grouped()
    all_preds = []

    for fileid, data in val_grouped.items():
        print(f"\nVal inference: {fileid}")
        preds = predict_file(model, tokenizer, fileid, data["text"], few_shot=False)
        all_preds.extend(preds)
        print(f"{len(preds)} spans")

    unique = [dict(t) for t in {tuple(sorted(d.items())) for d in all_preds}]
    df = pd.DataFrame(unique)[["fileid", "start", "end", "tag"]]
    df.to_json(output_path)
    print(f"\nSaved {len(df)} val predictions to {output_path}")
    return all_preds

TEST_IDS = [
    "(mmd) Complex Manifolds - Differential Analysis on Complex Manifolds - Wells.mmd-nilay-nilay-p195-196-FacebookAI_roberta-base.json",
    "(mmd) Number Theory - Number Theory - An Introduction to Mathematics - Coppel.mmd-nilay-laurel-p64-65-FacebookAI_roberta-base.json",
    "(mmd) A Term of Commutative Algebra - Altman.mmd-victoriacochran-victoria-p133-134-FacebookAI_roberta-base.json"
]

def get_test_predictions(output_path="test_predictions_finetuned.json"):
    """Run fine-tuned model on held-out test fileids and save predictions."""
    tokenizer = get_tokenizer()
    model = load_lora_model()

    with open("file_contents.json") as f:
        fc = json.load(f)

    all_preds = []
    for fileid in TEST_IDS:
        print(f"\nTest inference: {fileid}")
        preds = predict_file(model, tokenizer, fileid, fc[fileid], few_shot=False)
        all_preds.extend(preds)
        print(f"{len(preds)} spans")

    unique = [dict(t) for t in {tuple(sorted(d.items())) for d in all_preds}]
    df = pd.DataFrame(unique)[["fileid", "start", "end", "tag"]]
    df.to_json(output_path)
    print(f"\nSaved {len(df)} test predictions to {output_path}")
    print(df.groupby(["fileid", "tag"]).size().to_string())
    return all_preds