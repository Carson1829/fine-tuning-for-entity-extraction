import json
import random
from collections import defaultdict
from utils import chunk_text, build_messages_ft
import config


def load_and_group(annotation_path, file_contents_path="file_contents.json"):
    """Load annotations and full document texts, grouped by fileid.

    Converts the flat columnar JSON format of train.json/val.json into a list
    of per-document dicts, each containing the full text and all annotations.

    Args:
        annotation_path: path to train.json or val.json
        file_contents_path: path to file_contents.json containing full document texts

    Returns:
        list of dicts with keys: fileid, text, annotations
        where annotations is a list of {start, end, tag} dicts
    """
    with open(annotation_path) as f:
        raw = json.load(f)
    with open(file_contents_path) as f:
        fc = json.load(f)

    grouped = defaultdict(lambda: {"text": None, "annotations": []})
    for idx in raw["fileid"].keys():
        fileid = raw["fileid"][idx]
        grouped[fileid]["text"] = fc[fileid]
        grouped[fileid]["annotations"].append({
            "start": raw["start"][idx],
            "end":   raw["end"][idx],
            "tag":   raw["tag"][idx]
        })

    return [
        {"fileid": fid, "text": d["text"], "annotations": d["annotations"]}
        for fid, d in grouped.items()
    ]


def get_examples(grouped_data, tokenizer):
    """Convert grouped document data into tokenized training examples.

    For each document, chunks the text and finds annotations overlapping each chunk.
    Annotations that cross chunk boundaries are clipped to the chunk boundaries.
    Empty chunks (no annotations) are included at a 10% rate to teach the model
    to output [] rather than hallucinating spans on unannotated text.

    Each example consists of:
        - input_ids: prompt tokens + JSON output tokens
        - attention_mask: all 1s
        - labels: -100 for prompt tokens (masked from loss) + JSON output tokens

    Args:
        grouped_data: output of load_and_group()
        tokenizer: HuggingFace tokenizer for the base model

    Returns:
        list of dicts ready to be passed to HuggingFace Trainer
    """
    examples = []
    for item in grouped_data:
        text        = item["text"]
        annotations = item["annotations"]

        for chunk_start, chunk_end, chunk_text_ in chunk_text(text):
            chunk_annotations = []
            for ann in annotations:
                # Check if annotation overlaps with this chunk
                if ann["start"] < chunk_end and ann["end"] > chunk_start:
                    # Clip annotation boundaries to chunk boundaries and convert to local offsets
                    local_start = max(chunk_start, ann["start"]) - chunk_start
                    local_end   = min(chunk_end,   ann["end"])   - chunk_start
                    chunk_annotations.append({
                        "tag":  ann["tag"],
                        "text": chunk_text_[local_start:local_end]
                    })

            # Include empty chunks at 10% rate to teach model to output []
            if not chunk_annotations and random.random() > 0.10:
                continue

            # Build prompt and tokenize — fine-tuned model uses zero-shot prompt
            messages    = build_messages_ft(chunk_text_)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            # Tokenize expected JSON output and append EOS to signal end of generation
            json_str   = json.dumps(chunk_annotations, indent=2, ensure_ascii=False)
            output_ids = tokenizer(json_str, add_special_tokens=False)["input_ids"]
            output_ids += [tokenizer.eos_token_id]

            input_ids = prompt_ids + output_ids
            # Mask prompt tokens with -100 so loss is only computed on JSON output
            labels    = [-100] * len(prompt_ids) + output_ids

            # Truncate to MAX_LENGTH if combined length exceeds limit
            if len(input_ids) > config.MAX_LENGTH:
                input_ids = input_ids[:config.MAX_LENGTH]
                labels    = labels[:config.MAX_LENGTH]

            examples.append({
                "input_ids":      input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels":         labels
            })

    return examples


def load_val_grouped(val_path="val.json", contents_path="file_contents.json"):
    """Load validation annotations grouped by fileid, including full document texts.

    Similar to load_and_group() but returns a dict keyed by fileid instead of a list.
    The dict format is used by get_f1_scores() to look up gold annotations per document
    when comparing against model predictions.

    Args:
        val_path: path to val.json
        contents_path: path to file_contents.json

    Returns:
        dict mapping fileid -> {text, annotations}
    """
    with open(val_path) as f:
        val_raw = json.load(f)
    with open(contents_path) as f:
        fc = json.load(f)

    val_grouped = defaultdict(lambda: {"text": None, "annotations": []})
    for idx in val_raw["fileid"].keys():
        fileid = val_raw["fileid"][idx]
        val_grouped[fileid]["text"] = fc[fileid]
        val_grouped[fileid]["annotations"].append({
            "start": val_raw["start"][idx],
            "end":   val_raw["end"][idx],
            "tag":   val_raw["tag"][idx]
        })
    return val_grouped