import json
from collections import defaultdict
from utils import chunk_text, build_messages_ft
import config


def load_and_group(annotation_path, file_contents_path="file_contents.json"):
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
    examples = []
    for item in grouped_data:
        text = item["text"]
        annotations = item["annotations"]

        for chunk_start, chunk_end, chunk_text_ in chunk_text(text):
            chunk_annotations = []
            for ann in annotations:
                if ann["start"] < chunk_end and ann["end"] > chunk_start:
                    local_start = max(chunk_start, ann["start"]) - chunk_start
                    local_end = min(chunk_end, ann["end"]) - chunk_start
                    chunk_annotations.append({
                        "tag": ann["tag"],
                        "text": chunk_text_[local_start:local_end]
                    })

            messages = build_messages_ft(chunk_text_)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            json_str = json.dumps(chunk_annotations, indent=2, ensure_ascii=False)
            output_ids = tokenizer(json_str, add_special_tokens=False)["input_ids"]
            output_ids += [tokenizer.eos_token_id]

            input_ids = prompt_ids + output_ids
            labels = [-100] * len(prompt_ids) + output_ids

            if len(input_ids) > config.MAX_LENGTH:
                input_ids = input_ids[:config.MAX_LENGTH]
                labels = labels[:config.MAX_LENGTH]

            examples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels
            })

    return examples

def load_val_grouped(val_path="val.json", contents_path="file_contents.json"):
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
            "end": val_raw["end"][idx],
            "tag": val_raw["tag"][idx]
        })
    return val_grouped