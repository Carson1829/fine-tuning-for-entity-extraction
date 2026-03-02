import pandas as pd
from collections import defaultdict
from inference import predict_file
from data import load_val_grouped
import config

def get_f1_scores(predictions_path):
    """
    Compute token-level F1 from a saved predictions JSON file.
    Works for both few-shot and fine-tuned predictions.

    predictions_path: path to a JSON file saved by run_few_shot(),
                      get_val_predictions(), or get_test_predictions()
    """
    df = pd.read_json(predictions_path)
    val_grouped = load_val_grouped()

    all_pred_tokens = set()
    all_gold_tokens = set()
    per_tag_pred = defaultdict(set)
    per_tag_gold = defaultdict(set)

    for fileid, data in val_grouped.items():
        # Build gold token set
        gold_tokens = set()
        for ann in data["annotations"]:
            for ci in range(ann["start"], ann["end"]):
                gold_tokens.add((ci, ann["tag"]))
                per_tag_gold[ann["tag"]].add((fileid, ci))
        all_gold_tokens |= {(fileid, c, t) for c, t in gold_tokens}

        # Build pred token set for this file
        file_preds  = df[df["fileid"] == fileid].to_dict("records")
        pred_tokens = set()
        for p in file_preds:
            for ci in range(p["start"], p["end"]):
                pred_tokens.add((ci, p["tag"]))
                per_tag_pred[p["tag"]].add((fileid, ci))
        all_pred_tokens |= {(fileid, c, t) for c, t in pred_tokens}

        p, r, f = compute_f1(pred_tokens, gold_tokens)
        print(f"{fileid[:60]}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    # Overall
    p_all, r_all, f_all = compute_f1(all_pred_tokens, all_gold_tokens)
    print("\n" + "=" * 50)
    print("OVERALL TOKEN-LEVEL F1")
    print("=" * 50)
    print(f"  Precision : {p_all:.4f}")
    print(f"  Recall    : {r_all:.4f}")
    print(f"  F1        : {f_all:.4f}")

    # Per-tag
    print("\n" + "=" * 50)
    print("PER-TAG TOKEN-LEVEL F1")
    print("=" * 50)
    rows = []
    for tag in config.VALID_TAGS:
        ps = {(fid, c) for fid, c in per_tag_pred[tag]}
        gs = {(fid, c) for fid, c in per_tag_gold[tag]}
        tp = len(ps & gs)
        prec = tp / len(ps) if ps else 0.0
        rec = tp / len(gs) if gs else 0.0
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        print(f"  {tag:12s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"(pred={len(ps)}, gold={len(gs)})")
        rows.append({"tag": tag, "precision": prec, "recall": rec, "f1": f1,
                     "pred_tokens": len(ps), "gold_tokens": len(gs)})

    df_f1 = pd.DataFrame(rows)
    df_f1.loc[len(df_f1)] = {
        "tag": "OVERALL", "precision": p_all, "recall": r_all, "f1": f_all,
        "pred_tokens": len(all_pred_tokens), "gold_tokens": len(all_gold_tokens)
    }
    print()
    print(df_f1.to_string(index=False))
    return p_all, r_all, f_all