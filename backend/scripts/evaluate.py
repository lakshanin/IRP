#!/usr/bin/env python3
"""
Evaluation script for a multi-task Conformer model on phoneme recognition and error classification.
Computes:
  • Phone Error Rate (PER) from CTC outputs.
  • Accuracy, F1, and confusion matrix for 3-class phoneme error classification (Correct, Substituted, Deleted).
"""

import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from Levenshtein import distance as lev_distance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataloader import MultiTaskDataset, collate_fn
from model import MultiTaskConformer


def ctc_greedy_decode(logits_ctc, blank_id=0):
    # argmax over classes
    max_ids = logits_ctc.argmax(dim=-1)
    batch_seqs = []
    for seq in max_ids:
        out_seq = []
        prev = None
        for idx in seq.tolist():
            if idx != blank_id and idx != prev:
                out_seq.append(idx)
            prev = idx
        batch_seqs.append(out_seq)
    return batch_seqs


def evaluate(model, loader, id2phone, blank_id=0, device="cpu"):
    model.eval()

    total_edits = 0
    total_ref_phones = 0

    all_pred_errors = []
    all_gold_errors = []

    with torch.no_grad():
        for batch in loader:
            padded_feats, feat_lengths, padded_ctc, ctc_lengths, \
                padded_canonical, canonical_lengths, padded_canonical_err, _ = [
                    x.to(device) if torch.is_tensor(x) else x for x in batch
                ]

            # Forward pass
            out = model(
                padded_feats, feat_lengths,
                canonical_ids=padded_canonical,
                canonical_lengths=canonical_lengths
            )
            ctc_logits = out["ctc_logits"]
            error_logits = out["error_logits"]

            # CTC PER Computation
            if ctc_logits is not None:
                decoded_batch = ctc_greedy_decode(ctc_logits, blank_id=blank_id)
                B = padded_feats.size(0)
                for i in range(B):
                    ref_len = ctc_lengths[i].item()
                    ref_ids = padded_ctc[i, :ref_len].tolist()
                    hyp_ids = decoded_batch[i]

                    # Convert phone IDs
                    ref_str = " ".join(id2phone.get(x, f"UNK{x}") for x in ref_ids)
                    hyp_str = " ".join(id2phone.get(x, f"UNK{x}") for x in hyp_ids)

                    # Levenshtein distance on space-separated tokens
                    total_edits += lev_distance(ref_str, hyp_str)
                    total_ref_phones += len(ref_ids)

            # Error Classification
            if error_logits is not None:
                B_, L_, num_err = error_logits.shape
                # Flatten
                error_logits_flat = error_logits.view(B_ * L_, num_err)
                targets_flat = padded_canonical_err.view(B_ * L_)

                # Only evaluate on valid (non -100) labels
                valid = (targets_flat != -100)
                if valid.sum().item() > 0:
                    pred = error_logits_flat[valid].argmax(dim=-1).cpu().tolist()
                    gold = targets_flat[valid].cpu().tolist()
                    all_pred_errors.extend(pred)
                    all_gold_errors.extend(gold)

    # Compute PER
    if total_ref_phones > 0:
        per_value = total_edits / total_ref_phones
    else:
        per_value = 0.0

    # Compute Error Classification Metrics
    if len(all_gold_errors) == 0:
        # No valid tokens for error classification
        return per_value, None, None, None, None

    all_pred_errors = np.array(all_pred_errors)
    all_gold_errors = np.array(all_gold_errors)

    acc = accuracy_score(all_gold_errors, all_pred_errors)
    f1 = f1_score(all_gold_errors, all_pred_errors, average="macro")

    # Classes => 0=Correct, 1=Substituted, 2=Deleted
    classes = [0, 1, 2]
    label_names = ["Correct", "Substituted", "Deleted"]

    cm = confusion_matrix(all_gold_errors, all_pred_errors, labels=classes)
    report = classification_report(
        all_gold_errors, all_pred_errors,
        labels=classes, target_names=label_names,
        digits=3, zero_division=0
    )
    return per_value, acc, f1, cm, report


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        "Evaluation script for a multi-task Conformer model that uses both "
        "CTC (phone recognition) and 3-class error classification."
    )
    parser.add_argument("--test_manifest", required=True, help="Path to test_data.json")
    parser.add_argument("--phoneme_map", required=True, help="Path to phoneme_map.json")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--blank_id", type=int, default=0, help="CTC blank id")
    parser.add_argument("--eval_log", type=str, default="evaluation_results.txt",
                        help="Where to save final results")

    parser.add_argument("--input_dim", type=int, default=80)
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--dim_model", type=int, default=512)
    parser.add_argument("--dim_ff", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_error_classes", type=int, default=3)

    parsed_args = parser.parse_args(args)
    device = torch.device(parsed_args.device)

    # Load phoneme map
    with open(parsed_args.phoneme_map, "r", encoding="utf-8") as f:
        ph_map = json.load(f)
    id2phone = {v: k for k, v in ph_map.items()}

    # Build dataset + dataloader
    test_ds = MultiTaskDataset(parsed_args.test_manifest, ph_map, apply_specaug=False)
    test_dl = DataLoader(
        test_ds,
        batch_size=parsed_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Construct model
    num_ctc_classes = len(ph_map)
    model = MultiTaskConformer(
        input_dim=parsed_args.input_dim,
        num_blocks=parsed_args.num_blocks,
        dim_model=parsed_args.dim_model,
        dim_ff=parsed_args.dim_ff,
        num_heads=parsed_args.num_heads,
        kernel_size=parsed_args.kernel_size,
        dropout=parsed_args.dropout,
        num_ctc_classes=num_ctc_classes,
        num_error_classes=parsed_args.num_error_classes
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(parsed_args.model_checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.eval()

    # Evaluate
    per, acc, f1, cm, report = evaluate(
        model, test_dl,
        id2phone=id2phone,
        blank_id=parsed_args.blank_id,
        device=device
    )

    # Build result string + log it
    result_str = "\n"
    result_str += f"Phone Error Rate (CTC): {per * 100:.2f}%\n"
    if acc is None:
        result_str += "No valid tokens for error classification metrics.\n"
    else:
        result_str += f"Error Classification => Accuracy: {acc*100:.2f}%, F1 (macro): {f1*100:.2f}%\n"
        label_names = ["Correct", "Substituted", "Deleted"]
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        result_str += "Confusion Matrix:\n" + cm_df.to_string() + "\n"
        result_str += "Full Classification Report:\n" + report + "\n"

    print(result_str)
    with open(parsed_args.eval_log, "w", encoding="utf-8") as log_f:
        log_f.write(result_str)
    print(f"Results logged to {parsed_args.eval_log}")


if __name__ == "__main__":
    import sys
    example_args = [
        "--test_manifest",   r"E:\IRP\backend\data\preprocessed_five\test_data.json",
        "--phoneme_map", r"E:\IRP\backend\data\preprocessed_five\phoneme_map.json",
        "--model_checkpoint", r"E:\IRP\backend\exps\exp_9\model.pt",
        "--device", "cpu",
        "--batch_size", "8",
        "--eval_log", r"E:\IRP\backend\exps\exp_9\eval.txt",
        "--input_dim", "40",
        "--num_blocks", "4",
        "--dim_model", "256",
        "--dim_ff", "1024",
        "--num_heads", "4",
        "--kernel_size", "15",
        "--dropout", "0.1",
        "--num_error_classes", "3"
    ]
    sys.argv.extend(example_args)
    main()
