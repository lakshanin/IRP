#!/usr/bin/env python3
"""
Benchmarking script for a baseline model for multi-task phone recognition and error classification.
This baseline uses a simple stacked BiLSTM encoder to generate hidden representations of log-mel features,
then branches into:
  • A CTC-based phoneme recognition head.
  • A simple linear classifier for error classification.
"""

import argparse
import json
import os
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataloader import MultiTaskDataset, collate_fn


# Baseline Model Definition
class BaselineBiLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_layers=2,
                 num_ctc_classes=40, num_error_classes=3, dropout=0.1):

        super(BaselineBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        # Since it is bidirectional, the output dim is hidden_dim*2.
        self.ctc_out = nn.Linear(hidden_dim * 2, num_ctc_classes)
        self.error_out = nn.Linear(hidden_dim * 2, num_error_classes)

    def forward(self, feats, feat_lengths, canonical_ids=None, canonical_lengths=None):
        # Encode acoustic features with BiLSTM.
        encoder_out, _ = self.bilstm(feats)
        # CTC branch: linear projection on each time step.
        ctc_logits = self.ctc_out(encoder_out)

        error_logits = None
        if canonical_ids is not None and canonical_lengths is not None:
            batch_size = feats.size(0)
            aligned_features = []
            # For each instance, sample encoder output frames to align with the canonical sequence.
            for i in range(batch_size):
                T = feat_lengths[i].item()
                L = canonical_lengths[i].item()
                # Extract valid time steps.
                out_seq = encoder_out[i, :T, :]
                if L > 1:
                    # Linear interpolation of indices.
                    indices = torch.linspace(0, T - 1, steps=L).round().long().to(feats.device)
                else:
                    indices = torch.tensor([T // 2]).to(feats.device)
                aligned_feat = out_seq[indices]
                aligned_features.append(aligned_feat)
            # Pad the list to a tensor of shape
            aligned_features = nn.utils.rnn.pad_sequence(aligned_features, batch_first=True)
            # Error classification branch.
            error_logits = self.error_out(aligned_features)
        return {"ctc_logits": ctc_logits, "error_logits": error_logits}


# Utility Functions
def ctc_greedy_decode(ctc_logits, blank_id=0):
    max_ids = ctc_logits.argmax(dim=-1)
    decoded = []
    for seq in max_ids:
        out_seq = []
        prev = None
        for idx in seq.tolist():
            if idx != blank_id and idx != prev:
                out_seq.append(idx)
            prev = idx
        decoded.append(out_seq)
    return decoded


def noam_scheduler(step, warmup_steps, d_model):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def compute_multitask_loss(model, batch, alpha=1.0, device="cpu",
                           class_weights=None, blank_id=0):
    (padded_feats, feat_lengths, padded_ctc, ctc_lens,
     padded_canonical, canonical_lengths, padded_canonical_err, _) = batch

    padded_feats = padded_feats.to(device)
    feat_lengths = feat_lengths.to(device)
    padded_ctc = padded_ctc.to(device)
    ctc_lens = ctc_lens.to(device)
    padded_canonical = padded_canonical.to(device)
    padded_canonical_err = padded_canonical_err.to(device)

    out = model(padded_feats, feat_lengths, canonical_ids=padded_canonical,
                canonical_lengths=canonical_lengths)
    ctc_logits, error_logits = out["ctc_logits"], out["error_logits"]

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
    # CTC loss expects logits as (T, B, num_classes)
    ctc_logits_t = ctc_logits.transpose(0, 1)
    loss_ctc = ctc_loss_fn(ctc_logits_t, padded_ctc, feat_lengths, ctc_lens)

    if error_logits is not None:
        B, L, _ = error_logits.shape
        err_logits_flat = error_logits.view(B * L, -1)
        err_targets_flat = padded_canonical_err.view(B * L)
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        loss_error = ce_loss_fn(err_logits_flat, err_targets_flat)
    else:
        loss_error = torch.tensor(0.0, device=device)

    total_loss = loss_ctc + alpha * loss_error
    return total_loss, loss_ctc.item(), loss_error.item()


# Training and Validation Functions
def train_one_epoch(model, dataloader, optimizer, scheduler,
                    epoch, lambda_max, warmup_epochs,
                    device="cpu", grad_clip=5.0, class_weights=None):
    model.train()
    total_loss, total_ctc_loss, total_err_loss = 0.0, 0.0, 0.0
    
    alpha = lambda_max * min(epoch / warmup_epochs, 1.0)
    
    for batch in dataloader:
        loss, l_ctc, l_err = compute_multitask_loss(model, batch, alpha, device, class_weights)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_ctc_loss += l_ctc
        total_err_loss += l_err
        
    avg_loss = total_loss / len(dataloader)
    avg_ctc = total_ctc_loss / len(dataloader)
    avg_err = total_err_loss / len(dataloader)
    msg = f"Epoch {epoch}: Total Loss: {avg_loss:.4f}, CTC Loss: {avg_ctc:.4f}, Error Loss: {avg_err:.4f}, α: {alpha:.2f}"
    logging.info(msg)
    print(msg)
    return avg_loss


def validate_one_epoch(model, dataloader, alpha, device="cpu", class_weights=None):
    model.eval()
    total_loss, total_ctc_loss, total_err_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            loss, l_ctc, l_err = compute_multitask_loss(model, batch, alpha, device, class_weights)
            total_loss += loss.item()
            total_ctc_loss += l_ctc
            total_err_loss += l_err
    avg_loss = total_loss / len(dataloader)
    msg = f"Validation Loss: {avg_loss:.4f}"
    logging.info(msg)
    print(msg)
    return avg_loss


# Evaluation Function
def evaluate(model, dataloader, id2phone, blank_id=0, device="cpu"):
    model.eval()
    total_edits = 0
    total_ref_phones = 0
    all_pred_errors = []
    all_gold_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            padded_feats, feat_lengths, padded_ctc, ctc_lens, \
            padded_canonical, canonical_lengths, padded_canonical_err, _ = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            out = model(padded_feats, feat_lengths, canonical_ids=padded_canonical, canonical_lengths=canonical_lengths)
            ctc_logits = out["ctc_logits"]
            error_logits = out["error_logits"]
            
            # CTC Decoding and PER computation
            if ctc_logits is not None:
                decoded_batch = ctc_greedy_decode(ctc_logits, blank_id=blank_id)
                B = padded_feats.size(0)
                for i in range(B):
                    ref_len = ctc_lens[i].item()
                    ref_ids = padded_ctc[i, :ref_len].tolist()
                    hyp_ids = decoded_batch[i]
                    ref_seq = " ".join(id2phone.get(x, f"UNK{x}") for x in ref_ids)
                    hyp_seq = " ".join(id2phone.get(x, f"UNK{x}") for x in hyp_ids)
                    # Levenshtein distance on space-separated tokens
                    edits = levenshtein_distance(ref_seq.split(), hyp_seq.split())
                    total_edits += edits
                    total_ref_phones += len(ref_ids)
            
            # Error Classification Evaluation
            if error_logits is not None:
                B, L, _ = error_logits.shape
                logits_flat = error_logits.view(B * L, -1)
                targets_flat = padded_canonical_err.view(B * L)
                valid = (targets_flat != -100)
                if valid.sum().item() > 0:
                    preds = logits_flat[valid].argmax(dim=-1).cpu().tolist()
                    gold = targets_flat[valid].cpu().tolist()
                    all_pred_errors.extend(preds)
                    all_gold_errors.extend(gold)
    
    per = (total_edits / total_ref_phones) if total_ref_phones > 0 else 0.0
    if all_gold_errors:
        acc = accuracy_score(all_gold_errors, all_pred_errors)
        f1 = f1_score(all_gold_errors, all_pred_errors, average="macro")
        cm = confusion_matrix(all_gold_errors, all_pred_errors, labels=[0,1,2])
        report = classification_report(all_gold_errors, all_pred_errors,
                                       target_names=["Correct", "Substituted", "Deleted"],
                                       digits=3, zero_division=0)
    else:
        acc, f1, cm, report = None, None, None, None
    return per, acc, f1, cm, report


def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[n][m]


# Main Function
def main():
    parser = argparse.ArgumentParser("Benchmark Baseline Model (Resumable Training)")
    parser.add_argument("--train_manifest", required=True, help="Path to train_data.json")
    parser.add_argument("--val_manifest", required=True, help="Path to val_data.json")
    parser.add_argument("--test_manifest", required=True, help="Path to test_data.json")
    parser.add_argument("--phoneme_map", required=True, help="Path to phoneme_map.json")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--lambda_max", type=float, default=0.5, help="Max weight for error classification loss")
    parser.add_argument("--lambda_warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_model", default="baseline_model.pt", help="Path to save the best model")
    parser.add_argument("--blank_id", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default=None, help="Checkpoint path to resume training")
    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler()])
    
    device = torch.device(args.device)
    # Load phoneme map.
    with open(args.phoneme_map, "r", encoding="utf-8") as f:
        ph_map = json.load(f)
    id2phone = {v: k for k, v in ph_map.items()}
    
    # Prepare datasets and dataloaders.
    train_ds = MultiTaskDataset(args.train_manifest, ph_map, apply_specaug=True)
    val_ds = MultiTaskDataset(args.val_manifest, ph_map, apply_specaug=False)
    test_ds = MultiTaskDataset(args.test_manifest, ph_map, apply_specaug=False)
    logging.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Compute class weights for error classification (here we use ones; adapt as needed).
    class_weights = torch.ones(3).to(device)
    
    # Initialize model.
    model = BaselineBiLSTM(input_dim=40, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                           num_ctc_classes=len(ph_map), num_error_classes=3, dropout=args.dropout)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: noam_scheduler(step, args.warmup_steps, args.hidden_dim))
    
    # Resuming from a checkpoint if provided.
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_checkpoint is not None and os.path.isfile(args.resume_checkpoint):
        logging.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logging.info(f"Resumed at epoch {start_epoch}, best_val_loss so far: {best_val_loss}")
    else:
        logging.info("No checkpoint found; starting fresh training.")
    
    overall_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        logging.info(f"===== Epoch {epoch} =====")
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler,
                                     epoch, args.lambda_max, args.lambda_warmup_epochs,
                                     device=device, grad_clip=args.grad_clip, class_weights=class_weights)
        val_loss = validate_one_epoch(model, val_dl, alpha=args.lambda_max, device=device, class_weights=class_weights)
        
        # Save model only if validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }
            torch.save(checkpoint, args.save_model)
            logging.info(f"Best model saved at epoch {epoch} with validation loss {val_loss:.4f}")
        else:
            logging.info(f"No improvement at epoch {epoch}.")
    
    overall_time = time.time() - overall_start
    logging.info(f"Total training time: {overall_time:.2f} seconds")
    
    # Final evaluation on the test set.
    logging.info("Evaluating on test set...")
    per, acc, f1, cm, report = evaluate(model, test_dl, id2phone, blank_id=args.blank_id, device=device)
    eval_str = f"\nTest Results:\nPhone Error Rate (PER): {per*100:.2f}%\n"
    if acc is not None:
        eval_str += f"Error Classification Accuracy: {acc*100:.2f}%, Macro-F1: {f1*100:.2f}%\n"
        eval_str += "Confusion Matrix:\n" + str(cm) + "\n"
        eval_str += "Classification Report:\n" + report + "\n"
    else:
        eval_str += "No valid tokens for error classification metrics.\n"
    
    print(eval_str)
    with open("/content/drive/MyDrive/IRP/Final/bench/baseline_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(eval_str)
    logging.info("Evaluation complete. Results saved to baseline_evaluation.txt")


if __name__ == "__main__":

    example_args = [
        "--train_manifest", r"/content/drive/MyDrive/IRP/Implementation_New/New/preprocessed_data/train_data.json",
        "--val_manifest", r"/content/drive/MyDrive/IRP/Implementation_New/New/preprocessed_data/val_data.json",
        "--phoneme_map", r"/content/drive/MyDrive/IRP/Implementation_New/New/preprocessed_data/phoneme_map.json",
        "--test_manifest", r"/content/drive/MyDrive/IRP/Implementation_New/New/preprocessed_data/test_data.json",
        "--device", "cuda",
        "--epochs", "50",
        "--batch_size", "16",
        "--warmup_steps", "4000",
        "--lambda_max", "0.3",
        "--lambda_warmup_epochs", "10",
        "--grad_clip", "5.0",
        "--hidden_dim", "256",
        "--num_layers", "2",
        "--dropout", "0.1",
        "--save_model", r"/content/drive/MyDrive/IRP/Final/bench/baseline_model.pt",
        "--blank_id", "0",
    ]
    import sys
    sys.argv.extend(example_args)
    main()
