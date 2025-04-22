#!/usr/bin/env python3
"""
Training script for MultiTaskConformer on phoneme recognition and error classification.

Includes:
  • Multi-task loss combining CTC and phoneme-level error classification.
  • Focal loss with label smoothing for robust error detection.
  • Learning rate scheduling using Noam scheme.
  • Canonical-dependency probing and debug decoding after each epoch.
  • Logging and checkpointing with resume functionality.
"""

import argparse
import json
import time
import os
import torch
import numpy as np
from tqdm import tqdm
import logging
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

from dataloader import MultiTaskDataset, collate_fn
from model import MultiTaskConformer


# Decode function for the CTC branch
def ctc_greedy_decode(logits_ctc, blank_id=0):
    max_ids = logits_ctc.argmax(dim=-1)  # shape (B, T)
    decoded_batch = []
    for seq in max_ids:
        out_seq, prev = [], None
        for idx in seq.tolist():
            if idx != blank_id and idx != prev:
                out_seq.append(idx)
            prev = idx
        decoded_batch.append(out_seq)
    return decoded_batch


def noam_scheduler(step, warmup_steps, d_model):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


# Focal Loss with label smoothing
def focal_loss(
        logits, targets,
        gamma: float = 2.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.1):

    ce = F.cross_entropy(
        logits, targets,
        ignore_index = ignore_index,
        reduction    = 'none',
        label_smoothing = label_smoothing
    )
    pt   = torch.exp(-ce)
    loss = ((1.0 - pt)**gamma) * ce
    valid = targets != ignore_index
    return loss[valid].mean()


# Compute multi-task (CTC + error classification) loss
def compute_multitask_loss(
    model, batch, alpha=1.0, device="cpu",
    BLANK_ID=0,
    stop_ctc=False, stop_error=False
):
    (
        padded_feats, feat_lengths,
        padded_ctc, ctc_lens,
        padded_canonical, canonical_lengths,
        padded_canonical_err, _
    ) = batch

    # Move data to device
    padded_feats = padded_feats.to(device)
    feat_lengths = feat_lengths.to(device)
    padded_ctc = padded_ctc.to(device)
    ctc_lens = ctc_lens.to(device)
    padded_canonical = padded_canonical.to(device)
    padded_canonical_err = padded_canonical_err.to(device)

    # Forward pass
    out = model(
        padded_feats, feat_lengths,
        canonical_ids=padded_canonical,
        canonical_lengths=canonical_lengths
    )
    ctc_logits, error_logits = out["ctc_logits"], out["error_logits"]

    # CTC loss
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_ID, reduction='mean', zero_infinity=True)
    ctc_logits_tbc = ctc_logits.permute(1, 0, 2)
    loss_ctc = ctc_loss_fn(ctc_logits_tbc, padded_ctc, feat_lengths, ctc_lens)
    if stop_ctc:
        loss_ctc = torch.tensor(0.0, device=device)

    # Error classification loss
    if error_logits is not None and not stop_error:
        B, L, _ = error_logits.shape
        err_logits = error_logits.view(B * L, -1)
        err_targets = padded_canonical_err.view(B * L)
        loss_error = focal_loss(
                  err_logits,
                  err_targets,
                  gamma = 2.0,
              )

    else:
        loss_error = torch.tensor(0.0, device=device)

    total_loss = loss_ctc + alpha * loss_error
    return total_loss, loss_ctc.item(), loss_error.item()


# Training loop for one epoch
def train_one_epoch(
    model, dataloader, optimizer, scheduler,
    epoch, warmup_epochs, lambda_max,
    device="cpu", grad_clip=5.0, class_weights=None,
    log_f=None, stop_ctc=False, stop_error=False
):
    model.train()
    total_loss, total_ctc_loss, total_err_loss = 0.0, 0.0, 0.0
    alpha = lambda_max * min(epoch / warmup_epochs, 1.0)

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        loss, l_ctc, l_err = compute_multitask_loss(
            model, batch, alpha, device, class_weights,
            stop_ctc=stop_ctc, stop_error=stop_error
        )
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

    msg = (
        f"Training: Total Loss: {avg_loss:.4f}, "
        f"CTC Loss: {avg_ctc:.4f}, "
        f"ErrorCls Loss: {avg_err:.4f}, λ: {alpha:.2f}"
    )
    if stop_ctc:
        msg += " | Note: CTC task training is stopped."
    if stop_error:
        msg += " | Note: Error classification task training is stopped."

    print(msg)
    if log_f:
        log_f.info(msg)

    return avg_loss, avg_ctc, avg_err


# Validation loop for one epoch
def validate_one_epoch(
    model, dataloader, device="cpu", alpha=1.0,
    class_weights=None, log_f=None,
    stop_ctc=False, stop_error=False
):
    model.eval()
    total_loss, total_ctc_loss, total_err_loss = 0.0, 0.0, 0.0

    gold_err, pred_err = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            loss, l_ctc, l_err = compute_multitask_loss(
                model, batch, alpha, device, class_weights,
                stop_ctc=stop_ctc, stop_error=stop_error
            )
            total_loss += loss.item()
            total_ctc_loss += l_ctc
            total_err_loss += l_err

            if not stop_error:
                (
                    padded_feats, feat_lengths,
                    _, _,
                    padded_canonical, canonical_lengths,
                    padded_canonical_err, _
                ) = batch

                out = model(
                    padded_feats.to(device), feat_lengths.to(device),
                    canonical_ids=padded_canonical.to(device),
                    canonical_lengths=canonical_lengths
                )
                err_logits = out["error_logits"]
                if err_logits is not None:
                    mask = (padded_canonical_err != -100)
                    pred_err.extend(err_logits.argmax(-1)[mask].cpu().tolist())
                    gold_err.extend(padded_canonical_err[mask].cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    avg_ctc = total_ctc_loss / len(dataloader)
    avg_err = total_err_loss / len(dataloader)

    msg = (
        f"Validation: Total Loss: {avg_loss:.4f}, "
        f"CTC Loss: {avg_ctc:.4f}, "
        f"ErrorCls Loss: {avg_err:.4f}"
    )
    if stop_ctc:
        msg += " | Note: CTC task validation is stopped."
    if stop_error:
        msg += " | Note: Error classification task validation is stopped."

    print(msg)
    if log_f:
        log_f.info(msg)

    if gold_err:
        cm = confusion_matrix(gold_err, pred_err, labels=[0, 1, 2])
        cm_str = "\nConfusion Matrix (rows=gold 0/1/2, cols=pred 0/1/2):\n" + str(cm)
        print(cm_str)
        if log_f:
            log_f.info(cm_str)

    return avg_loss, avg_ctc, avg_err


#  Debug function: decode and print some examples after each epoch
def debug_decode(
    model,
    val_dl,
    ph_map,
    device="cpu",
    blank_id=0,
    logger=None,
    max_samples=2
):
    # Attempt to get the first batch
    try:
        sample_batch = next(iter(val_dl))
    except StopIteration:
        # If val_dl is empty
        if logger:
            logger.info("[DEBUG] Validation dataloader is empty, skipping debug decode.")
        return

    # Unpack the batch
    (
        feats, feat_lens,
        ctc_ref, ctc_ref_lens,
        canonical_ids, canonical_lens,
        canonical_err_labels, sample_indices
    ) = sample_batch

    # Move to device
    feats = feats.to(device)
    feat_lens = feat_lens.to(device)
    canonical_ids = canonical_ids.to(device)

    model.eval()
    with torch.no_grad():
        out = model(
            feats, feat_lens,
            canonical_ids=canonical_ids,
            canonical_lengths=canonical_lens
        )
    ctc_logits = out["ctc_logits"]
    error_logits = out["error_logits"]

    id2phone = {v: k for k, v in ph_map.items()}

    # CTC decode
    recognized_batch = ctc_greedy_decode(ctc_logits, blank_id=blank_id)

    # Decode error classes
    label_map = ["Correct", "Substituted", "Deleted"]
    B = feats.size(0)

    # Print up to 'max_samples' items
    n_print = min(B, max_samples)
    for i in range(n_print):
        # recognized phones
        recognized_ids = recognized_batch[i]
        recognized_phones = [id2phone.get(pid, f"UNK{pid}") for pid in recognized_ids]

        debug_str = f"\n[DEBUG] Sample {i} recognized phones: {' '.join(recognized_phones)}"
        if error_logits is not None:
            pred_err_ids = error_logits[i].argmax(dim=-1).cpu().tolist()
            can_len_i = canonical_lens[i].item()
            can_ids_i = canonical_ids[i, :can_len_i].cpu().tolist()
            can_phones = [id2phone.get(x, f"UNK{x}") for x in can_ids_i]
            error_labels = [
                label_map[e] if e < len(label_map) else f"UNK({e})"
                for e in pred_err_ids[:can_len_i]
            ]
            debug_str += f"\n[DEBUG]    Canonical phones: {can_phones}"
            debug_str += f"\n[DEBUG]    Predicted errors: {error_labels}"

        if logger:
            logger.info(debug_str)
        else:
            print(debug_str)


#  Canonical‑dependency probe
def probe_canonical_dependency(
    model, sample_batch, device="cpu", threshold=1e-3
):
    (
        feats, feat_lens,
        _, _,
        canonical_ids, canonical_lens,
        _, _
    ) = sample_batch

    feats = feats.to(device)
    feat_lens = feat_lens.to(device)
    canonical_ids = canonical_ids.to(device)

    with torch.no_grad():
        out_real = model(
            feats, feat_lens,
            canonical_ids=canonical_ids,
            canonical_lengths=canonical_lens
        )["error_logits"]

        rand_ids = torch.randint_like(canonical_ids, high=model.num_ctc_classes)
        out_rand = model(
            feats, feat_lens,
            canonical_ids=rand_ids,
            canonical_lengths=canonical_lens
        )["error_logits"]

    diff = (out_real - out_rand).abs().mean().item()
    flag = diff < threshold
    return diff, flag


# Argparse for training
def parse_args():
    parser = argparse.ArgumentParser("Train MultiTaskConformer")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--phoneme_map", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--lambda_max", type=float, default=0.5)
    parser.add_argument("--lambda_warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--input_dim", type=int, default=80)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--dim_model", type=int, default=512)
    parser.add_argument("--dim_ff", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_error_classes", type=int, default=3)
    parser.add_argument("--log_file", default="train.log")
    parser.add_argument("--save_model", default="model.pt")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    return parser.parse_args()


# Main training function
def main():
    args = parse_args()

    # Setup logging
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(args.log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Training started with config: {vars(args)}")

    # Load phoneme map
    with open(args.phoneme_map, "r", encoding="utf-8") as f:
        ph_map = json.load(f)

    # Create dataset/dataloader
    train_ds = MultiTaskDataset(args.train_manifest, ph_map, apply_specaug=True)
    val_ds = MultiTaskDataset(args.val_manifest, ph_map, apply_specaug=False)
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Construct the model
    model = MultiTaskConformer(
        input_dim=args.input_dim,
        num_blocks=args.num_blocks,
        dim_model=args.dim_model,
        dim_ff=args.dim_ff,
        num_heads=args.num_heads,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        num_ctc_classes=len(ph_map),
        num_error_classes=args.num_error_classes
    ).to(args.device)

    # Optimizer + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: noam_scheduler(step, args.warmup_steps, args.dim_model)
    )

    # Possibly resume training
    best_val_loss = float("inf")
    stop_ctc = False
    stop_error = False
    start_epoch = 1

    if args.resume_checkpoint is not None and os.path.isfile(args.resume_checkpoint):
        logger.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if "best_val_loss" in checkpoint:
            best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Resumed at epoch {start_epoch}, best_val_loss so far: {best_val_loss}")
    else:
        logger.info("No valid checkpoint found for resuming. Starting fresh training.")

    # Main epoch loop
    overall_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"===== Epoch {epoch} =====")
        start_epoch_time = time.time()

        # 1) Train
        train_loss, train_ctc, train_err = train_one_epoch(
            model, train_dl, optimizer, scheduler,
            epoch, args.lambda_warmup_epochs, args.lambda_max,
            device=args.device, grad_clip=args.grad_clip,
            log_f=logger,
            stop_ctc=stop_ctc, stop_error=stop_error
        )

        # 2) Validate
        val_loss, val_ctc, val_err = validate_one_epoch(
            model, val_dl, device=args.device,
            alpha=args.lambda_max,
            log_f=logger, stop_ctc=stop_ctc, stop_error=stop_error
        )

        # 3) Debug decode step: see partial outputs for one batch
        debug_decode(
            model, val_dl, ph_map,
            device=args.device,
            blank_id=0,
            logger=logger,
            max_samples=2
        )

        try:
            first_batch = next(iter(val_dl))
            diff, flag = probe_canonical_dependency(
                model, first_batch, device=args.device)
            probe_msg = (f"[DEBUG] Canonical‑dependency Δ={diff:.4e} "
                         + ("<‑‑ possible issue" if flag else ""))
            print(probe_msg)
            logger.info(probe_msg)
        except StopIteration:
            pass

        epoch_time = time.time() - start_epoch_time
        logger.info(f"Epoch {epoch} training time: {epoch_time:.2f} seconds")

        # 4) Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = f"{args.save_model}_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, ckpt_path)
            logger.info(f"Checkpoint saved at: {ckpt_path}")

        # 5) Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, args.save_model)
            logger.info(f"Model saved at epoch {epoch} (val loss improved).")
        else:
            logger.info(f"No improvement at epoch {epoch}.")

    overall_time = time.time() - overall_start
    logger.info(f"Total training time: {overall_time:.2f} seconds")
    logger.info("Training finished.")


if __name__ == "__main__":
    import sys
    args = [
        "--train_manifest", r"/content/drive/MyDrive/IRP/Implementation_New/New/preprocessed_data/train_data.json",
        "--val_manifest", r"/content/drive/MyDrive/IRP/Implementation_New/New/preprocessed_data/val_data.json",
        "--phoneme_map", r"/content/drive/MyDrive/IRP/Implementation_New/New/preprocessed_data/phoneme_map.json",
        "--device", "cuda",
        "--epochs", "50",
        "--batch_size", "16",
        "--warmup_steps", "4000",
        "--lambda_max", "0.1",
        "--lambda_warmup_epochs", "10",
        "--grad_clip", "5.0",
        "--input_dim", "40",
        "--num_blocks", "12",
        "--dim_model", "512",
        "--dim_ff", "2048",
        "--num_heads", "8",
        "--kernel_size", "15",
        "--dropout", "0.1",
        "--num_error_classes", "3",
        "--log_file", r"/content/drive/MyDrive/IRP/Final/exp_7/train.txt",
        "--save_model", r"/content/drive/MyDrive/IRP/Final/exp_7/model.pt",
        "--resume_checkpoint", r"/content/drive/MyDrive/IRP/Final/exp_7/model.pt",
    ]
    sys.argv.extend(args)
    main()
