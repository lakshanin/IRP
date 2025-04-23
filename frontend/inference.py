#!/usr/bin/env python3
"""
Inference script for phoneme recognition and pronunciation error classification using the Multi-Task Conformer model.

This script processes a single audio file and a corresponding transcript to:
  • Extract log-Mel spectrogram features from the audio.
  • Convert the transcript into a sequence of canonical phonemes using G2P.
  • Predict the phoneme sequence spoken using CTC decoding.
  • Classify each canonical phoneme as Correct, Substituted, or Deleted via a cross-attention mechanism.
  • Output a summary of results and a word-level breakdown of phoneme errors.
"""

import json
import torch
import librosa
import numpy as np
from collections import Counter
from model import MultiTaskConformer

# Parameters
NUM_BLOCKS = 12
DIM_MODEL = 512
DIM_FF = 2048
NUM_HEADS = 8
KERNEL_SIZE = 15
DROPOUT = 0.1
NUM_ERROR_CLASSES = 3
INPUT_DIM = 40
SAMPLE_RATE = 16000
HOP_LENGTH = 512
WIN_LENGTH = 1024
DEFAULT_BLANK_ID = 0

try:
    from g2p_en import G2p

    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    print("[WARNING] g2p_en not installed; falling back to naive split.")


def text_to_phonemes(text: str):
    if not G2P_AVAILABLE:
        return text.strip().split()
    g2p = G2p()
    raw_phones = g2p(text)
    phone_list = [p for p in raw_phones if p.strip() and p != "'"]

    def strip_stress(p):
        return p[:-1] if p and p[-1].isdigit() else p

    return [strip_stress(p) for p in phone_list]


def extract_log_mel_spectrogram(audio, sr, n_mels=INPUT_DIM, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        fmin=20,
        fmax=sr // 2
    )
    return librosa.power_to_db(mel_spec, ref=np.max).T


def run_inference(
        audio_file: str,
        transcript_text: str,
        phoneme_map_path: str,
        model_checkpoint: str,
        device="cpu",
        blank_id=DEFAULT_BLANK_ID
):
    # Build word-level canonical list & boundaries
    words = transcript_text.strip().split()
    all_canonical_phones = []
    word_boundaries = []
    for w in words:
        w_phones = text_to_phonemes(w)
        start_idx = len(all_canonical_phones)
        all_canonical_phones.extend(w_phones)
        word_boundaries.append((w, start_idx, len(w_phones)))

    # Load phoneme map
    with open(phoneme_map_path, "r", encoding="utf-8") as f:
        ph_map = json.load(f)
    id2phone = {v: k for k, v in ph_map.items()}

    # Map canonical to IDs
    canonical_ids = [ph_map.get(ph, blank_id) for ph in all_canonical_phones]
    num_ctc_classes = len(ph_map)

    # Load model
    model = MultiTaskConformer(
        input_dim=INPUT_DIM,
        num_blocks=NUM_BLOCKS,
        dim_model=DIM_MODEL,
        dim_ff=DIM_FF,
        num_heads=NUM_HEADS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        num_ctc_classes=num_ctc_classes,
        num_error_classes=NUM_ERROR_CLASSES
    ).to(device)

    checkpoint = torch.load(model_checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Audio → features
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
    feats = extract_log_mel_spectrogram(audio, SAMPLE_RATE)
    feats_tensor = torch.tensor((feats - feats.mean(0)) / (feats.std(0) + 1e-5), dtype=torch.float)
    feats_tensor = feats_tensor.unsqueeze(0).to(device)
    feat_len = torch.tensor([feats.shape[0]], dtype=torch.long).to(device)

    canon_tensor = torch.tensor(canonical_ids, dtype=torch.long).unsqueeze(0).to(device)
    canon_len = torch.tensor([len(canonical_ids)], dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        out = model(feats_tensor, feat_len, canonical_ids=canon_tensor, canonical_lengths=canon_len)
    ctc_logits = out["ctc_logits"].squeeze(0)
    error_logits = out.get("error_logits")
    if error_logits is not None:
        error_logits = error_logits.squeeze(0)

    # CTC decoding → recognized_phones
    ctc_pred_ids = ctc_logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    prev = None
    for idx in ctc_pred_ids:
        if idx != blank_id and idx != prev:
            decoded.append(idx)
        prev = idx
    recognized_phones = [id2phone.get(i, f"UNK({i})") for i in decoded]

    # Error classification
    label_map = ["Correct", "Substituted", "Deleted"]
    if error_logits is not None and error_logits.shape[-1] == NUM_ERROR_CLASSES:
        pred_ids = error_logits.argmax(dim=-1).cpu().tolist()
        canonical_error_labels = [label_map[i] for i in pred_ids]
    else:
        canonical_error_labels = ["Correct"] * len(all_canonical_phones)

    # Summary stats
    counts = Counter(canonical_error_labels)
    total = len(all_canonical_phones)
    correct = counts["Correct"]
    subs = counts["Substituted"]
    dels = counts["Deleted"]
    accuracy = (correct / total * 100) if total > 0 else 0.0

    summary_str = (
        f"There were {total} target phones across {len(words)} word(s).\n"
        f"  - Correct: {correct}\n"
        f"  - Substituted: {subs}\n"
        f"  - Deleted: {dels}\n\n"
        f"Overall phone-level accuracy: {accuracy:.1f}%\n\n"
    )

    # Word-level breakdown
    word_breakdown = []
    for w, start, length in word_boundaries:
        segment = all_canonical_phones[start:start + length]
        errs = canonical_error_labels[start:start + length]
        recs = recognized_phones[start:start + length]
        word_breakdown.append({
            "word": w,
            "phones": segment,
            "error_labels": errs,
            "recognized": recs
        })

    return {
        "canonical_phones": all_canonical_phones,
        "canonical_error_labels": canonical_error_labels,
        "recognized_phones": recognized_phones,
        "summary": {
            "plain_text": summary_str,
            "num_total_target_phones": total,
            "num_correct": correct,
            "num_substitutions": subs,
            "num_deletions": dels,
            "phone_accuracy_percent": accuracy
        },
        "word_breakdown": word_breakdown
    }
