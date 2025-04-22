#!/usr/bin/env python3
"""
Inference script for phoneme recognition and error classification using a multi-task Conformer model.
Processes a single audio file and optional transcript to:
  • Predict phoneme sequence via CTC decoding.
  • Classify each canonical phoneme as Correct, Substituted, or Deleted.
"""

import torch
import librosa
import numpy as np
from termcolor import colored
import json

from model import MultiTaskConformer

# Parameters
NUM_BLOCKS         = 12
DIM_MODEL          = 512
DIM_FF             = 2048
NUM_HEADS          = 8
KERNEL_SIZE        = 15
DROPOUT            = 0.1
NUM_ERROR_CLASSES  = 3
INPUT_DIM          = 40

BLANK_ID           = 0
SAMPLE_RATE        = 16000
HOP_LENGTH         = 512
WIN_LENGTH         = 1024
DEVICE             = "cpu"

# g2p_en for transcript -> canonical phones
try:
    from g2p_en import G2p
    g2p = G2p()

    def text_to_phonemes(text):
        raw_phones = g2p(text)
        phone_list = [p for p in raw_phones if p.strip() and p != "'"]
        def strip_stress(ph):
            # remove trailing digits like AA1 -> AA
            return ph[:-1] if ph and ph[-1].isdigit() else ph
        return [strip_stress(p) for p in phone_list]

    G2P_AVAILABLE = True
except ImportError:
    # Fallback if g2p_en not installed
    def text_to_phonemes(text):
        return text.strip().split()
    G2P_AVAILABLE = False
    print("[WARNING] g2p_en not installed; transcript_text will be naïvely split.")


# Audio feature extraction
def extract_log_mel_spectrogram(audio, sr,
                                n_mels=INPUT_DIM,
                                hop_length=HOP_LENGTH,
                                win_length=WIN_LENGTH):

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        fmin=20,
        fmax=sr // 2
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.T


# CTC greedy decode
def ctc_greedy_decode(logits_ctc, blank_id=BLANK_ID):
    max_ids = logits_ctc.argmax(dim=-1).squeeze(0)  # shape (T,)
    out_seq = []
    prev = None
    for idx in max_ids.tolist():
        if idx != blank_id and idx != prev:
            out_seq.append(idx)
        prev = idx
    return out_seq


# Main inference function
def run_inference(
    audio_file: str,
    model_checkpoint: str,
    phoneme_map: str,
    transcript_text: str = None
):
    device = torch.device(DEVICE)

    # Load the phoneme map
    with open(phoneme_map, "r", encoding="utf-8") as f:
        ph_map = json.load(f)
    id2phone = {v: k for k, v in ph_map.items()}
    num_ctc_classes = len(ph_map)

    # Construct the model
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

    # Load model checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Load & preprocess audio
    audio, sr_actual = librosa.load(audio_file, sr=SAMPLE_RATE)
    feats = extract_log_mel_spectrogram(audio, sr_actual)
    T = feats.shape[0]

    # Convert to torch, then do normalization
    feats_tensor = torch.tensor(feats, dtype=torch.float)
    feats_tensor = (feats_tensor - feats_tensor.mean(dim=0)) / (feats_tensor.std(dim=0) + 1e-5)
    feats_tensor = feats_tensor.unsqueeze(0).to(device)

    feat_len_tensor = torch.tensor([T], dtype=torch.long).to(device)

    # Prepare canonical IDs if transcript is provided
    canonical_tensor = None
    canonical_len_tensor = None
    canonical_phones = []
    if transcript_text:
        canonical_phones = text_to_phonemes(transcript_text)
        ph_ids = [ph_map.get(p, 0) for p in canonical_phones]
        canonical_tensor = torch.tensor(ph_ids, dtype=torch.long).unsqueeze(0).to(device)
        canonical_len_tensor = torch.tensor([len(ph_ids)], dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        out = model(
            feats_tensor, feat_len_tensor,
            canonical_ids=canonical_tensor,
            canonical_lengths=canonical_len_tensor
        )
    ctc_logits = out["ctc_logits"]
    error_logits = out["error_logits"]

    # CTC decoding
    recognized_ids = ctc_greedy_decode(ctc_logits, blank_id=BLANK_ID)
    recognized_phones = [id2phone.get(pid, f"UNK{pid}") for pid in recognized_ids]

    # Error classification
    label_map = ["Correct", "Substituted", "Deleted"]
    if error_logits is not None:
        pred_errors = error_logits.squeeze(0).argmax(dim=-1).cpu().tolist()
        canonical_error_labels = [
            label_map[i] if i < len(label_map) else f"UNK({i})"
            for i in pred_errors
        ]
    else:
        canonical_error_labels = []

    # Return results
    return {
        "recognized_phones": recognized_phones,
        "canonical_phones": canonical_phones,
        "canonical_error_labels": canonical_error_labels
    }


# Print results
def print_results(results):
    recognized_phones = results["recognized_phones"]
    canonical_phones = results["canonical_phones"]
    canonical_error_labels = results["canonical_error_labels"]

    print("\n=== Recognized Phoneme Sequence (CTC) ===")
    if recognized_phones:
        print(" ".join(recognized_phones))
    else:
        print("(No phones recognized)")

    if canonical_phones and canonical_error_labels:
        print("\n=== Canonical Phones & Inferred Error Labels ===")
        color_map = {
            "Correct": "green",
            "Substituted": "yellow",
            "Deleted": "red"
        }
        for ph, err in zip(canonical_phones, canonical_error_labels):
            c = color_map.get(err, "white")
            print(colored(f"{ph:>10s}: {err}", c))
    else:
        print("\n(No transcript_text => no error classification)")


if __name__ == "__main__":
    AUDIO_FILE       = r"E:\IRP\test_cases\tc\arctic_a0356.wav"
    TRANSCRIPT_TEXT  = "You don't catch me at any such foolishness"
    MODEL_CHECKPOINT = r"E:\IRP\models\model_epoch_115.pt"
    PHONEME_MAP      = r"E:\IRP\backend\data\preprocessed_five\phoneme_map.json"

    result_dict = run_inference(
        audio_file=AUDIO_FILE,
        model_checkpoint=MODEL_CHECKPOINT,
        phoneme_map=PHONEME_MAP,
        transcript_text=TRANSCRIPT_TEXT
    )
    print_results(result_dict)
