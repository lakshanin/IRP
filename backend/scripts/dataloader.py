#!/usr/bin/env python3
"""
Data loading and augmentation script for multi-task phoneme recognition and error classification.
Includes:
  • SpecAugment for feature masking.
  • Dataset class for loading audio features, phoneme labels, and error tags.
  • Collate function for padding variable-length inputs and targets.
"""

import json
import torch
import random
from torch.utils.data import Dataset


def spec_augment(features: torch.Tensor,
                 max_freq_mask: int = 8,
                 max_time_mask: int = 20) -> torch.Tensor:
    T, F = features.shape

    # Frequency Mask
    freq_mask_width = random.randint(0, max_freq_mask)
    if freq_mask_width > 0 and freq_mask_width < F:
        freq_mask_start = random.randint(0, F - freq_mask_width)
        features[:, freq_mask_start:freq_mask_start + freq_mask_width] = 0

    # Time Mask
    time_mask_width = random.randint(0, max_time_mask)
    if 0 < time_mask_width < T:
        time_mask_start = random.randint(0, T - time_mask_width)
        features[time_mask_start:time_mask_start + time_mask_width, :] = 0

    return features


class MultiTaskDataset(Dataset):

    def __init__(self,
                 manifest_path: str,
                 phoneme_map: dict,
                 apply_specaug: bool = False,
                 max_freq_mask: int = 6,
                 max_time_mask: int = 10):

        super().__init__()
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.phoneme_map = phoneme_map
        self.samples = []
        self.apply_specaug = apply_specaug
        self.max_freq_mask = max_freq_mask
        self.max_time_mask = max_time_mask

        for entry in self.data:
            # Audio features: shape (T, feat_dim) as a list of lists
            feats = entry["features"]

            # Actual phone IDs (for CTC)
            actual_phs = entry.get("actual_phonemes", [])
            actual_ids = [self.phoneme_map[p] for p in actual_phs if p in self.phoneme_map]

            # Canonical phone IDs
            canonical_phs = entry.get("canonical_phonemes", [])
            canonical_ids = [self.phoneme_map[p] for p in canonical_phs if p in self.phoneme_map]

            # Error labels (0=Correct,1=Subst,2=Deleted)
            error_labels = entry.get("error_labels", [])

            self.samples.append({
                "features": feats,
                "actual_ids": actual_ids,
                "canonical_ids": canonical_ids,
                "canonical_error_labels": error_labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Convert to torch.Tensor
        feats = torch.tensor(item["features"], dtype=torch.float)
        feats = (feats - feats.mean(dim=0)) / (feats.std(dim=0) + 1e-5)
        actual_ids = torch.tensor(item["actual_ids"], dtype=torch.long)
        canonical_ids = torch.tensor(item["canonical_ids"], dtype=torch.long)
        canonical_err = torch.tensor(item["canonical_error_labels"], dtype=torch.long)

        # Apply SpecAugment if enabled
        if self.apply_specaug:
            feats = spec_augment(feats,
                                 max_freq_mask=self.max_freq_mask,
                                 max_time_mask=self.max_time_mask)

        return feats, actual_ids, canonical_ids, canonical_err


def collate_fn(batch):
    feats_list, ctc_list, canon_list, canon_err_list = zip(*batch)

    # Audio features
    feat_lengths = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
    maxT = max(feat_lengths)
    feat_dim = feats_list[0].shape[1] if len(feats_list) > 0 else 40
    padded_feats = torch.zeros(len(feats_list), maxT, feat_dim, dtype=torch.float)
    for i, f in enumerate(feats_list):
        padded_feats[i, :f.shape[0], :] = f

    # CTC sequences
    ctc_lengths = torch.tensor([len(x) for x in ctc_list], dtype=torch.long)
    maxC = max(ctc_lengths) if ctc_lengths.numel() > 0 else 0
    padded_ctc = torch.zeros(len(ctc_list), maxC, dtype=torch.long)
    for i, seq in enumerate(ctc_list):
        seq_tensor = torch.as_tensor(seq, dtype=torch.long)
        padded_ctc[i, :seq_tensor.shape[0]] = seq_tensor

    # Canonical sequences
    canonical_lengths = torch.tensor([len(x) for x in canon_list], dtype=torch.long)
    maxL = max(canonical_lengths) if canonical_lengths.numel() > 0 else 0
    padded_canonical = torch.zeros(len(canon_list), maxL, dtype=torch.long)
    for i, seq in enumerate(canon_list):
        seq_tensor = torch.as_tensor(seq, dtype=torch.long)
        padded_canonical[i, :seq_tensor.shape[0]] = seq_tensor

    # Error labels
    padded_canonical_err = torch.full((len(canon_err_list), maxL), -100, dtype=torch.long)
    for i, seq in enumerate(canon_err_list):
        seq_tensor = torch.as_tensor(seq, dtype=torch.long)
        target_len = canonical_lengths[i].item()
        # pad or truncate
        if seq_tensor.shape[0] < target_len:
            pad_len = target_len - seq_tensor.shape[0]
            seq_tensor = torch.cat([seq_tensor, torch.full((pad_len,), -100, dtype=torch.long)])
        elif seq_tensor.shape[0] > target_len:
            seq_tensor = seq_tensor[:target_len]
        padded_canonical_err[i, :target_len] = seq_tensor

    return (
        padded_feats,
        feat_lengths,
        padded_ctc,
        ctc_lengths,
        padded_canonical,
        canonical_lengths,
        padded_canonical_err,
        canonical_lengths
    )
