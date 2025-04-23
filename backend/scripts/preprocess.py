#!/usr/bin/env python3
"""
Generates the minimal data needed:
  - features (audio frames)
  - actual_phonemes (from TextGrid)
  - canonical_phonemes (from transcript)
  - error_labels for each canonical phone
"""

import os
import json
import random
import librosa
import numpy as np
from g2p_en import G2p
import tgt
import nltk
nltk.download('averaged_perceptron_tagger_eng')
  
# Paths and parameters.
all_data_dir       = r"/content/drive/MyDrive/IRP/Implementation_New/L2-ARCTIC"
output_dir         = r"/content/drive/MyDrive/IRP/Final/preprocessed_data"
partial_save_dir   = os.path.join(output_dir, "partials")

os.makedirs(partial_save_dir, exist_ok=True)

sample_rate        = 16000
SPEAKERS_PER_BATCH = 5

train_ratio        = 0.8
val_ratio          = 0.1
test_ratio         = 0.1

random_seed        = 42

# Mel-spec settings
N_MELS             = 40
HOP_LENGTH         = 512
WIN_LENGTH         = 1024

# G2P object for transcripts
g2p = G2p()

# Skip these "phones"
UNWANTED_PHONES = {"sil", "sp", "spn", ""}

# Remove trailing numeric stress, e.g. "AA1" -> "AA"
STRESS_LEVELS = {"0","1","2"}


def strip_stress(phoneme: str) -> str:
    if phoneme and phoneme[-1] in STRESS_LEVELS:
        return phoneme[:-1]
    return phoneme


# DTW alignment cost: match if same base phone, else cost=2; insertion/deletion cost=1
def phoneme_cost(p1: str, p2: str) -> float:
    return 0.0 if strip_stress(p1) == strip_stress(p2) else 2.0


def improved_dtw_align(actual_phones, canonical_phones):
    nC = len(canonical_phones)
    nA = len(actual_phones)

    import math
    dp = [[math.inf]*(nA+1) for _ in range(nC+1)]
    backptr = [[None]*(nA+1) for _ in range(nC+1)]
    dp[0][0] = 0

    def cost_fn(cph, aph):
        if cph == "N/A" and aph != "N/A":
            return 1.0
        elif aph == "N/A" and cph != "N/A":
            return 1.0
        else:
            return phoneme_cost(cph, aph)

    # Fill top row / left column
    for j in range(1, nA+1):
        dp[0][j] = dp[0][j-1] + cost_fn("N/A", actual_phones[j-1])
        backptr[0][j] = ("I", 0, j-1)
    for i in range(1, nC+1):
        dp[i][0] = dp[i-1][0] + cost_fn(canonical_phones[i-1], "N/A")
        backptr[i][0] = ("D", i-1, 0)

    # Fill DP
    for i in range(1, nC+1):
        for j in range(1, nA+1):
            cph = canonical_phones[i-1]
            aph = actual_phones[j-1]
            s_cost = dp[i-1][j-1] + cost_fn(cph, aph)
            i_cost = dp[i][j-1]   + cost_fn("N/A", aph)
            d_cost = dp[i-1][j]   + cost_fn(cph, "N/A")
            if s_cost <= i_cost and s_cost <= d_cost:
                dp[i][j] = s_cost
                backptr[i][j] = ("S", i-1, j-1)
            elif i_cost <= d_cost:
                dp[i][j] = i_cost
                backptr[i][j] = ("I", i, j-1)
            else:
                dp[i][j] = d_cost
                backptr[i][j] = ("D", i-1, j)

    # Reconstruct
    alignment = []
    i, j = nC, nA
    while i>0 or j>0:
        op, i2, j2 = backptr[i][j]
        if op=="S":
            cph = canonical_phones[i-1]
            aph = actual_phones[j-1]
            label = "Correct" if strip_stress(cph)==strip_stress(aph) else "Substituted"
            alignment.append((cph, aph, label))
        elif op=="I":
            cph = "N/A"
            aph = actual_phones[j-1]
            alignment.append((cph, aph, "Inserted"))
        elif op=="D":
            cph = canonical_phones[i-1]
            aph = "N/A"
            alignment.append((cph, aph, "Deleted"))
        i, j = i2, j2

    alignment.reverse()
    return alignment


def parse_textgrid_get_boundaries(textgrid_path, sr=16000, hop_len=512):
    if not os.path.exists(textgrid_path):
        return []
    try:
        tg_obj = tgt.read_textgrid(textgrid_path)
        phone_tier = tg_obj.get_tier_by_name('phones')
    except Exception as e:
        print(f"Error reading {textgrid_path}: {e}")
        return []
    boundaries = []
    for interval in phone_tier.intervals:
        ph = interval.text.strip()
        if ph in UNWANTED_PHONES:
            continue
        start_sec = interval.start_time
        end_sec   = interval.end_time
        startF = int(start_sec*sr/hop_len)
        endF   = int(end_sec*sr/hop_len)
        if endF>startF:
            boundaries.append({
                "phone": strip_stress(ph),
                "start": startF,
                "end": endF
            })
    return boundaries


def load_transcript_g2p(txt_path):
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    raw_phones = g2p(text)
    phones = [p for p in raw_phones if p.strip() and p!="'"]
    return [strip_stress(p) for p in phones]


def extract_log_mel_spectrogram(audio, sr, n_mels=40, hop_len=512, win_len=1024):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_len,
        win_length=win_len,
        fmin=20,
        fmax=sr//2
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T


def load_wav_file(wav_path, sr=16000):
    audio, sr_new = librosa.load(wav_path, sr=sr)
    return audio, sr_new


# Map label strings -> numeric classes (0=Correct,1=Substituted,2=Deleted)
ERROR_LABEL_MAP = {
    "Correct": 0,
    "Substituted": 1,
    "Deleted": 2
}


def preprocess_speaker(speaker_dir, speaker_id):
    data_entries = []

    wav_dir       = os.path.join(speaker_dir, "wav")
    textgrid_dir  = os.path.join(speaker_dir, "textgrid")
    transcript_dir= os.path.join(speaker_dir, "transcript")

    if not os.path.isdir(wav_dir):
        return data_entries

    for fname in sorted(os.listdir(wav_dir)):
        if not fname.endswith(".wav"):
            continue

        base_id = os.path.splitext(fname)[0]
        file_id = f"{speaker_id}_{base_id}"
        wav_path = os.path.join(wav_dir, fname)

        # Audio => log-mel
        audio, sr_actual = load_wav_file(wav_path, sr=sample_rate)
        log_mel = extract_log_mel_spectrogram(
            audio, sr_actual,
            n_mels=N_MELS,
            hop_len=HOP_LENGTH,
            win_len=WIN_LENGTH
        )
        log_mel_list = log_mel.astype(np.float32).tolist()

        # actual_phonemes from TextGrid
        tg_path = os.path.join(textgrid_dir, base_id + ".TextGrid")
        phone_boundaries = parse_textgrid_get_boundaries(
            tg_path, sr=sample_rate, hop_len=HOP_LENGTH
        )
        actual_seq = [pb["phone"] for pb in phone_boundaries]

        # canonical_phonemes from transcript
        txt_path = os.path.join(transcript_dir, base_id + ".txt")
        canonical_seq = load_transcript_g2p(txt_path)

        # Single-pass DTW => build error_labels for canonical phones only
        error_labels = []
        if len(actual_seq) > 0 or len(canonical_seq) > 0:
            alignment = improved_dtw_align(actual_seq, canonical_seq)
            for (cph, aph, lab) in alignment:
                # skip any inserted phone where canonical is "N/A"
                if cph == "N/A":
                    continue
                mapped = ERROR_LABEL_MAP.get(lab, 0)
                error_labels.append(mapped)
        else:
            # no phones => error_labels are all "Deleted"
            error_labels = [ERROR_LABEL_MAP["Deleted"]]*len(canonical_seq)

        # Safety check
        if len(error_labels)!=len(canonical_seq):
            print(f"Warning: mismatch in lengths for {file_id}. "
                  f"canonical={len(canonical_seq)}, error_labels={len(error_labels)}")

        # Build final entry
        data_entry = {
            "speaker_id": speaker_id,
            "file_id": file_id,
            "wav_path": wav_path,
            "sample_rate": sr_actual,
            "features": log_mel_list,
            "actual_phonemes": actual_seq,
            "canonical_phonemes": canonical_seq,
            "error_labels": error_labels,
        }
        data_entries.append(data_entry)

    return data_entries


def save_partial_data(entries, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Saved partial batch of size {len(entries)} to {filename}")


def load_partial_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def split_data(data_entries, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    total = train_ratio + val_ratio + test_ratio
    if abs(total-1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0!")

    random.seed(seed)
    random.shuffle(data_entries)
    n = len(data_entries)
    train_end = int(n*train_ratio)
    val_end   = train_end + int(n*val_ratio)

    train_data = data_entries[:train_end]
    val_data   = data_entries[train_end:val_end]
    test_data  = data_entries[val_end:]
    return train_data, val_data, test_data


if __name__=="__main__":
    os.makedirs(partial_save_dir, exist_ok=True)
    speakers = sorted([
        d for d in os.listdir(all_data_dir)
        if os.path.isdir(os.path.join(all_data_dir,d))
    ])

    all_partial_files=[]
    start=0
    batch_idx=0

    while start < len(speakers):
        batch_idx += 1
        chunk = speakers[start:start+SPEAKERS_PER_BATCH]
        start += SPEAKERS_PER_BATCH
        print(f"\n=== Processing Batch {batch_idx}: {chunk} ===")
        batch_entries=[]
        for spk in chunk:
            spk_dir = os.path.join(all_data_dir, spk)
            spk_entries = preprocess_speaker(spk_dir, spk)
            batch_entries.extend(spk_entries)

        partial_file = os.path.join(partial_save_dir, f"partial_batch_{batch_idx}.json")
        save_partial_data(batch_entries, partial_file)
        all_partial_files.append(partial_file)

    # Merge partial data
    print("\nMerging partial data from all batches...")
    combined_entries=[]
    for pf in all_partial_files:
        part_data = load_partial_data(pf)
        combined_entries.extend(part_data)

    print(f"Total combined entries: {len(combined_entries)}")
    if not combined_entries:
        print("No data found after merging partials. Exiting.")
        exit(0)

    # Shuffle & split
    train_data, val_data, test_data = split_data(
        combined_entries,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=random_seed
    )

    os.makedirs(output_dir, exist_ok=True)
    train_manifest = os.path.join(output_dir, "train_data.json")
    val_manifest   = os.path.join(output_dir, "val_data.json")
    test_manifest  = os.path.join(output_dir, "test_data.json")

    with open(train_manifest, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved train_data.json with {len(train_data)} entries.")

    with open(val_manifest, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved val_data.json with {len(val_data)} entries.")

    with open(test_manifest, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Saved test_data.json with {len(test_data)} entries.")

    print("\nPreprocessing complete for multi-task (CTC + monotonic alignment) setup!")
    print(f"Train/Val/Test split: {train_ratio}/{val_ratio}/{test_ratio}")
