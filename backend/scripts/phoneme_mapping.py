#!/usr/bin/env python3
"""
Reads dataset JSON manifests, collects all unique phonemes
from the "actual_phonemes" and/or "canonical_phonemes" fields
then writes a phoneme->integer mapping to phoneme_map.json.
"""

import json


def create_phoneme_map(manifest_paths, output_map="phoneme_map.json"):
    all_phonemes = set()

    # Gather all phonemes
    for mpath in manifest_paths:
        with open(mpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:

            actual = entry.get("actual_phonemes", [])
            canonical = entry.get("canonical_phonemes", [])
            # Collect them
            for p in actual:
                all_phonemes.add(p)
            for p in canonical:
                all_phonemes.add(p)

    # Sort them for consistency
    sorted_phonemes = sorted(list(all_phonemes))

    # Build mapping with <BLANK>=0
    phoneme_map = {"<BLANK>": 0}
    idx = 1
    for p in sorted_phonemes:
        phoneme_map[p] = idx
        idx += 1

    # Write out JSON
    with open(output_map, 'w', encoding='utf-8') as f:
        json.dump(phoneme_map, f, indent=2, ensure_ascii=False)
    print(f"Created poneme map with {len(phoneme_map)} entries (including <BLANK>).")
    print(f"Saved to {output_map}.")


def main():
    manifests = [r"/content/drive/MyDrive/IRP/Final/preprocessed_data/train_data.json", 
                 r"/content/drive/MyDrive/IRP/Final/preprocessed_data/test_data.json",
                 r"/content/drive/MyDrive/IRP/Final/preprocessed_data/val_data.json"]
    output_map = r"/content/drive/MyDrive/IRP/Final/preprocessed_data/phoneme_map.json"

    create_phoneme_map(manifests, output_map)


if __name__ == "__main__":
    main()
